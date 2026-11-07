"""Piper engine adapter (espeak-ng phonemes + VITS, CPU).

Kept as the second engine on purpose: it proves the HTTP contract is
engine-agnostic, and it is the fallback voice if the ONNX diffusion engine ever
misbehaves on a host. It exposes the four knobs the app used to hardcode
(speed, expressiveness, rhythm, sentence pause), so the DJ card renders
identically from the schema while the engine behind it is a different one.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import Engine, EngineError

log = logging.getLogger(__name__)

HF_ROOT = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

_LANGUAGE_NAMES = {"en": "English", "ru": "Russian"}

_VOICES: list[dict[str, Any]] = [
    {"id": "en_GB-alan-medium", "name": "UK Alan", "gender": "male",
     "languages": ["en"], "description": "British male, steady and clear."},
    {"id": "en_US-ryan-medium", "name": "US Ryan", "gender": "male",
     "languages": ["en"], "description": "American male, warm delivery."},
    {"id": "en_US-amy-medium", "name": "US Amy", "gender": "female",
     "languages": ["en"], "description": "American female, neutral."},
    {"id": "ru_RU-irina-medium", "name": "Irina", "gender": "female",
     "languages": ["ru"], "description": "Russian female, the station's "
     "long-standing default."},
    {"id": "ru_RU-denis-medium", "name": "Denis", "gender": "male",
     "languages": ["ru"], "description": "Russian male."},
    {"id": "ru_RU-dmitri-medium", "name": "Dmitri", "gender": "male",
     "languages": ["ru"], "description": "Russian male, deeper."},
    {"id": "ru_RU-ruslan-medium", "name": "Ruslan", "gender": "male",
     "languages": ["ru"], "description": "Russian male, brighter."},
]

_PARAMS: list[dict[str, Any]] = [
    {"key": "speed", "label": "Speaking speed", "type": "float",
     "min": 0.7, "max": 1.4, "step": 0.01, "default": 1.0, "unit": "\u00d7",
     "help": "Mapped to Piper's length_scale (inverse).", "group": "prosody"},
    {"key": "expressiveness", "label": "Expression", "type": "float",
     "min": 0.1, "max": 1.5, "step": 0.005, "default": 0.667,
     "help": "Piper noise_scale: lower is flatter/robotic, higher is livelier.",
     "group": "prosody"},
    {"key": "rhythm", "label": "Rhythm variation", "type": "float",
     "min": 0.0, "max": 1.5, "step": 0.005, "default": 0.8,
     "help": "Piper noise_w: timing jitter — lower is metronomic.",
     "group": "prosody"},
    {"key": "sentence_pause", "label": "Pause after each sentence",
     "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.25,
     "unit": "s", "help": "Silence after each detected sentence.",
     "group": "prosody"},
]


def _voice_urls(voice: str) -> tuple[str, str]:
    parts = voice.split("-")
    if len(parts) != 3:
        raise ValueError(f"not a Piper voice id: {voice!r}")
    country_code, name, quality = parts
    lang = country_code[:2]
    base = f"{HF_ROOT}/{lang}/{country_code}/{name}/{quality}"
    return f"{base}/{voice}.onnx", f"{base}/{voice}.onnx.json"


class PiperEngine(Engine):
    id = "piper"
    name = "Piper"
    version = "1.2"
    license = "MIT (engine) / CC-BY voices"
    runtime = "onnxruntime"

    def __init__(self, config) -> None:  # noqa: ANN001 - config module
        super().__init__(config)
        self._lock = threading.Lock()
        self._root = config.model_dir() / "piper" / "voices"

    # --- paths / binary ---------------------------------------------------
    @property
    def bin_path(self) -> str | None:
        local = Path("/usr/local/bin/piper")
        if local.exists():
            return str(local)
        return shutil.which("piper")

    def model_path(self, voice: str) -> Path:
        return self._root / f"{voice}.onnx"

    def _installed(self, voice: str) -> bool:
        return (self.model_path(voice).exists()
                and self._root.joinpath(f"{voice}.onnx.json").exists())

    # --- metadata ---------------------------------------------------------
    def languages(self) -> list[dict[str, str]]:
        seen = {code for v in _VOICES for code in v["languages"]}
        return [{"code": c, "name": _LANGUAGE_NAMES.get(c, c.upper())}
                for c in self.advertised_languages if c in seen]

    def voices(self) -> list[dict[str, Any]]:
        out = []
        for v in _VOICES:
            langs = [c for c in v["languages"] if c in self.advertised_languages]
            if not langs:
                continue
            entry = dict(v)
            entry["languages"] = langs
            entry["default_language"] = langs[0]
            entry["installed"] = self._installed(v["id"])
            out.append(entry)
        return out

    def params(self) -> list[dict[str, Any]]:
        return [dict(p) for p in _PARAMS]

    def text_support(self) -> dict[str, bool]:
        # espeak-ng obeys the combining acute the app emits for Russian.
        return {"stress_marks": True, "ssml": False, "phonemes": True}

    def sample_rate(self) -> int:
        for v in _VOICES:
            cfg = self._root / f"{v['id']}.onnx.json"
            if cfg.exists():
                try:
                    return int(json.loads(cfg.read_text())
                               ["audio"]["sample_rate"])
                except (KeyError, ValueError, OSError):
                    continue
        return 22050

    # --- lifecycle --------------------------------------------------------
    def ensure_model(self, progress: Callable[[str], None] | None = None) -> None:
        if not self.bin_path:
            raise EngineError("the piper executable is missing from this image")

    def model_ready(self) -> tuple[bool, str | None]:
        if not self.bin_path:
            return False, "the piper executable is missing from this image"
        if not any(self._installed(v["id"]) for v in _VOICES):
            return False, f"no Piper voice is installed under {self._root}"
        return True, None

    def _download(self, url: str, dest: Path, timeout: float = 300.0) -> None:
        import httpx

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", url, timeout=timeout,
                          follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
        tmp.replace(dest)

    def install_voice(self, voice_id: str) -> dict[str, Any]:
        known = {v["id"] for v in _VOICES}
        if voice_id not in known:
            raise KeyError(voice_id)
        if self._installed(voice_id):
            return {"voice": voice_id, "installed": True, "already": True,
                    "bytes": self.model_path(voice_id).stat().st_size}
        onnx_url, json_url = _voice_urls(voice_id)
        log.info("downloading Piper voice %s", voice_id)
        with self._lock:
            self._download(onnx_url, self.model_path(voice_id))
            size = self.model_path(voice_id).stat().st_size
            if size < 1_000_000:
                self.model_path(voice_id).unlink(missing_ok=True)
                raise EngineError(
                    f"voice {voice_id} downloaded truncated ({size} bytes)")
            self._download(json_url, self._root / f"{voice_id}.onnx.json")
        return {"voice": voice_id, "installed": True, "already": False,
                "bytes": size}

    # --- the verb ---------------------------------------------------------
    def synthesize(self, text: str, voice: str, language: str,
                   params: dict[str, Any]) -> tuple[np.ndarray, int]:
        binary = self.bin_path
        if not binary:
            raise EngineError("the piper executable is missing from this image")
        if not self._installed(voice):
            raise EngineError(f"voice {voice} is not installed")
        speed = max(0.5, min(float(params.get("speed", 1.0) or 1.0), 2.0))
        noise_scale = float(params.get("expressiveness", 0.667) or 0.667)
        noise_w = float(params.get("rhythm", 0.8))
        pause = float(params.get("sentence_pause", 0.25) or 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.wav"
            cmd = [binary, "-m", str(self.model_path(voice)), "-f", str(out),
                   "--length_scale", str(round(1.0 / speed, 4)),
                   "--noise_scale", str(round(noise_scale, 4)),
                   "--noise_w_scale", str(round(noise_w, 4)),
                   "--sentence_silence", str(round(pause, 3))]
            proc = subprocess.run(cmd, input=text.encode("utf-8"),
                                  capture_output=True, timeout=300, check=False)
            if proc.returncode != 0 or not out.exists():
                raise EngineError(
                    "piper failed: "
                    + proc.stderr.decode("utf-8", "replace")[:400])
            with wave.open(str(out), "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, int(rate)


__all__ = ["PiperEngine"]
