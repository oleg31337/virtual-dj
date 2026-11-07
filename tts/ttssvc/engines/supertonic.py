"""Supertonic 3 engine adapter (ONNX Runtime, CPU, 44.1 kHz).

The model bundle (HF ``Supertone/supertonic-3``) is one model with ten speaker
*styles*; the language is a per-request tag, not a property of the style, so
every voice speaks both advertised languages. Quality is fixed at
``TTS_STEPS`` (5 by default) because the 2-step fast mode garbles words.

Everything engine-specific stays in this file — the HTTP layer only sees the
contract in ``base.py``.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import supertonic_helper as helper
from .base import Engine, EngineError

log = logging.getLogger(__name__)

REPO_ID = "Supertone/supertonic-3"
ALLOW_PATTERNS = ["config.json", "README.md", "onnx/*", "voice_styles/*"]
REQUIRED_ONNX = ("duration_predictor.onnx", "text_encoder.onnx",
                 "vector_estimator.onnx", "vocoder.onnx")

_LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "uk": "Ukrainian", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "pl": "Polish",
    "pt": "Portuguese", "tr": "Turkish", "nl": "Dutch", "cs": "Czech",
    "ko": "Korean", "ja": "Japanese", "ar": "Arabic", "hi": "Hindi",
}

# The ten speaker styles bundled with the model. Names/notes are for the GUI;
# the model itself only knows the id.
_VOICES: list[dict[str, Any]] = [
    {"id": "M1", "name": "Male 1", "gender": "male",
     "description": "Neutral, even-paced male voice."},
    {"id": "M2", "name": "Male 2", "gender": "male",
     "description": "Warm male voice — the best Russian of the ten styles.",
     "note": "Recommended for Russian."},
    {"id": "M3", "name": "Male 3", "gender": "male",
     "description": "Brighter, younger-sounding male voice."},
    {"id": "M4", "name": "Male 4", "gender": "male",
     "description": "Lower, calmer male voice."},
    {"id": "M5", "name": "Male 5", "gender": "male",
     "description": "Crisper male voice with more attack."},
    {"id": "F1", "name": "Female 1", "gender": "female",
     "description": "Clear, friendly female voice."},
    {"id": "F2", "name": "Female 2", "gender": "female",
     "description": "Softer, lower female voice."},
    {"id": "F3", "name": "Female 3", "gender": "female",
     "description": "Bright female voice."},
    {"id": "F4", "name": "Female 4", "gender": "female",
     "description": "Measured, newsreader-like female voice."},
    {"id": "F5", "name": "Female 5", "gender": "female",
     "description": "Warmer female voice."},
]

# Characters the model's Unicode indexer has no embedding for (index -1).
# Writing them through would index -1 and corrupt the render, so they are mapped
# to something speakable first. Verified against onnx/unicode_indexer.json.
_CHAR_MAP = {
    "\u2014": "-",   # em dash            (-1 in the indexer)
    "\u2013": "-",   # en dash
    "\u2011": "-",   # non-breaking hyphen(-1)
    "\u2026": ".",   # ellipsis           (-1)
    "\u00a0": " ",   # nbsp
    "\u2009": " ", "\u200a": " ", "\u202f": " ", "\u200b": "",
    "\u00b4": "", "\u02b9": "", "\u0301": "", "\u0300": "", "\u0341": "",
    "`": "'", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2116": "N",   # numero sign
}

_PARAMS: list[dict[str, Any]] = [
    {
        "key": "speed", "label": "Speaking speed", "type": "float",
        "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "unit": "\u00d7",
        "help": "1.0 is the model's natural pace; below 1.0 is slower.",
        "group": "prosody",
    },
    {
        "key": "sentence_pause", "label": "Pause after each sentence",
        "type": "float", "min": 0.0, "max": 2.0, "step": 0.05,
        "default": 0.3, "unit": "s",
        "help": "Silence inserted between sentence chunks (the model's own "
                "punctuation pauses are short).",
        "group": "prosody",
    },
]


class Supertonic3Engine(Engine):
    id = "supertonic3"
    name = "Supertonic 3"
    version = "3"
    license = "openrail"
    runtime = "onnxruntime"

    def __init__(self, config) -> None:  # noqa: ANN001 - config module
        super().__init__(config)
        self._lock = threading.Lock()
        self._tts = None
        self._indexer: list[int] | None = None
        self._styles: dict[str, Any] = {}
        self._load_error: str | None = None
        self._root = config.model_dir() / "supertonic3"

    # --- paths ------------------------------------------------------------
    @property
    def onnx_dir(self) -> Path:
        return self._root / "onnx"

    @property
    def styles_dir(self) -> Path:
        return self._root / "voice_styles"

    def _files_present(self) -> bool:
        if not self.onnx_dir.is_dir() or not self.styles_dir.is_dir():
            return False
        if not all((self.onnx_dir / n).exists() for n in REQUIRED_ONNX):
            return False
        return bool(list(self.styles_dir.glob("*.json")))

    # --- metadata ---------------------------------------------------------
    def languages(self) -> list[dict[str, str]]:
        return [{"code": c, "name": _LANGUAGE_NAMES.get(c, c.upper())}
                for c in self.advertised_languages]

    def voices(self) -> list[dict[str, Any]]:
        installed = self.model_ready()[0] and self._files_present()
        out = []
        for v in _VOICES:
            entry = dict(v)
            entry["languages"] = list(self.advertised_languages)
            entry["default_language"] = (self.config.default_language()
                                         or self.advertised_languages[0])
            entry["installed"] = installed and (
                self.styles_dir / f"{v['id']}.json").exists()
            out.append(entry)
        return out

    def params(self) -> list[dict[str, Any]]:
        return [dict(p) for p in _PARAMS]

    def text_support(self) -> dict[str, bool]:
        # No espeak: a combining acute is just an odd glyph to this model.
        return {"stress_marks": False, "ssml": False, "phonemes": False}

    def sample_rate(self) -> int:
        return int(self._tts.sample_rate) if self._tts is not None else 44100

    # --- lifecycle --------------------------------------------------------
    def ensure_model(self, progress: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if self._tts is not None:
                return
            if not self._files_present():
                if not self.config.autodownload():
                    raise EngineError(
                        f"model files are missing under {self._root} and "
                        "TTS_AUTODOWNLOAD is off")
                self._download(progress)
            self._load()

    def _download(self, progress: Callable[[str], None] | None = None) -> None:
        say = progress or (lambda msg: log.info("%s", msg))
        say(f"downloading {REPO_ID} -> {self._root}")
        from huggingface_hub import snapshot_download

        self._root.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=REPO_ID, local_dir=str(self._root),
                              allow_patterns=ALLOW_PATTERNS)
        except Exception as exc:  # noqa: BLE001 - reported through model_ready()
            raise EngineError(f"model download failed: {exc}") from exc
        if not self._files_present():
            raise EngineError("model download finished but files are missing")

    def _load(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = self.config.threads()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = max(1, threads // 2)
        providers = ["CPUExecutionProvider"]
        try:
            dp, text_enc, vector_est, vocoder = helper.load_onnx_all(
                str(self.onnx_dir), opts, providers)
            cfgs = helper.load_cfgs(str(self.onnx_dir))
            processor = helper.load_text_processor(str(self.onnx_dir))
            self._tts = helper.TextToSpeech(
                cfgs, processor, dp, text_enc, vector_est, vocoder)
            import json
            self._indexer = json.loads(
                (self.onnx_dir / "unicode_indexer.json").read_text())
            self.runtime = f"onnxruntime {ort.__version__}"
            self._load_error = None
            log.info("Supertonic 3 loaded from %s (steps=%d, sr=%d)",
                     self._root, self.config.steps(), self._tts.sample_rate)
        except Exception as exc:  # noqa: BLE001 - surfaced via model_ready()
            self._tts = None
            self._load_error = f"model failed to load: {exc}"
            log.error("%s", self._load_error)
            raise EngineError(self._load_error) from exc

    def model_ready(self) -> tuple[bool, str | None]:
        if self._tts is not None:
            return True, None
        if self._load_error:
            return False, self._load_error
        if not self._files_present():
            return False, (f"model files are missing under {self._root}; "
                           "they download on first use")
        return False, "model not loaded yet"

    def install_voice(self, voice_id: str) -> dict[str, Any]:
        if voice_id not in {v["id"] for v in _VOICES}:
            raise KeyError(voice_id)
        already = self.model_ready()[0] and (
            self.styles_dir / f"{voice_id}.json").exists()
        self.ensure_model()
        size = sum(p.stat().st_size for p in self._root.rglob("*") if p.is_file())
        return {"voice": voice_id, "installed": True, "already": already,
                "bytes": size}

    # --- text cleaning ----------------------------------------------------
    def _sanitize(self, text: str) -> str:
        """Map characters the model cannot index, keep everything else."""
        indexer = self._indexer
        if indexer is None:  # pragma: no cover - only before load
            return text
        out = []
        for ch in text:
            ch = _CHAR_MAP.get(ch, ch)
            for part in ch:
                if part.isspace():
                    out.append(" ")
                    continue
                code = ord(part)
                if code < len(indexer) and indexer[code] >= 0:
                    out.append(part)
                else:
                    log.debug("dropping character %r (not in the indexer)", part)
                    out.append(" ")
        cleaned = "".join(out)
        # Collapse the spaces the substitutions leave behind.
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        return cleaned.strip()

    # --- the verb ---------------------------------------------------------
    def synth_style(self, voice: str):
        if voice not in self._styles:
            path = self.styles_dir / f"{voice}.json"
            if not path.exists():
                raise KeyError(voice)
            self._styles[voice] = helper.load_voice_style([str(path)])
        return self._styles[voice]

    def synthesize(self, text: str, voice: str, language: str,
                   params: dict[str, Any]) -> tuple[np.ndarray, int]:
        ok, why = self.model_ready()
        if not ok or self._tts is None:
            try:
                self.ensure_model()
            except EngineError:
                raise
        if self._tts is None:  # pragma: no cover - defensive
            raise EngineError(why or "engine not loaded")
        if language not in self.advertised_languages:
            raise ValueError(f"unsupported language {language!r}")
        clean = self._sanitize(text)
        if not clean:
            raise ValueError("text is empty after cleaning")
        style = self.synth_style(voice)
        speed = float(params.get("speed", 1.0) or 1.0)
        pause = float(params.get("sentence_pause", 0.3) or 0.0)
        wav, _dur = self._tts(clean, language, style, self.config.steps(),
                              speed, silence_duration=pause)
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        return samples, int(self._tts.sample_rate)


__all__ = ["Supertonic3Engine", "REPO_ID"]
