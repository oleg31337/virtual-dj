"""Voice model management for the DJ.

Piper voice models are large (~80-110 MB each) binary files that live in
``data/voices/`` and are intentionally git-ignored. A fresh clone therefore
ships with **no** voices, and the DJ cannot speak until at least the default
English voice (and, for Russian tracks, the default Russian voice) is present.

This module downloads voices from the public ``rhasspy/piper-voices`` Hugging
Face repo on demand. Two paths are supported:

* ``ensure_default_voices()`` -- called at startup; downloads the two default
  voices the app needs to function (``dj.voice`` and ``dj.russian_voice`` from
  config) if they are missing. Best-effort and never fatal: if there is no
  network at boot, the app still starts (the DJ just stays silent until a
  voice is fetched later).
* ``download_voice(name)`` / ``download_voices(names)`` -- fetch specific
  voices, used by the CLI (``python -m app.voices``) and the on-demand
  ``POST /api/dj/voices/download`` endpoint so the web UI can grab any voice
  the user picks.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

import httpx

from . import config

log = logging.getLogger(__name__)

_HF_ROOT = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# A voice id is "<lang>_<country>-<name>-<quality>", e.g. "en_US-amy-medium".
def _voice_urls(voice: str) -> tuple[str, str]:
    """Return (onnx_url, json_url) for a Piper voice id.

    A voice id is ``<lang>_<country>-<name>-<quality>`` (e.g.
    ``en_US-amy-medium``). The HuggingFace ``rhasspy/piper-voices`` layout is
    ``<lang>/<lang>_<country>/<name>/<quality>/<id>.onnx[.json]``, so for
    ``en_US-amy-medium`` that is ``en/en_US/amy/medium/en_US-amy-medium.onnx``.
    """
    parts = voice.split("-")
    if len(parts) != 3:
        raise ValueError(f"not a Piper voice id: {voice!r}")
    country_code, name, quality = parts
    lang = country_code[:2]
    base = f"{_HF_ROOT}/{lang}/{country_code}/{name}/{quality}"
    return f"{base}/{voice}.onnx", f"{base}/{voice}.onnx.json"


def _download_file(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream ``url`` to ``dest`` (atomic via a temp file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def voice_model_path(voice: str | None = None) -> Path | None:
    """Absolute path to ``voice``'s ``.onnx`` if present, else ``None``."""
    name = voice or config.get("dj.voice", "en_US-amy-medium")
    voices_dir = Path(config.get("voices_dir", config.VOICES_DIR))
    path = voices_dir / f"{name}.onnx"
    return path if path.exists() else None


def download_voice(voice: str, timeout: float = 180.0) -> Path:
    """Download ``voice`` (onnx + json) into ``data/voices/``.

    Raises on network/HTTP failure. Returns the path to the ``.onnx`` file.
    """
    onnx_url, json_url = _voice_urls(voice)
    out_dir = Path(config.get("voices_dir", config.VOICES_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{voice}.onnx"
    json_path = out_dir / f"{voice}.onnx.json"
    log.info("downloading voice %s -> %s", voice, onnx_path)
    _download_file(onnx_url, onnx_path, timeout=timeout)
    # The .onnx.json sidecar is metadata; not fatal if it is missing.
    try:
        _download_file(json_url, json_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - sidecar is optional
        log.warning("voice %s: metadata json unavailable (%s)", voice, exc)
    return onnx_path


def download_voices(voices: Iterable[str], timeout: float = 180.0) -> list[str]:
    """Download several voices; return the ids that succeeded."""
    done = []
    for voice in voices:
        try:
            download_voice(voice, timeout=timeout)
            done.append(voice)
        except Exception as exc:  # noqa: BLE001 - never abort the batch
            log.error("voice download failed for %s: %s", voice, exc)
    return done


def ensure_default_voices(timeout: float = 180.0) -> list[str]:
    """Best-effort download of the voices the app needs at minimum.

    Fetches ``dj.voice`` and ``dj.russian_voice`` from config if they are not
    already present. Never raises -- a missing network at boot must not stop
    the station from starting; the DJ simply stays silent until a voice is
    fetched (manually or via the web UI).
    """
    needed = []
    for key in ("dj.voice", "dj.russian_voice"):
        voice = config.get(key)
        if voice and voice_model_path(voice) is None:
            needed.append(voice)
    if not needed:
        return []
    log.info("first-run voice setup: fetching %s", ", ".join(needed))
    return download_voices(needed, timeout=timeout)


def main() -> None:
    """CLI: ``python -m app.voices [--all] [voice ...]``.

    With no args, fetches the default voices (same as startup). ``--all``
    fetches every curated voice in ``dj.VOICE_PROFILES``. Any extra positional
    args are treated as explicit voice ids to fetch.
    """
    import argparse

    from . import dj

    parser = argparse.ArgumentParser(description="Download Piper voice models")
    parser.add_argument("voices", nargs="*", help="explicit voice ids to fetch")
    parser.add_argument("--all", action="store_true",
                        help="fetch every curated voice in the catalogue")
    args = parser.parse_args()

    targets = list(args.voices)
    if args.all:
        targets = [p["id"] for p in dj.VOICE_PROFILES]
    if not targets:
        targets = [config.get("dj.voice"), config.get("dj.russian_voice")]
    targets = [t for t in targets if t]

    if not targets:
        print("nothing to download")
        return
    done = download_voices(targets)
    print(f"downloaded {len(done)} voice(s): {', '.join(done) or 'none'}")
    missing = [t for t in targets if t not in done]
    if missing:
        print(f"FAILED: {', '.join(missing)}")


if __name__ == "__main__":
    main()
