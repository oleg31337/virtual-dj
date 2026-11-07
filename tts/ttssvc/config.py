"""Configuration for the standalone TTS service.

Everything is env-driven so the container is configured from .env / compose and
a different deployment (other host, other engine) needs no code change.
"""

from __future__ import annotations

import os
from pathlib import Path

API_VERSION = "1"
SERVICE_NAME = "virtual-dj-tts"


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = _str(name, "1" if default else "0").lower()
    return value not in ("0", "false", "no", "off", "")


def engine_id() -> str:
    """Which engine adapter to load: ``supertonic3`` (default) or ``piper``."""
    return _str("TTS_ENGINE", "supertonic3").lower()


def model_dir() -> Path:
    return Path(_str("TTS_MODEL_DIR", "/models"))


def steps() -> int:
    """Diffusion steps for engines that have them.

    Fixed by deployment, NOT a user-facing parameter: Supertonic's 2-step
    "fast" mode garbles words (measured 54-77 % Russian WER vs 9-36 % at 5).
    Clamped to a sane band so a typo cannot produce noise.
    """
    return max(2, min(_int("TTS_STEPS", 5), 8))


def languages() -> list[str]:
    """Languages the service advertises, e.g. ``en,ru``.

    The engine may support more; the station only needs these two and an
    honest list keeps the app's picker short.
    """
    raw = _str("TTS_LANGUAGES", "en,ru")
    return [code.strip().lower() for code in raw.split(",") if code.strip()]


def autoload() -> bool:
    """Load the model at startup (else on the first request)."""
    return _bool("TTS_AUTOLOAD", True)


def autodownload() -> bool:
    """Fetch model files at startup when they are missing."""
    return _bool("TTS_AUTODOWNLOAD", True)


def threads() -> int:
    """ONNX Runtime intra-op threads. 0 = let onnxruntime decide."""
    return max(0, _int("TTS_THREADS", 0))


def max_text_chars() -> int:
    return max(50, _int("TTS_MAX_TEXT_CHARS", 2000))


def default_voice() -> str:
    return _str("TTS_DEFAULT_VOICE", "")


def default_language() -> str:
    return _str("TTS_DEFAULT_LANGUAGE", "")


def host() -> str:
    return _str("TTS_HOST", "0.0.0.0")


def port() -> int:
    return _int("TTS_PORT", 9000)


def log_level() -> str:
    return _str("TTS_LOG_LEVEL", "info").lower()
