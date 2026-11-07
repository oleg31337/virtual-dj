"""The contract every TTS engine adapter implements.

Adding an engine means: one module in this package, an ``Engine`` subclass, and
one line in ``engines.REGISTRY``. The HTTP layer only ever talks to this
interface, which is what keeps the app independent of the engine.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class EngineError(RuntimeError):
    """An engine cannot fulfil the request (missing files, not loaded, ...)."""


class Engine:
    """Base class; subclasses fill in the metadata and the three verbs."""

    #: stable id used by ``TTS_ENGINE`` (e.g. ``supertonic3``)
    id: str = "base"
    #: human-readable name shown in the app's DJ card
    name: str = "Base"
    version: str = "0"
    license: str = "unknown"
    runtime: str = "python"
    #: languages this deployment advertises (filtered from ``languages()``)
    advertised_languages: tuple[str, ...] = ("en", "ru")

    def __init__(self, config) -> None:  # noqa: ANN001 - the service config module
        self.config = config

    # --- metadata ---------------------------------------------------------
    def languages(self) -> list[dict[str, str]]:
        """Languages this engine can speak, as ``[{code, name}]``."""
        raise NotImplementedError

    def voices(self) -> list[dict[str, Any]]:
        """Every voice, installed or not.

        Each entry: ``{id, name, gender, languages: [code...],
        default_language, description, installed: bool}``.
        """
        raise NotImplementedError

    def params(self) -> list[dict[str, Any]]:
        """The controllable parameters, as the schema documented in the plan."""
        raise NotImplementedError

    def text_support(self) -> dict[str, bool]:
        """What the engine understands in the text itself."""
        return {"stress_marks": False, "ssml": False, "phonemes": False}

    def sample_rate(self) -> int:
        raise NotImplementedError

    # --- lifecycle --------------------------------------------------------
    def ensure_model(self, progress: Callable[[str], None] | None = None) -> None:
        """Make the model usable: download if needed, then load it."""
        raise NotImplementedError

    def model_ready(self) -> tuple[bool, str | None]:
        """``(ready, why_not)`` — cheap enough to call on every health check."""
        raise NotImplementedError

    def install_voice(self, voice_id: str) -> dict[str, Any]:
        """Make one voice usable. Returns ``{voice, installed, already, bytes}``."""
        raise NotImplementedError

    # --- the verb ---------------------------------------------------------
    def synthesize(self, text: str, voice: str, language: str,
                   params: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Render ``text``; return ``(float32 mono samples, sample_rate)``."""
        raise NotImplementedError


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono in [-1, 1] as 16-bit PCM WAV."""
    import io
    import wave

    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
