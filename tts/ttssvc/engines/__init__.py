"""Engine registry — the one place that knows which adapters exist."""

from __future__ import annotations

from typing import Type

from .base import Engine, EngineError, wav_bytes
from .piper import PiperEngine
from .supertonic import Supertonic3Engine

REGISTRY: dict[str, Type[Engine]] = {
    Supertonic3Engine.id: Supertonic3Engine,
    PiperEngine.id: PiperEngine,
}


def build_engine(config) -> Engine:  # noqa: ANN001 - config module
    """Instantiate the adapter named by ``TTS_ENGINE``."""
    wanted = config.engine_id()
    cls = REGISTRY.get(wanted)
    if cls is None:
        known = ", ".join(sorted(REGISTRY))
        raise EngineError(f"unknown TTS_ENGINE {wanted!r} (known: {known})")
    return cls(config)


__all__ = ["REGISTRY", "build_engine", "Engine", "EngineError", "wav_bytes"]
