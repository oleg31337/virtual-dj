"""An in-process stand-in for the TTS service.

The app is written against the service's HTTP contract, so the tests fake that
contract (capabilities + speech), not an engine. That keeps every app test
offline and fast while still exercising the real code path: text shaping, the
capability-driven branches, the ffmpeg mastering chain and the cache.
"""

from __future__ import annotations

import array
import io
import math
import wave

from app import tts_client

LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "ru", "name": "Russian"},
]

# Two voices: one speaks both languages, one is Russian-only (the shape a
# single-language engine produces, which the UI must handle).
VOICES = [
    {"id": "V1", "name": "Voice One", "gender": "male",
     "languages": ["en", "ru"], "default_language": "en",
     "description": "speaks both", "installed": True},
    {"id": "V2", "name": "Voice Two", "gender": "female",
     "languages": ["ru"], "default_language": "ru",
     "description": "russian only", "installed": True},
    {"id": "V3", "name": "Voice Three", "gender": "male",
     "languages": ["en"], "default_language": "en",
     "description": "not installed yet", "installed": False},
]

PARAMS = [
    {"key": "speed", "label": "Speaking speed", "type": "float",
     "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "unit": "\u00d7"},
    {"key": "sentence_pause", "label": "Pause after each sentence",
     "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.3,
     "unit": "s"},
]


def wav_bytes(duration: float = 0.25, rate: int = 44100,
              freq: float = 220.0) -> bytes:
    """A real, playable mono WAV (the app feeds it straight to ffmpeg)."""
    frames = int(rate * duration)
    pcm = array.array("h", (int(8000 * math.sin(2 * math.pi * freq * i / rate))
                            for i in range(frames)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class FakeTTS:
    """Configurable fake service; every call is recorded for assertions."""

    def __init__(self, *, engine: str = "fake-engine",
                 engine_name: str = "Fake Engine",
                 voices: list[dict] | None = None,
                 params: list[dict] | None = None,
                 languages: list[dict] | None = None,
                 ready: bool = True, stress_marks: bool = False,
                 render: bytes | None = None, fail: str | None = None,
                 problem: str | None = None) -> None:
        self.engine = engine
        self.engine_name = engine_name
        self.voices = [dict(v) for v in (voices if voices is not None else VOICES)]
        self.params = [dict(p) for p in (params if params is not None else PARAMS)]
        self.languages = languages if languages is not None else LANGUAGES
        self.ready = ready
        self.stress_marks = stress_marks
        self.render = render if render is not None else wav_bytes()
        self.fail = fail
        self.problem_text = problem
        self.calls: list[dict] = []
        self.installs: list[str] = []

    # --- payloads ---------------------------------------------------------
    def capabilities_payload(self) -> dict:
        first = self.voices[0] if self.voices else {}
        return {
            "api": {"version": "1", "service": "virtual-dj-tts"},
            "engine": {"id": self.engine, "name": self.engine_name,
                       "version": "1", "runtime": "test"},
            "defaults": {"voice": first.get("id", ""),
                         "language": first.get("default_language", "en"),
                         "params": {p["key"]: p["default"] for p in self.params}},
            "languages": self.languages,
            "voices": self.voices,
            "params": self.params,
            "output": {"formats": ["wav"], "sample_rates": [44100],
                       "channels": [1]},
            "text": {"stress_marks": self.stress_marks, "ssml": False,
                     "phonemes": False},
            "quality": {"steps": 5, "fixed": True},
        }

    def health_payload(self) -> dict:
        return {
            "ok": self.ready,
            "engine": {"id": self.engine, "name": self.engine_name},
            "model_ready": self.ready,
            "detail": None if self.ready else "model not loaded yet",
            "voices_ready": sum(1 for v in self.voices if v.get("installed")),
            "renders": len(self.calls),
            "uptime_s": 1.0,
        }

    # --- tts_client API ---------------------------------------------------
    def installed_voice_ids(self) -> list[str]:
        return [v["id"] for v in self.voices if v.get("installed")]

    def speech(self, text, voice=None, language=None, params=None):
        self.calls.append({"text": text, "voice": voice,
                           "language": language, "params": dict(params or {})})
        if self.fail:
            raise tts_client.TTSError(self.fail)
        return self.render, {
            "engine": self.engine, "voice": voice, "language": language,
            "duration": 0.25, "sample_rate": 44100, "render_ms": 12,
            "params_used": "{}", "params_ignored": "[]",
            "params_clamped": "[]", "steps": "5",
        }

    def problem(self) -> str | None:
        if self.problem_text is not None:
            return self.problem_text
        if not self.ready:
            return "TTS service model not loaded yet"
        return None


def install(monkeypatch, fake: FakeTTS | None = None, **kwargs) -> FakeTTS:
    """Point ``app.tts_client`` at ``fake`` for the duration of a test."""
    fake = fake or FakeTTS(**kwargs)
    payload = fake.capabilities_payload()
    monkeypatch.setattr(tts_client, "base_url", lambda: "http://fake-tts:9000")
    monkeypatch.setattr(tts_client, "capabilities",
                        lambda force=False: payload)
    monkeypatch.setattr(tts_client, "invalidate", lambda: None)
    monkeypatch.setattr(tts_client, "health", fake.health_payload)
    monkeypatch.setattr(tts_client, "voices", lambda: list(payload["voices"]))
    monkeypatch.setattr(tts_client, "params", lambda: list(payload["params"]))
    monkeypatch.setattr(tts_client, "languages", lambda: payload["languages"])
    monkeypatch.setattr(tts_client, "engine", lambda: payload["engine"])
    monkeypatch.setattr(tts_client, "defaults", lambda: payload["defaults"])
    monkeypatch.setattr(tts_client, "voice",
                        lambda vid: next((v for v in payload["voices"]
                                          if v["id"] == vid), None))
    monkeypatch.setattr(tts_client, "installed_voice_ids",
                        fake.installed_voice_ids)
    monkeypatch.setattr(tts_client, "speech", fake.speech)
    monkeypatch.setattr(tts_client, "problem", fake.problem)
    monkeypatch.setattr(tts_client, "install_voice",
                        lambda vid: fake.installs.append(vid) or
                        {"voice": vid, "installed": True, "already": False,
                         "bytes": 1})
    return fake
