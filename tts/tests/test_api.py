"""HTTP contract tests against a stub engine.

The point is the *contract*, not the model: these run in a second without any
weights, so a change to the API shape or the lenient-parameter rules fails here
before it can reach the app.
"""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ttssvc import config, main
from ttssvc.engines.base import Engine, EngineError


class StubEngine(Engine):
    id = "stub"
    name = "Stub Engine"
    version = "9.9"
    license = "test"
    runtime = "python"
    advertised_languages = ("en", "ru")

    def __init__(self, cfg, *, ready: bool = True,
                 installed: tuple[str, ...] = ("V1", "V2")) -> None:
        super().__init__(cfg)
        self._ready = ready
        self._installed = set(installed)
        self.calls: list[tuple[str, str, str, dict]] = []
        self.install_calls: list[str] = []

    def languages(self):
        return [{"code": "en", "name": "English"}, {"code": "ru", "name": "Russian"}]

    def voices(self):
        return [
            {"id": "V1", "name": "Voice One", "gender": "male",
             "languages": ["en", "ru"], "default_language": "en",
             "description": "first", "installed": "V1" in self._installed},
            {"id": "V2", "name": "Voice Two", "gender": "female",
             "languages": ["ru"], "default_language": "ru",
             "description": "second", "installed": "V2" in self._installed},
        ]

    def params(self):
        return [
            {"key": "speed", "label": "Speed", "type": "float", "min": 0.5,
             "max": 2.0, "step": 0.05, "default": 1.0},
            {"key": "mode", "label": "Mode", "type": "enum", "default": "a",
             "choices": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}]},
            {"key": "loud", "label": "Loud", "type": "bool", "default": False},
        ]

    def text_support(self):
        return {"stress_marks": False, "ssml": False, "phonemes": False}

    def sample_rate(self):
        return 22050

    def ensure_model(self, progress=None):
        return None

    def model_ready(self):
        return (True, None) if self._ready else (False, "stub is not ready")

    def install_voice(self, voice_id):
        self.install_calls.append(voice_id)
        self._installed.add(voice_id)
        return {"voice": voice_id, "installed": True,
                "already": False, "bytes": 1234}

    def synthesize(self, text, voice, language, params):
        self.calls.append((text, voice, language, dict(params)))
        rate = 22050
        n = int(rate * 0.25)
        wave_data = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / rate)
        return wave_data.astype(np.float32), rate


@pytest.fixture()
def stub():
    return StubEngine(config)


@pytest.fixture()
def client(monkeypatch, stub):
    monkeypatch.setattr(main, "build_engine", lambda cfg: stub)
    monkeypatch.setattr(config, "autoload", lambda: False)
    with TestClient(main.app) as c:
        main.STATE["engine"] = stub
        main.STATE["boot_error"] = None
        main.STATE["boot_note"] = "ready"
        yield c


def _wav_info(payload: bytes) -> tuple[int, int, float]:
    with wave.open(io.BytesIO(payload), "rb") as wf:
        frames = wf.getnframes()
        return wf.getnchannels(), wf.getframerate(), frames / wf.getframerate()


def test_root_and_health(client):
    assert client.get("/").json()["service"] == "virtual-dj-tts"
    health = client.get("/v1/health").json()
    assert health["ok"] is True
    assert health["engine"]["id"] == "stub"
    assert health["voices_ready"] == 2
    assert health["detail"] is None
    assert health["steps"] == config.steps()


def test_health_reports_not_ready(monkeypatch):
    stub = StubEngine(config, ready=False)
    monkeypatch.setattr(main, "build_engine", lambda cfg: stub)
    monkeypatch.setattr(config, "autoload", lambda: False)
    with TestClient(main.app) as c:
        main.STATE["engine"] = stub
        health = c.get("/v1/health").json()
    assert health["ok"] is False
    assert "not ready" in health["detail"]


def test_capabilities_declares_everything_the_gui_needs(client):
    caps = client.get("/v1/capabilities").json()
    assert caps["api"]["version"] == "1"
    assert caps["engine"]["id"] == "stub"
    assert [lang["code"] for lang in caps["languages"]] == ["en", "ru"]
    assert caps["defaults"]["voice"] == "V1"
    assert caps["defaults"]["params"] == {"speed": 1.0, "mode": "a", "loud": False}
    assert caps["output"]["sample_rates"] == [22050]
    assert caps["text"]["stress_marks"] is False
    assert caps["quality"] == {"steps": config.steps(), "fixed": True}
    keys = {p["key"] for p in caps["params"]}
    assert keys == {"speed", "mode", "loud"}
    for p in caps["params"]:
        assert {"key", "label", "type", "default"} <= set(p)
        assert p["type"] in ("float", "int", "enum", "bool")


def test_speech_returns_wav_and_headers(client, stub):
    resp = client.post("/v1/speech", json={
        "text": "Привет, это тест.",
        "voice": "V2", "language": "ru",
        "params": {"speed": 1.25},
    })
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("audio/wav")
    channels, rate, duration = _wav_info(resp.content)
    assert (channels, rate) == (1, 22050)
    assert duration == pytest.approx(0.25, abs=0.01)
    assert resp.headers["X-TTS-Engine"] == "stub"
    assert resp.headers["X-TTS-Voice"] == "V2"
    assert resp.headers["X-TTS-Language"] == "ru"
    assert float(resp.headers["X-TTS-Duration"]) == pytest.approx(0.25, abs=0.01)
    used = json.loads(resp.headers["X-TTS-Params-Used"])
    assert used["speed"] == 1.25
    assert stub.calls[-1][3]["speed"] == 1.25


def test_language_defaults_to_the_voice(client):
    resp = client.post("/v1/speech", json={"text": "hello", "voice": "V2"})
    assert resp.status_code == 200
    assert resp.headers["X-TTS-Language"] == "ru"


def test_unknown_voice_is_404(client):
    resp = client.post("/v1/speech", json={"text": "hi", "voice": "NOPE"})
    assert resp.status_code == 404
    body = resp.json()["detail"]
    assert body["code"] == "unknown_voice"
    assert body["available"] == ["V1", "V2"]


def test_unsupported_language_is_400(client):
    resp = client.post("/v1/speech",
                       json={"text": "hi", "voice": "V2", "language": "en"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["languages"] == ["ru"]


def test_not_installed_voice_is_409(monkeypatch):
    stub = StubEngine(config, installed=())
    monkeypatch.setattr(main, "build_engine", lambda cfg: stub)
    monkeypatch.setattr(config, "autoload", lambda: False)
    with TestClient(main.app) as c:
        main.STATE["engine"] = stub
        resp = c.post("/v1/speech", json={"text": "hi", "voice": "V1"})
        assert resp.status_code == 409
        assert resp.json()["detail"]["install"] == "/v1/voices/V1/install"
        # installing then retrying works
        assert c.post("/v1/voices/V1/install").json()["installed"] is True
        assert c.post("/v1/speech", json={"text": "hi", "voice": "V1"}).status_code == 200


def test_unknown_param_is_ignored_not_fatal(client):
    resp = client.post("/v1/speech", json={
        "text": "hi", "voice": "V1",
        "params": {"speed": 1.0, "noise_w": 0.8, "sentence_silence": 0.25},
    })
    assert resp.status_code == 200
    ignored = json.loads(resp.headers["X-TTS-Params-Ignored"])
    assert set(ignored) == {"noise_w", "sentence_silence"}


def test_out_of_range_param_is_clamped(client, stub):
    resp = client.post("/v1/speech", json={
        "text": "hi", "voice": "V1", "params": {"speed": 9.0}})
    assert resp.status_code == 200
    assert json.loads(resp.headers["X-TTS-Params-Clamped"]) == ["speed"]
    assert stub.calls[-1][3]["speed"] == 2.0


def test_wrong_param_type_is_422(client):
    resp = client.post("/v1/speech", json={
        "text": "hi", "voice": "V1", "params": {"speed": {"a": 1}}})
    assert resp.status_code == 422
    assert resp.json()["detail"]["param"] == "speed"


def test_stale_enum_value_is_ignored(client, stub):
    resp = client.post("/v1/speech", json={
        "text": "hi", "voice": "V1", "params": {"mode": "zzz"}})
    assert resp.status_code == 200
    assert "mode" in json.loads(resp.headers["X-TTS-Params-Ignored"])
    assert stub.calls[-1][3]["mode"] == "a"


def test_empty_and_overlong_text(client, monkeypatch):
    assert client.post("/v1/speech", json={"text": "   "}).status_code == 400
    monkeypatch.setattr(config, "max_text_chars", lambda: 50)
    resp = client.post("/v1/speech", json={"text": "x" * 51})
    assert resp.status_code == 400


def test_model_not_ready_is_503(monkeypatch):
    stub = StubEngine(config, ready=False)
    monkeypatch.setattr(main, "build_engine", lambda cfg: stub)
    monkeypatch.setattr(config, "autoload", lambda: False)
    with TestClient(main.app) as c:
        main.STATE["engine"] = stub
        resp = c.post("/v1/speech", json={"text": "hi", "voice": "V1"})
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "model_not_ready"


def test_install_unknown_voice_is_404(client):
    resp = client.post("/v1/voices/NOPE/install")
    assert resp.status_code == 404
    assert resp.json()["detail"]["available"] == ["V1", "V2"]


def test_engine_error_at_boot_is_reported(monkeypatch):
    def boom(cfg):
        raise EngineError("unknown TTS_ENGINE 'nope'")

    monkeypatch.setattr(main, "build_engine", boom)
    monkeypatch.setattr(config, "autoload", lambda: False)
    with TestClient(main.app) as c:
        main.STATE["engine"] = None
        health = c.get("/v1/health").json()
        assert health["ok"] is False
        assert "nope" in health["detail"]
        assert c.get("/v1/capabilities").status_code == 503
