"""The app's side of the TTS HTTP contract.

Everything the app knows about speech engines lives behind these calls, so this
is the file that keeps an outage, a stale document and an engine swap from
reaching the station: the client caches capabilities, maps failures to a
sentence a human can act on, and never raises its way into the audio pipeline.
"""

from __future__ import annotations

import pytest

from app import tts_client


class FakeResponse:
    def __init__(self, status_code: int = 200, json_payload=None,
                 content: bytes = b"", headers: dict | None = None,
                 text: str = "") -> None:
        self.status_code = status_code
        self._json = json_payload
        self.content = content
        self.headers = headers or {}
        self.text = text or (str(json_payload) if json_payload is not None else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTP:
    """Records requests and replays a scripted answer per (method, path)."""

    def __init__(self, answers: dict | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, str, dict]] = []

    def _answer(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        key = (method, url.split("9000", 1)[-1])
        answer = self.answers.get(key, self.answers.get(method))
        if answer is None:
            raise AssertionError(f"unexpected {method} {url}")
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer()
        return answer

    def get(self, url, **kwargs):
        return self._answer("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._answer("POST", url, **kwargs)


def caps_doc() -> dict:
    """The cached capability document, asserted non-empty."""
    return tts_client.capabilities() or {}


CAPS = {
    "api": {"version": "1"},
    "engine": {"id": "supertonic3", "name": "Supertonic 3"},
    "defaults": {"voice": "M2", "language": "ru", "params": {"speed": 1.0}},
    "languages": [{"code": "en", "name": "English"}, {"code": "ru", "name": "Russian"}],
    "voices": [
        {"id": "M2", "name": "Male 2", "languages": ["en", "ru"],
         "default_language": "ru", "installed": True},
        {"id": "F1", "name": "Female 1", "languages": ["en", "ru"],
         "default_language": "en", "installed": False},
    ],
    "params": [
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0,
         "min": 0.5, "max": 2.0, "step": 0.05},
        {"key": "sentence_pause", "label": "Pause", "type": "float",
         "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05},
    ],
    "output": {"formats": ["wav"], "sample_rates": [44100], "channels": [1]},
    "text": {"stress_marks": False},
    "quality": {"steps": 5, "fixed": True},
}

HEADERS = {
    "X-TTS-Engine": "supertonic3",
    "X-TTS-Voice": "M2",
    "X-TTS-Language": "ru",
    "X-TTS-Duration": "9.47",
    "X-TTS-Sample-Rate": "44100",
    "X-TTS-Render-Ms": "812",
    "X-TTS-Params-Used": '{"speed": 1.0}',
    "X-TTS-Params-Ignored": "[]",
    "X-TTS-Params-Clamped": "[]",
    "X-TTS-Steps": "5",
}


@pytest.fixture(autouse=True)
def fresh_client(monkeypatch):
    """Each test starts with an empty cache and a fixed service URL."""
    monkeypatch.setattr(tts_client, "_caps", None)
    monkeypatch.setattr(tts_client, "_caps_at", 0.0)
    monkeypatch.setattr(tts_client, "_caps_error", None)
    monkeypatch.setattr(tts_client, "base_url", lambda: "http://fake:9000")
    monkeypatch.setattr(tts_client.time, "sleep", lambda *_: None)


# --- the capability document -------------------------------------------------

def test_capabilities_are_fetched_once_and_cached(monkeypatch):
    http = FakeHTTP({("GET", "/v1/capabilities"): FakeResponse(json_payload=CAPS)})
    monkeypatch.setattr(tts_client, "httpx", http)
    assert caps_doc()["engine"]["id"] == "supertonic3"
    assert caps_doc()["engine"]["id"] == "supertonic3"
    assert len(http.calls) == 1
    tts_client.invalidate()
    tts_client.capabilities()
    assert len(http.calls) == 2


def test_a_dead_service_keeps_serving_the_last_document(monkeypatch):
    """A capability blip must not blank the DJ card or stop the station."""
    monkeypatch.setattr(tts_client, "_caps", dict(CAPS))
    monkeypatch.setattr(tts_client, "_caps_at", 0.0)  # expired
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP(
        {("GET", "/v1/capabilities"): RuntimeError("connection refused")}))
    assert tts_client.voices()[0]["id"] == "M2"


def test_no_document_and_no_service_is_none_not_an_error(monkeypatch):
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP(
        {("GET", "/v1/capabilities"): RuntimeError("connection refused")}))
    assert tts_client.capabilities() is None
    assert tts_client.voices() == []
    assert tts_client.params() == []
    assert tts_client.default_voice() == ""


def test_health_reports_an_unreachable_service(monkeypatch):
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP(
        {("GET", "/v1/health"): RuntimeError("timed out")}))
    health = tts_client.health()
    assert health["ok"] is False and health["unreachable"] is True
    assert "http://fake:9000" in health["detail"]


def test_an_empty_base_url_is_explained(monkeypatch):
    monkeypatch.setattr(tts_client, "base_url", lambda: "")
    assert tts_client.health()["ok"] is False
    assert "no tts.base_url" in tts_client.health()["detail"]
    assert "VDJ_TTS_URL" in (tts_client.problem() or "")


# --- speech ------------------------------------------------------------------

def test_speech_posts_the_request_and_reads_the_metadata(monkeypatch):
    http = FakeHTTP({("POST", "/v1/speech"): FakeResponse(
        content=b"RIFFdata", headers=HEADERS)})
    monkeypatch.setattr(tts_client, "httpx", http)

    audio, meta = tts_client.speech("Привет", "M2", "ru", {"speed": 1.0})
    assert audio == b"RIFFdata"
    assert meta["engine"] == "supertonic3"
    assert meta["duration"] == 9.47
    assert meta["steps"] == "5"
    body = http.calls[0][2]["json"]
    assert body == {"text": "Привет", "voice": "M2", "language": "ru",
                    "params": {"speed": 1.0}}


def test_an_omitted_voice_is_sent_as_null_so_the_service_decides(monkeypatch):
    http = FakeHTTP({("POST", "/v1/speech"): FakeResponse(
        content=b"RIFFdata", headers=HEADERS)})
    monkeypatch.setattr(tts_client, "httpx", http)
    tts_client.speech("hello")
    assert http.calls[0][2]["json"]["voice"] is None


def test_a_transport_failure_is_retried_once(monkeypatch):
    attempts: list[int] = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("reset by peer")
        return FakeResponse(content=b"RIFFdata", headers=HEADERS)

    monkeypatch.setattr(tts_client, "httpx",
                        FakeHTTP({("POST", "/v1/speech"): flaky}))
    audio, _ = tts_client.speech("hello")
    assert audio == b"RIFFdata" and len(attempts) == 2


def test_two_transport_failures_raise_a_readable_error(monkeypatch):
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP(
        {("POST", "/v1/speech"): RuntimeError("connection refused")}))
    with pytest.raises(tts_client.TTSError) as excinfo:
        tts_client.speech("hello")
    assert "unreachable" in str(excinfo.value)


def test_a_rejected_request_surfaces_the_service_message(monkeypatch):
    """The service's 422 detail names the offending field — pass it through."""
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP({("POST", "/v1/speech"):
        FakeResponse(status_code=422, json_payload={
            "detail": {"message": "voice 'F9' does not exist"}})}))
    with pytest.raises(tts_client.TTSError) as excinfo:
        tts_client.speech("hello", "F9")
    assert "does not exist" in str(excinfo.value)


def test_install_posts_and_invalidates_the_cached_document(monkeypatch):
    http = FakeHTTP({
        ("GET", "/v1/capabilities"): FakeResponse(json_payload=CAPS),
        ("POST", "/v1/voices/F1/install"): FakeResponse(
            json_payload={"voice": "F1", "installed": True}),
    })
    monkeypatch.setattr(tts_client, "httpx", http)
    tts_client.capabilities()
    assert tts_client.install_voice("F1")["installed"] is True
    tts_client.capabilities()          # cache was dropped, so this refetches
    assert [c[0] for c in http.calls] == ["GET", "POST", "GET"]


def test_a_failed_install_raises(monkeypatch):
    monkeypatch.setattr(tts_client, "httpx", FakeHTTP({
        ("POST", "/v1/voices/F9/install"): FakeResponse(
            status_code=404, json_payload={"detail": "unknown voice"})}))
    with pytest.raises(tts_client.TTSError) as excinfo:
        tts_client.install_voice("F9")
    assert "unknown voice" in str(excinfo.value)


# --- resolution --------------------------------------------------------------

def test_resolve_params_merges_saved_values_over_declared_defaults(monkeypatch):
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    assert tts_client.resolve_params()["sentence_pause"] == 0.3
    resolved = tts_client.resolve_params({"speed": 1.4})
    assert resolved["speed"] == 1.4
    # A key from a previous engine is kept: the service reports and ignores it,
    # so switching engines back and forth does not lose the user's setting.
    assert tts_client.resolve_params({"noise_w": 0.8})["noise_w"] == 0.8


def test_resolve_params_falls_back_to_the_configured_values(monkeypatch):
    from app import config

    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setitem(config.DEFAULTS["dj"], "tts_params", {"speed": 1.2})
    config.load_config(force=True)
    assert tts_client.resolve_params()["speed"] == 1.2


def test_resolve_language_never_exceeds_the_voice(monkeypatch):
    english_only = {"languages": ["en"], "default_language": "en"}
    both = {"languages": ["en", "ru"], "default_language": "ru"}
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    assert tts_client.resolve_language(both, "ru") == "ru"
    assert tts_client.resolve_language(english_only, "ru") == "en"
    assert tts_client.resolve_language(both, "de") == "ru"   # voice default
    assert tts_client.resolve_language({}, "de") == "de"     # nothing to go on


def test_default_voice_prefers_installed_ones(monkeypatch):
    from app import config

    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "F1")  # not installed
    config.load_config(force=True)
    assert tts_client.default_voice() == "M2"


# --- problem() ---------------------------------------------------------------

def test_problem_names_an_unreachable_service(monkeypatch):
    monkeypatch.setattr(tts_client, "health",
                        lambda: {"ok": False, "detail": "connection refused"})
    problem = tts_client.problem() or ""
    assert "connection refused" in problem and "http://fake:9000" in problem


def test_problem_names_a_service_with_no_voice_installed(monkeypatch):
    monkeypatch.setattr(tts_client, "health", lambda: {"ok": True})
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setattr(tts_client, "installed_voice_ids", lambda: [])
    assert "no voice installed" in (tts_client.problem() or "")


def test_problem_is_silent_when_a_usable_voice_resolves(monkeypatch):
    """A configured voice that cannot speak is a substitution, not an outage.

    The card names it; the station keeps talking with the voice it does have.
    """
    from app import config

    monkeypatch.setattr(tts_client, "health", lambda: {"ok": True})
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setattr(tts_client, "installed_voice_ids", lambda: ["M2"])
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "M2")
    config.load_config(force=True)
    config.save_config({"dj": {"voice": "F1"}})      # F1 exists, not installed
    assert tts_client.problem() is None
    assert tts_client.default_voice() == "M2"


def test_problem_is_none_when_everything_is_in_place(monkeypatch):
    from app import config

    monkeypatch.setattr(tts_client, "health", lambda: {"ok": True})
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setattr(tts_client, "installed_voice_ids", lambda: ["M2"])
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "M2")
    config.load_config(force=True)
    assert tts_client.problem() is None


def test_a_dead_voice_id_does_not_silence_the_station(monkeypatch):
    """A config from a previous engine names a voice this one never heard of.

    `problem()` must judge the voice that will actually speak — otherwise a
    working station reports "DJ breaks are OFF" forever.
    """
    from app import config

    monkeypatch.setattr(tts_client, "health", lambda: {"ok": True})
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: CAPS)
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "M2")
    config.load_config(force=True)
    # The stored config names a Piper id; M2 is installed and is our own pick.
    config.save_config({"dj": {"voice": "en_GB-alan-medium"}})
    assert tts_client.default_voice() == "M2"
    assert tts_client.problem() is None


def test_no_usable_voice_is_still_named(monkeypatch):
    monkeypatch.setattr(tts_client, "health", lambda: {"ok": True})
    only_missing = dict(CAPS, voices=[{"id": "F1", "name": "Female 1",
                                       "languages": ["en"],
                                       "default_language": "en",
                                       "installed": False}])
    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: only_missing)
    assert "no voice installed" in (tts_client.problem() or "")
