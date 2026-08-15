"""Tests for the Piper voice auto-downloader (app.voices).

Real downloads hit HuggingFace; these tests stub the HTTP stream so they run
offline and fast while still exercising URL construction, atomic file write,
and the ``voice_model_path`` resolver.
"""

from __future__ import annotations

import io

import pytest

from app import config, voices


def _fake_stream(payload: bytes, status: int = 200):
    """Return a context-manager stand-in for ``httpx.stream``."""

    class _Resp:
        def __init__(self):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def iter_bytes(self, n):
            yield payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _CM:
        def __enter__(self):
            return _Resp()

        def __exit__(self, *exc):
            return False

    return _CM()


def test_voice_urls_are_well_formed():
    onnx, jsonf = voices._voice_urls("en_US-amy-medium")
    assert onnx.endswith("en/en_US/amy/medium/en_US-amy-medium.onnx")
    assert jsonf.endswith("en/en_US/amy/medium/en_US-amy-medium.onnx.json")

    ru_onnx, _ = voices._voice_urls("ru_RU-irina-medium")
    assert ru_onnx.endswith("ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx")


def test_download_voice_writes_onnx_and_json(monkeypatch):
    payload = b"FAKE-ONNX-BYTES"

    class _FakeClient:
        def stream(self, method, url, **kw):
            # Serve the .onnx; the .json sidecar gets a tiny stub too.
            return _fake_stream(payload)

    monkeypatch.setattr(voices.httpx, "stream", _FakeClient().stream)

    path = voices.download_voice("en_US-amy-medium")
    assert path.exists()
    assert path.read_bytes() == payload
    assert (path.with_suffix(".onnx.json")).exists()
    # The resolver now finds it.
    assert voices.voice_model_path("en_US-amy-medium") == path


def test_download_voice_missing_json_is_non_fatal(monkeypatch):
    # The .onnx.json sidecar is optional; a failure there must not abort.

    class _FakeClient:
        def stream(self, method, url, **kw):
            if url.endswith(".onnx.json"):
                return _fake_stream(b"", status=404)
            return _fake_stream(b"ONNXDATA")

    monkeypatch.setattr(voices.httpx, "stream", _FakeClient().stream)
    path = voices.download_voice("en_US-amy-medium")
    assert path.exists()
    assert not (path.with_suffix(".onnx.json")).exists()


def test_ensure_default_voices_fetches_missing(monkeypatch):
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "en_US-amy-medium")
    monkeypatch.setitem(config.DEFAULTS["dj"], "russian_voice", "ru_RU-irina-medium")

    captured = {}

    class _FakeClient:
        def stream(self, method, url, **kw):
            captured.setdefault("urls", []).append(url)
            return _fake_stream(b"x")

    monkeypatch.setattr(voices.httpx, "stream", _FakeClient().stream)
    done = voices.ensure_default_voices()
    assert set(done) == {"en_US-amy-medium", "ru_RU-irina-medium"}
    # Re-running finds them already present -> no network.
    done2 = voices.ensure_default_voices()
    assert done2 == []
