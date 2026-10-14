"""Tests for the Piper voice auto-downloader (app.voices).

Real downloads hit HuggingFace; these tests stub the HTTP stream so they run
offline and fast while still exercising URL construction, atomic file write,
and the ``voice_model_path`` resolver.
"""

from __future__ import annotations

import pytest

from app import config, dj, voices


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
    # Simulate a realistic voice blob (>= 1 MB) so the post-download
    # integrity check in download_voice() passes.
    payload = b"X" * (2 * 1024 * 1024)

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


def test_download_voice_requires_the_json_sidecar(monkeypatch):
    # Piper loads "<model>.onnx.json" whenever it loads a model, so a sidecar
    # failure must fail the download (and drop the orphaned model) instead of
    # looking like success and then silently dropping every DJ break.
    onnx_payload = b"X" * (2 * 1024 * 1024)

    class _FakeClient:
        def stream(self, method, url, **kw):
            if url.endswith(".onnx.json"):
                return _fake_stream(b"", status=404)
            return _fake_stream(onnx_payload)

    monkeypatch.setattr(voices.httpx, "stream", _FakeClient().stream)
    with pytest.raises(RuntimeError):
        voices.download_voice("en_US-amy-medium")
    onnx = config.voices_dir() / "en_US-amy-medium.onnx"
    assert not onnx.exists()
    assert voices.voice_model_path("en_US-amy-medium") is None


def test_voice_without_sidecar_is_not_installed(tmp_path, monkeypatch):
    # A model that landed without its sidecar must not be offered as installed.
    monkeypatch.setattr(config, "voices_dir", lambda: tmp_path)
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"X" * 10)
    assert voices.voice_model_path("en_US-amy-medium") is None
    (tmp_path / "en_US-amy-medium.onnx.json").write_text("{}")
    assert voices.voice_model_path("en_US-amy-medium") is not None
    # The UI's voice list is the same test, so the picker agrees.
    assert dj.available_voices() == ["en_US-amy-medium"]


def test_ensure_default_voices_fetches_missing(monkeypatch):
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "en_US-amy-medium")

    captured = {}

    class _FakeClient:
        def stream(self, method, url, **kw):
            captured.setdefault("urls", []).append(url)
            # Simulate a realistic voice blob (>= 1 MB) so the integrity
            # check in download_voice() passes.
            return _fake_stream(b"X" * (2 * 1024 * 1024))

    monkeypatch.setattr(voices.httpx, "stream", _FakeClient().stream)
    done = voices.ensure_default_voices()
    assert done == ["en_US-amy-medium"]
    # Re-running finds them already present -> no network.
    done2 = voices.ensure_default_voices()
    assert done2 == []
