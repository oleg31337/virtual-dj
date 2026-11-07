"""Voice installation now happens on the TTS service (app.voices is a thin CLI).

The app no longer owns a voice format: a voice belongs to whatever engine the
service runs, so "install" means one POST to that service. These tests fake the
service and check the app's side of that deal.
"""

from __future__ import annotations

import pytest

from app import config, voices


def test_known_voices_come_from_the_service(monkeypatch):
    import fake_tts

    fake_tts.install(monkeypatch)
    assert voices.known_voices() == ["V1", "V2", "V3"]


def test_install_voice_success(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)
    assert voices.install_voice("V3") is True
    assert fake.installs == ["V3"]


def test_install_voice_failure_is_reported_not_raised(monkeypatch):
    import fake_tts
    from app import tts_client

    fake_tts.install(monkeypatch)

    def boom(voice_id):
        raise tts_client.TTSError("no network in the tts container")

    monkeypatch.setattr(tts_client, "install_voice", boom)
    assert voices.install_voice("V3") is False
    assert voices.install_voices(["V3", "V1"]) == []


def test_install_voices_returns_only_the_successes(monkeypatch):
    import fake_tts
    from app import tts_client

    fake_tts.install(monkeypatch)
    calls: list[str] = []

    def flaky(voice_id):
        calls.append(voice_id)
        if voice_id == "V3":
            raise tts_client.TTSError("boom")
        return {"voice": voice_id, "installed": True}

    monkeypatch.setattr(tts_client, "install_voice", flaky)
    assert voices.install_voices(["V1", "V3", "V2"]) == ["V1", "V2"]
    assert calls == ["V1", "V3", "V2"]


def test_ensure_default_voices_skips_what_is_already_installed(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)  # V1 is installed
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "V1")
    config.load_config(force=True)
    assert voices.ensure_default_voices() == []
    assert fake.installs == []


def test_ensure_default_voices_installs_the_configured_one(monkeypatch):
    import fake_tts

    # A service where nothing is installed yet: the configured voice must be
    # installed.
    payload = [dict(v, installed=False) for v in fake_tts.VOICES]
    fake = fake_tts.install(monkeypatch, voices=payload)
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "V2")
    config.load_config(force=True)
    assert voices.ensure_default_voices() == ["V2"]
    assert fake.installs == ["V2"]


def test_a_service_that_offers_nothing_is_not_an_error(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch, voices=[], languages=[])
    assert voices.known_voices() == []
    assert voices.ensure_default_voices() == []
    assert fake.installs == []


def test_an_unreachable_service_defers_the_install(monkeypatch, caplog):
    """The service may still be downloading its model at boot.

    That is not "the engine offers nothing" — nothing to install, and nothing to
    alarm the user about, because the DJ card installs on demand.
    """
    from app import tts_client

    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: None)
    monkeypatch.setattr(tts_client, "voices", lambda: [])

    def must_not_install(voice_id):
        raise AssertionError("nothing should be installed while the service is down")

    monkeypatch.setattr(tts_client, "install_voice", must_not_install)
    with caplog.at_level("INFO", logger="app.voices"):
        assert voices.ensure_default_voices() == []
    assert any("deferred" in r.getMessage() for r in caplog.records), caplog.text


def test_cli_lists_what_the_engine_offers(monkeypatch, capsys):
    import fake_tts

    fake_tts.install(monkeypatch)
    monkeypatch.setattr("sys.argv", ["app.voices", "--list"])
    voices.main()
    out = capsys.readouterr().out
    assert "V1" in out and "installed" in out
    assert "V3" in out and "not installed" in out


def test_cli_installs_an_explicit_voice(monkeypatch, capsys):
    import fake_tts

    fake = fake_tts.install(monkeypatch)
    monkeypatch.setattr("sys.argv", ["app.voices", "V3"])
    voices.main()
    assert fake.installs == ["V3"]
    assert "installed 1 voice" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["V1", "V2"])
def test_the_catalogue_matches_what_the_service_reports(monkeypatch, name):
    import fake_tts

    fake_tts.install(monkeypatch)
    assert name in voices.known_voices()
