"""Migrating a config.json written for the old in-app Piper engine.

An existing install carries four Piper knobs (`speed`, `noise_scale`, `noise_w`,
`sentence_silence`) and a Piper voice id. Both are meaningless to the TTS service,
and the migration is what stops a real deployment from coming up silent or
speaking the wrong language.
"""

from __future__ import annotations

import json

from app import config

LEGACY = {
    "music_dir": "/music",
    "dj": {
        "enabled": True,
        "voice": "en_GB-alan-medium",
        "speed": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.25,
        "style": "warm, witty late-night radio host",
        "talk_min": 2,
        "talk_max": 4,
        "sent_min": 1,
        "sent_max": 3,
        "gain_db": 0.0,
        "prefetch_depth": 3,
    },
}


def _load(monkeypatch, tmp_path, stored):
    config.CONFIG_PATH.write_text(json.dumps(stored), "utf-8")
    monkeypatch.setattr(config, "_CACHE", None)
    return config.load_config()


def test_the_four_piper_knobs_become_tts_params(monkeypatch, tmp_path):
    dj = _load(monkeypatch, tmp_path, LEGACY)["dj"]
    params = dj["tts_params"]
    assert params["speed"] == 1.0
    assert params["sentence_pause"] == 0.25        # was dj.sentence_silence
    assert params["expressiveness"] == 0.667       # was dj.noise_scale
    assert params["rhythm"] == 0.8                 # was dj.noise_w
    # The old keys stay for anyone reading the file; nothing reads them now.
    assert dj["noise_w"] == 0.8


def test_an_english_piper_voice_does_not_pin_the_language(monkeypatch, tmp_path):
    """The migrated station speaks the configured default (ru), not "en".

    `en_GB-alan-medium` is a Piper model the TTS service does not offer, so the
    language it implies would commit the station to a language its new default
    voice (M2) was not chosen for. The DJ card shows the switch either way.
    """
    dj = _load(monkeypatch, tmp_path, LEGACY)["dj"]
    assert dj["language"] == config.DEFAULTS["dj"]["language"] == "ru"


def test_a_legacy_russian_voice_still_speaks_russian(monkeypatch, tmp_path):
    stored = json.loads(json.dumps(LEGACY))
    stored["dj"]["voice"] = "ru_RU-irina-medium"
    dj = _load(monkeypatch, tmp_path, stored)["dj"]
    assert dj["language"] == "ru"


def test_an_explicit_language_is_never_overwritten(monkeypatch, tmp_path):
    stored = json.loads(json.dumps(LEGACY))
    stored["dj"]["language"] = "en"
    assert _load(monkeypatch, tmp_path, stored)["dj"]["language"] == "en"


def test_a_migrated_config_keeps_saved_tts_params(monkeypatch, tmp_path):
    """Re-loading must not re-derive them: the user's own values win."""
    stored = json.loads(json.dumps(LEGACY))
    stored["dj"]["tts_params"] = {"speed": 1.3, "sentence_pause": 0.6}
    params = _load(monkeypatch, tmp_path, stored)["dj"]["tts_params"]
    assert params == {"speed": 1.3, "sentence_pause": 0.6}
