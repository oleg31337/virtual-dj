"""The DJ's language and its Russian text shaping.

Two things changed with the engine swap and both are contracts worth pinning:

* the language is now an explicit setting (``dj.language``) validated against the
  languages the chosen voice declares — not a property of a voice id;
* Russian numerals are written out as words, because a neural voice invents
  numbers when it has to read digits ("в 1989 году" came out as 979/1089/1977 in
  different takes; the spelled year was correct every time).
"""

from __future__ import annotations

import pytest

from app import config, dj


def _set_config(monkeypatch, **dj_values):
    for key, value in dj_values.items():
        monkeypatch.setitem(config.DEFAULTS["dj"], key, value)
    config.load_config(force=True)


def test_language_follows_the_configured_setting(monkeypatch):
    import fake_tts

    fake_tts.install(monkeypatch)  # V1 speaks en+ru, V2 ru only, V3 en only
    _set_config(monkeypatch, voice="V1", language="ru")
    assert dj.language_for() == "ru"
    _set_config(monkeypatch, language="en")
    assert dj.language_for() == "en"


def test_language_cannot_exceed_what_the_voice_speaks(monkeypatch):
    import fake_tts

    # Every style installed, as the real Supertonic service has them.
    fake_tts.install(monkeypatch, voices=[dict(v, installed=True)
                                          for v in fake_tts.VOICES])
    # V3 declares English only: asking for Russian must land on English, because
    # the engine physically cannot speak it with that voice.
    _set_config(monkeypatch, voice="V3", language="ru")
    assert dj.language_for() == "en"
    assert dj.language_for("V3", "ru") == "en"
    # ... and a Russian-only voice cannot be asked for English either.
    assert dj.language_for("V2", "en") == "ru"


def test_explicit_language_argument_wins(monkeypatch):
    import fake_tts

    fake_tts.install(monkeypatch)
    _set_config(monkeypatch, voice="V1", language="ru")
    assert dj.language_for("V1", "en") == "en"


def test_language_survives_a_dead_service(monkeypatch):
    """With no capability data the configured value is all we have."""
    from app import tts_client

    monkeypatch.setattr(tts_client, "capabilities", lambda force=False: None)
    monkeypatch.setattr(tts_client, "voices", lambda: [])
    monkeypatch.setattr(tts_client, "voice", lambda vid: None)
    monkeypatch.setattr(tts_client, "default_voice", lambda: "")
    _set_config(monkeypatch, voice="M2", language="ru")
    assert dj.language_for() == "ru"


def test_russian_numbers_are_spelled_out(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)
    _set_config(monkeypatch, voice="V2", language="ru")
    out = dj.synthesize("Этот трек вышел в 1989 году, темп 120 BPM.")
    assert out is not None
    spoken = fake.calls[-1]["text"]
    assert "1989" not in spoken and "120" not in spoken
    assert "тысяча девятьсот восемьдесят девятом году" in spoken
    assert "сто двадцать ударов в минуту" in spoken


def test_russian_keeps_cyrillic_and_drops_espeak_stress_for_this_engine(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)  # declares stress_marks: false
    _set_config(monkeypatch, voice="V2", language="ru")
    assert dj.synthesize("Ага\u0301та Кристи") is not None
    spoken = fake.calls[-1]["text"]
    assert "Агата Кристи" in spoken
    assert "\u0301" not in spoken
    assert "Agata" not in spoken  # never transliterated on the Russian path


def test_an_espeak_engine_keeps_the_stress_marks(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch, stress_marks=True)
    _set_config(monkeypatch, voice="V2", language="ru")
    assert dj.synthesize("Ага\u00b4та Кристи") is not None
    assert "\u0301" in fake.calls[-1]["text"]


def test_english_transliterates_russian_names(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)
    _set_config(monkeypatch, voice="V1", language="en")
    assert dj.synthesize("Агата Кристи") is not None
    spoken = fake.calls[-1]["text"]
    assert "Agata Kristi" in spoken
    assert "Агата" not in spoken


def test_english_years_are_spelled_as_words(monkeypatch):
    import fake_tts

    fake = fake_tts.install(monkeypatch)
    _set_config(monkeypatch, voice="V1", language="en")
    assert dj.synthesize("A track from 1984.") is not None
    assert "nineteen eighty-four" in fake.calls[-1]["text"]


def test_fallback_script_russian_spells_the_year():
    track = {"title": "Котики-наркотики", "artist": "Мёртвые Дельфины",
             "year": "2007"}
    out = dj.fallback_script(track, language="ru")
    assert "Котики-наркотики" in out
    # The template's year is written out, not left as digits — and in the case
    # its own noun asks for ("2007 года" is genitive).
    assert "2007" not in out
    assert "две тысячи седьмого года" in out
    assert "Котики" in out  # still Cyrillic, never transliterated


def test_prepare_script_uses_the_configured_language(monkeypatch):
    import fake_tts

    fake_tts.install(monkeypatch)
    _set_config(monkeypatch, voice="V2", language="ru")
    out = dj.prepare_script("Приве\u0301т, слушаем хит 1984 года.")
    assert "1984" not in out and "тысяча девятьсот восемьдесят четвёртого" in out
    assert "\u0301" in out  # the display form keeps the mark
    # An explicit language still wins.
    english = dj.prepare_script("hit from 1984", language="en")
    assert "nineteen eighty-four" in english


@pytest.mark.parametrize("voice", ["V1", "V2"])
def test_voice_catalogue_is_whatever_the_service_says(monkeypatch, voice):
    import fake_tts

    fake_tts.install(monkeypatch)
    ids = {v["id"] for v in dj.voice_catalogue()}
    assert voice in ids
    assert dj.available_voices() == ["V1", "V2"]
