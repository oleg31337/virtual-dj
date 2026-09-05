"""Tests for voice-driven DJ language: a Russian voice makes the DJ speak Russian."""

import sys
sys.path.insert(0, ".")

from app import dj, config


def _fake_run_factory(calls: dict):
    """subprocess.run replacement that writes the wav/mp3 the pipeline expects."""
    def fake_run(cmd, **kw):
        if "-f" in cmd:  # piper -> write the wav it would have produced
            wav = cmd[cmd.index("-f") + 1]
            with open(wav, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVE")  # minimal valid-ish header
            calls["text"] = kw.get("input", b"").decode("utf-8", "replace")
            calls["cmd"] = list(cmd)
        else:  # ffmpeg -> write the mp3 output (last positional arg)
            out = cmd[-1]
            with open(out, "wb") as f:
                f.write(b"ID3 fake mp3")

        class R:
            returncode = 0
            stderr = b""
        return R()
    return fake_run


def _patch_synth(monkeypatch, calls):
    monkeypatch.setattr(dj, "voice_model_path",
                        lambda v=None: config.VOICES_DIR / f"{v}.onnx")
    monkeypatch.setattr(dj, "_cache_path",
                        lambda *a, **k: config.DJ_CACHE_DIR / "x.mp3")
    monkeypatch.setattr(dj, "piper_binary", lambda: "piper")
    monkeypatch.setattr(dj, "audio_duration", lambda p: 1.0)
    monkeypatch.setattr(dj.subprocess, "run", _fake_run_factory(calls))
    monkeypatch.setattr(dj, "shutil", __import__("shutil"))


def test_russian_dates_left_as_digits():
    # Russian voice reads year numerals natively — years/dates must NOT be
    # converted to words on the Russian path. (English path still spells them.)
    assert "1984" in dj._clean_script("в 1984 году", 3, language="russian")
    # And the English path does spell them out.
    assert "nineteen eighty-four" in dj._clean_script(
        "in 1984", 3, language="english")


def test_voice_language():
    assert dj.voice_language("ru_RU-irina-medium") == "russian"
    assert dj.voice_language("ru_RU-denis-medium") == "russian"
    assert dj.voice_language("en_US-amy-medium") == "english"
    assert dj.voice_language("en_GB-alan-medium") == "english"
    assert dj.voice_language(None) == "english"
    # Un-curated but installed Russian voice is still Russian.
    assert dj.voice_language("ru_RU-someone-medium") == "russian"


def test_synthesize_russian_voice_skips_translit_and_spelling(monkeypatch):
    calls: dict = {}
    _patch_synth(monkeypatch, calls)

    out = dj.synthesize("Песня Агаты Кристи, 1984 года",
                        voice="ru_RU-irina-medium")
    assert out is not None
    # Russian voice model is selected for the piper call.
    assert "ru_RU-irina-medium" in calls["cmd"][calls["cmd"].index("-m") + 1]
    # Cyrillic passed through verbatim (NO transliteration for Russian).
    assert "Агаты Кристи" in calls["text"]
    assert "Agata" not in calls["text"]
    # Year left as digits (Russian voice reads them natively), NOT spelled out.
    assert "1984" in calls["text"]


def test_synthesize_english_voice_transliterates_russian_names(monkeypatch):
    calls: dict = {}
    _patch_synth(monkeypatch, calls)

    out = dj.synthesize("Агата Кристи", voice="en_US-amy-medium")
    assert out is not None
    # English voice path: transliteration applied, no raw Cyrillic.
    assert "Agata Kristi" in calls["text"]
    assert "Агата" not in calls["text"]


def test_synthesize_defaults_to_configured_voice(monkeypatch):
    calls: dict = {}
    _patch_synth(monkeypatch, calls)
    monkeypatch.setitem(config.DEFAULTS["dj"], "voice", "ru_RU-irina-medium")

    out = dj.synthesize("Привет, слушатели!")
    assert out is not None
    assert "ru_RU-irina-medium" in calls["cmd"][calls["cmd"].index("-m") + 1]
    assert "Привет" in calls["text"]  # no transliteration for RU voice


def test_voice_profiles_split_by_language():
    ru = dj.voice_profiles("russian")
    en = dj.voice_profiles("english")
    assert all(p["lang"] == "russian" for p in ru)
    assert all(p["lang"] == "english" for p in en)
    ids = {p["id"] for p in ru}
    assert {"ru_RU-irina-medium", "ru_RU-denis-medium",
            "ru_RU-dmitri-medium", "ru_RU-ruslan-medium"} <= ids


def test_fallback_script_russian():
    track = {"title": "Котики-наркотики", "artist": "Мёртвые Дельфины",
             "year": "2007"}
    out = dj.fallback_script(track, language="russian")
    assert "Котики-наркотики" in out
    # Year left as digits (Russian voice reads them natively), not spelled out.
    assert "2007" in out
    assert "две тысячи семь" not in out
    # Must be Cyrillic (not transliterated).
    assert "Котики" in out
