"""Tests for Russian-language DJ voice routing and year spelling."""

import sys
import subprocess as _sp
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
        # capture stdin text for the piper invocation if present

        class R:
            returncode = 0
            stderr = b""
        return R()
    return fake_run


def _patch_synth(monkeypatch, calls, russian=True):
    voice = "ru_RU-irina-medium" if russian else "en_US-amy-medium"
    monkeypatch.setattr(dj, "voice_model_path",
                        lambda v=None: config.VOICES_DIR / f"{voice}.onnx")
    monkeypatch.setattr(dj, "_cache_path",
                        lambda *a, **k: config.DJ_CACHE_DIR / "x.mp3")
    monkeypatch.setattr(dj, "piper_binary", lambda: "piper")
    monkeypatch.setattr(dj, "audio_duration", lambda p: 1.0)
    monkeypatch.setattr(dj.subprocess, "run", _fake_run_factory(calls))
    monkeypatch.setattr(dj, "shutil", __import__("shutil"))


def test_russian_year_spelling():
    assert dj._int_to_words_ru(1984) == "тысяча девятьсот восемьдесят четыре"
    assert dj._int_to_words_ru(2024) == "две тысячи двадцать четыре"
    assert dj._spell_dates_ru("from 1984") == \
        "from тысяча девятьсот восемьдесят четыре"


def test_synthesize_routes_russian_voice_and_skips_translit(monkeypatch):
    calls: dict = {}
    _patch_synth(monkeypatch, calls, russian=True)
    monkeypatch.setitem(config.DEFAULTS["dj"], "russian_voice", "ru_RU-irina-medium")

    out = dj.synthesize("Песня Агаты Кристи, 1984 года", language="russian")
    assert out is not None
    # Russian voice model is selected for the piper call.
    assert "ru_RU-irina-medium" in calls["cmd"][calls["cmd"].index("-m") + 1]
    # Cyrillic passed through verbatim (NO transliteration for Russian).
    assert "Агаты Кристи" in calls["text"]
    assert "Agata" not in calls["text"]
    # Year spelled in Russian words.
    assert "тысяча девятьсот восемьдесят четыре" in calls["text"]


def test_synthesize_english_path_transliterates_russian_names(monkeypatch):
    calls: dict = {}
    _patch_synth(monkeypatch, calls, russian=False)

    out = dj.synthesize("Агата Кристи", language="english")
    assert out is not None
    # English voice path: transliteration applied, no raw Cyrillic.
    assert "Agata Kristi" in calls["text"]
    assert "Агата" not in calls["text"]


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
             "year": "2007", "language": "russian"}
    out = dj.fallback_script(track, language="russian")
    assert "Котики-наркотики" in out
    # 2007 -> "две тысячи семь" (Russian year words, not digits).
    assert "две тысячи семь" in out
    # Must be Cyrillic (not transliterated).
    assert "Котики" in out
