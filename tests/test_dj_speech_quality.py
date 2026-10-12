"""DJ text preparation and the voice knobs added for speech quality.

Covers what the DJ's text must look like before it reaches Piper, and that the
new synthesis settings actually reach the command line:

  * stress marks are rewritten into the form espeak-ng obeys (U+0301 after the
    vowel) — the plain acute the prompt used to ask for is silently ignored, and
    some plausible stand-ins are read ALOUD as their character names;
  * non-Russian text has stress marks stripped (they mean nothing to English);
  * `dj.noise_w` and `dj.sentence_silence` are passed to Piper and are part of
    the audio cache key, so a cached line is never reused across settings.
"""

from __future__ import annotations

import pytest

from app import config, dj

CA = "\u0301"          # combining acute: the ONLY mark espeak obeys


# --- stress marks ------------------------------------------------------------

@pytest.mark.parametrize("written,expected", [
    ("замо\u00b4к", "замо" + CA + "к"),        # plain acute after the vowel
    ("з\u00b4амок", "за" + CA + "мок"),        # written BEFORE the vowel
    ("за\u02b9мок", "за" + CA + "мок"),        # modifier prime (was read aloud)
    ("за\u02bcмок", "за" + CA + "мок"),        # modifier letter (was read aloud)
    ("за\u2019мок", "за" + CA + "мок"),        # typographic apostrophe
    ("музыка\u00b4льный", "музыка" + CA + "льный"),
    ("просто текст", "просто текст"),          # nothing to do
    ("замо" + CA + "к", "замо" + CA + "к"),    # already correct: not doubled
    ("текст \u00b4 без гласной", "текст  без гласной"),   # no vowel: dropped
])
def test_stress_marks_become_the_form_espeak_obeys(written, expected):
    assert dj._normalize_stress_marks(written) == expected


def test_a_mark_is_never_left_to_be_read_aloud():
    """U+02B9/U+02BC made espeak say "zˈɛ a ɪdʒˈɛktɪv ˈɛm ʌ k" before this."""
    for mark in ("\u00b4", "\u02b9", "\u02bc", "\u2019", "\u2032", "`"):
        assert mk_survives(dj._normalize_stress_marks(f"за{mark}мок")) is False


def mk_survives(text: str) -> bool:
    return any(ch in text for ch in ("\u00b4", "\u02b9", "\u02bc", "\u2019", "\u2032", "`"))


def test_english_text_has_stress_marks_stripped():
    assert dj._strip_stress_marks("P\u00b4ink and Ke\u02b9sha") == "Pink and Kesha"


def test_russian_script_cleanup_normalises_marks():
    out = dj._clean_script("Слу\u00b4шай, ага\u02b9та кри\u00b4сти.", 3,
                           language="russian")
    assert CA in out, "the obeyed mark must survive into the script"
    assert all(mark not in out for mark in ("\u00b4", "\u02b9", "\u02bc"))


def test_english_script_cleanup_spells_years_and_drops_marks():
    out = dj._clean_script("From 1999 by P\u00b4ink.", 3, language="english")
    assert "1999" not in out and "nineteen ninety-nine" in out
    assert "\u00b4" not in out


# --- the two new Piper knobs -------------------------------------------------

def _fake_piper(monkeypatch, calls: dict):
    """Record what synthesis asks Piper for, without running Piper."""
    def fake_run(cmd, **kw):
        if "-f" in cmd:
            with open(cmd[cmd.index("-f") + 1], "wb") as fh:
                fh.write(b"RIFF\x00\x00\x00\x00WAVE")
            calls["cmd"] = list(cmd)
        else:
            with open(cmd[-1], "wb") as fh:
                fh.write(b"ID3 fake")

        class R:
            returncode = 0
            stderr = b""
        return R()

    monkeypatch.setattr(dj, "voice_model_path",
                        lambda v=None: config.VOICES_DIR / "en_GB-alan-medium.onnx")
    monkeypatch.setattr(dj, "piper_binary", lambda: "piper")
    monkeypatch.setattr(dj.subprocess, "run", fake_run)


def test_new_knobs_reach_the_piper_command(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DJ_CACHE_DIR", tmp_path)
    calls: dict = {}
    _fake_piper(monkeypatch, calls)
    monkeypatch.setattr(config, "_CACHE", {
        "dj": {"voice": "en_GB-alan-medium", "noise_w": 0.55, "sentence_silence": 0.4},
        "stream": {"sample_rate": 22050, "channels": 2, "bitrate_kbps": 128},
    })
    assert dj.synthesize("A line to say.") is not None
    cmd = calls["cmd"]
    assert "--noise_w_scale" in cmd, cmd
    assert cmd[cmd.index("--noise_w_scale") + 1] == "0.55"
    assert "--sentence_silence" in cmd
    assert cmd[cmd.index("--sentence_silence") + 1] == "0.4"


def test_the_cache_key_includes_the_new_knobs(tmp_path, monkeypatch):
    """Two renders of one line with different rhythm settings must not collide."""
    monkeypatch.setattr(config, "DJ_CACHE_DIR", tmp_path)
    a = dj._cache_path("line", "en_GB-alan-medium", 1.0, 0.667, 0.8, 0.25)
    b = dj._cache_path("line", "en_GB-alan-medium", 1.0, 0.667, 0.5, 0.25)
    c = dj._cache_path("line", "en_GB-alan-medium", 1.0, 0.667, 0.8, 0.5)
    assert len({a.name, b.name, c.name}) == 3


def test_the_llm_answering_produces_a_prepared_script(monkeypatch):
    """The branch that a fuzzy edit broke: Ollama answers, so `content` is set.

    Regression for a NameError ('raw') that made EVERY DJ break die inside the
    prefetch thread — the LLM ran (GPU load) and no audio was ever prepared,
    exactly the symptom reported from a host with Ollama. No test covered this
    path because CI has no Ollama and always took the fallback branch.
    """
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": "That was from 1999 — a classic."}}

    called: list[tuple] = []

    def fake_post(url, **kw):
        called.append((url, kw))
        return FakeResponse()

    monkeypatch.setattr(dj.httpx, "post", fake_post)
    monkeypatch.setattr(config, "_CACHE", {"llm": {"enabled": True, "base_url": "http://x",
                                                   "model": "m", "retries": 0}})

    script = dj.generate_script({"id": 1, "title": "Time", "artist": "Pink Floyd",
                                 "album": "DSOTM", "year": "1973"})
    assert called, "the LLM must actually be called"
    assert script, "the LLM's answer must come back as a script"
    assert "1999" not in script, f"years must be spoken, got {script!r}"
    assert "nineteen ninety-nine" in script


def test_a_missing_voice_is_named_and_stops_the_llm_call(monkeypatch, tmp_path, caplog):
    """The silent-DJ trap: the configured voice has no model file.

    Symptom on a host that pulled a newer default: the LLM still runs (GPU
    spike), no DJ audio, and nothing in the UI. Now the reason is named in one
    warning and the LLM call is skipped instead of being wasted.
    """
    monkeypatch.setattr(config, "VOICES_DIR", tmp_path / "voices")   # empty
    monkeypatch.setattr(dj, "piper_binary", lambda: "/usr/bin/piper")
    monkeypatch.setattr(dj, "_tts_warned", False)

    problem = dj.tts_problem("en_GB-alan-medium")
    assert problem and "en_GB-alan-medium" in problem, problem
    assert "not installed" in problem

    called: list[str] = []
    monkeypatch.setattr(dj, "generate_script",
                        lambda *a, **k: called.append("llm") or "a script")
    with caplog.at_level("WARNING", logger="app.dj"):
        assert dj.prepare_break({"id": 1, "artist": "A", "title": "T"}) is None
    assert called == [], "no LLM call when the voice cannot speak"
    assert any("DJ breaks are OFF" in r.getMessage() for r in caplog.records), (
        f"the reason must be logged once: {caplog.text}")

    # ...and once the model appears, breaks work again (no sticky state).
    (tmp_path / "voices").mkdir(exist_ok=True)
    (tmp_path / "voices" / "en_GB-alan-medium.onnx").write_bytes(b"model")
    monkeypatch.setattr(dj, "voice_model_path",
                        lambda v=None: (tmp_path / "voices" / f"{v}.onnx"))
    monkeypatch.setattr(dj, "synthesize", lambda *a, **k: tmp_path / "x.mp3")
    monkeypatch.setattr(dj, "audio_duration", lambda p: 1.0)
    monkeypatch.setattr(dj, "db", __import__("app.db", fromlist=["db"]))
    assert dj.tts_problem("en_GB-alan-medium") is None


def test_health_payload_carries_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(dj, "piper_binary", lambda: "/usr/bin/piper")
    health = dj.tts_health()
    assert health["ok"] is False
    assert "en_GB-alan-medium" in health["problem"]
    assert health["voices_available"] == []


def test_the_default_voice_is_the_measured_one():
    """alan-medium moves its pitch ~19 semitones; amy-medium measured ~7."""
    assert config.DEFAULTS["dj"]["voice"] == "en_GB-alan-medium"
    assert config.DEFAULTS["dj"]["noise_w"] == 0.8
    assert config.DEFAULTS["dj"]["sentence_silence"] == 0.25
