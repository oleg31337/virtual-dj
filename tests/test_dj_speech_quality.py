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

import json

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


def _fake_llm(monkeypatch, replies="A short intro about a song."):
    """Install a stubbed Ollama chat endpoint; return the list of requests sent.

    ``replies`` is one string (always the same answer) or a list consumed in
    order, the last one repeating — enough to drive the retry paths.
    """
    sent: list[dict] = []
    answers = [replies] if isinstance(replies, str) else list(replies)

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": self._content}}

    def fake_post(url, **kw):
        sent.append(kw.get("json", {}))
        return FakeResponse(answers[min(len(sent) - 1, len(answers) - 1)])

    monkeypatch.setattr(dj.httpx, "post", fake_post)
    monkeypatch.setattr(config, "_CACHE", {
        "llm": {"enabled": True, "base_url": "http://x", "model": "m",
                "retries": 0}})
    return sent


def _capture_llm_messages(monkeypatch, track, **kwargs):
    """Run generate_script against a stubbed LLM and return (script, system, user).

    Lets a test assert on the exact prompt without a live model.
    """
    sent = _fake_llm(monkeypatch)
    script = dj.generate_script(track, **kwargs)
    assert sent, "the LLM must actually be called"
    by_role = {m["role"]: m["content"] for m in sent[0].get("messages", [])}
    return script, by_role.get("system", ""), by_role.get("user", "")


def test_a_line_naming_a_prompt_example_is_thrown_away(monkeypatch, caplog):
    """The model copies the rule examples into the line (the reported bug).

    Measured on nemotron-3-nano:4b for "Mindfields" by Prodigy: "Mindfields by
    Pink, the Prodigy", "Mindfields by Pink and Keesha" — the track carries no
    such name. The prompt cannot stop it, so the output is guarded.
    """
    track = {"id": 1, "title": "Mindfields", "artist": "Prodigy", "album": "The Fat Of The Land"}
    sent = _fake_llm(monkeypatch, [
        "Mindfields by Pink and Keesha, from The Fat Of The Land",
        "Mindfields is a track from The Fat Of The Land",
    ])
    with caplog.at_level("INFO"):
        script = dj.generate_script(track)
    assert "Pink" not in script and "Keesha" not in script
    assert "The Fat Of The Land" in script
    assert len(sent) == 2, "the contaminated line must be asked for again"
    assert "prompt example" in caplog.text


def test_a_line_naming_the_tracks_own_artist_is_fine(monkeypatch):
    """The guard must not reject a line for naming the artist it introduces."""
    sent = _fake_llm(monkeypatch, "Right now: Pink Floyd with Time")
    script = dj.generate_script(
        {"id": 1, "title": "Time", "artist": "Pink Floyd"})
    assert "Pink Floyd" in script
    assert len(sent) == 1, "a good line must not be asked for twice"


def test_imagery_is_not_mistaken_for_the_example_artist():
    """The guard is case-sensitive on purpose: only the NAME is a leak."""
    prodigy = {"id": 1, "title": "Mindfields", "artist": "Prodigy"}
    assert dj._usable_script("A pink neon glow over the city", prodigy)
    assert not dj._usable_script("Pink and the Prodigy, live", prodigy)
    assert not dj._usable_script("Keesha is back", prodigy)
    assert not dj._usable_script("dead mouse on the decks", prodigy)
    # A name the track itself carries is not a leak.
    floyd = {"id": 2, "title": "Time", "artist": "Pink Floyd"}
    assert dj._usable_script("Pink Floyd, live in nineteen seventy-two", floyd)


def test_a_looping_line_is_rejected(monkeypatch, caplog):
    """Measured: a Russian line came back as "да-да-да-…" eighty times over."""
    sent = _fake_llm(monkeypatch, ["да-" * 40, "Сейчас играет Земфира."])
    with caplog.at_level("INFO"):
        script = dj.generate_script({"id": 1, "title": "T", "artist": "Zemfira"},
                                    voice="ru_RU-irina-medium")
    assert "Земфира" in script and "да-да" not in script
    assert len(sent) == 2
    assert "looping" in caplog.text


def test_two_bad_lines_fall_back_to_the_template(monkeypatch):
    sent = _fake_llm(monkeypatch, "Mindfields by Pink and Keesha")
    script = dj.generate_script(
        {"id": 1, "title": "Mindfields", "artist": "Prodigy", "year": "1997"})
    assert script == dj.fallback_script(
        {"id": 1, "title": "Mindfields", "artist": "Prodigy", "year": "1997"})
    assert len(sent) == 2, "one retry, not an endless loop"


def test_rule_examples_do_not_sit_next_to_the_song_facts(monkeypatch):
    """The pronunciation examples are illustrations, so they travel in the system message.

    In the user prompt they were read as song data: for "Mindfields" by Prodigy,
    a track with no such field, nemotron-3-nano:4b opened with "Mindfields by
    Pink, the Prodigy" (straight out of the "P!nk is Pink" example), then
    "Mindfields by Pink and the Prodigy", and recited the rules' own
    "two songs ... not 2 songs" phrasing as content.
    """
    script, system, user = _capture_llm_messages(
        monkeypatch, {"id": 1, "title": "Mindfields", "artist": "Prodigy"})
    assert script == "A short intro about a song."
    for example in ("P!nk", "Ke$ha", "deadmau5", "Blink-182", "Агата", "Ага\u00b4та"):
        assert example not in user, f"{example!r} is an illustration, not song data"
    assert "P!nk" in system, "the rule example belongs in the system message"
    assert "ILLUSTRATE" in system and "never be said" in system
    assert "Say only the names listed above." in user


def test_one_rule_set_per_language_not_two(monkeypatch):
    """Russian needs years as digits; the English rules demand words for them.

    Both used to be handed over at once, leaving a small model to pick.
    """
    _, system, user = _capture_llm_messages(
        monkeypatch, {"id": 1, "title": "МЫ РАЗБИВАЕМСЯ", "artist": "Zemfira"},
        voice="ru_RU-irina-medium")
    assert "ПО-РУССКИ" in system
    for english_only in ("nineteen ninety-nine", "P!nk", "deadmau5"):
        assert english_only not in system, english_only
        assert english_only not in user, english_only


def test_russian_rules_still_demand_a_spoken_sentence(monkeypatch):
    """Splitting the rule sets by language cost Russian its anti-list rule.

    The English block used to ride along with the Russian one and opened with
    "write it the way it will be SPOKEN, not as a catalogue listing". Measured
    after the split: half the Russian lines came back as bare catalogue lists —
    "Fly Away, Lenny Kravitz, rock", "Наш дом, Mashina Vremeni, 1993, Rock".
    """
    _, system, _ = _capture_llm_messages(
        monkeypatch, {"id": 1, "title": "T", "artist": "A"},
        voice="ru_RU-irina-medium")
    assert "как список из каталога" in system
    assert "короткими фразами" in system
    assert "Не называй жанр" in system


def test_the_rules_forbid_announcing_the_genre(monkeypatch):
    """Even three style tags got recited: "...make rock electronic pop"."""
    _, system, _ = _capture_llm_messages(
        monkeypatch, {"id": 1, "title": "T", "artist": "A"})
    assert "Never announce the genre" in system


def test_the_prompt_never_mentions_the_program(monkeypatch):
    """The DJ is not told about the themed program the track belongs to.

    It used to be: a track that opened a themed block got "This song opens a new
    program of <label> music. Welcome the listeners into that vibe...", and the
    DJ duly said things like "we are switching to a new program". The program
    stays a playlist device — it decides what plays — and is never spoken.
    """
    script, system, user = _capture_llm_messages(
        monkeypatch,
        {"id": 1, "title": "Time", "artist": "Pink Floyd", "genre": "Electronic"},
        previous={"title": "Yesterday", "artist": "The Beatles"})
    assert script == "A short intro about a song."
    for word in ("Trip-Hop", "program", "vibe", "spotlight", "run of", "set"):
        assert word.lower() not in (system + user).lower(), (
            f"{word!r} leaked into the DJ prompt")


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
