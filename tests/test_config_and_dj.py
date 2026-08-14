"""Config, presets and DJ-logic tests (no network, no LLM)."""

from __future__ import annotations

from app import config, db, dj


def test_defaults_present():
    cfg = config.load_config()
    assert cfg["music_dir"]
    assert cfg["dj"]["talk_min"] >= 0
    assert cfg["dj"]["talk_max"] >= cfg["dj"]["talk_min"]
    assert cfg["stream"]["bitrate_kbps"] > 0


def test_save_config_deep_merges_and_persists():
    config.save_config({"dj": {"talk_min": 1, "talk_max": 9}})
    cfg = config.load_config(force=True)
    assert cfg["dj"]["talk_min"] == 1
    assert cfg["dj"]["talk_max"] == 9
    # Sibling keys inside the same section survive the patch.
    assert "style" in cfg["dj"]
    assert cfg["stream"]["bitrate_kbps"] == config.DEFAULTS["stream"]["bitrate_kbps"]


def test_config_survives_a_reload():
    config.save_config({"music_dir": "/tmp/songs"})
    config._CACHE = None
    assert config.load_config()["music_dir"] == "/tmp/songs"


def test_corrupt_config_falls_back_to_defaults():
    config.CONFIG_PATH.write_text("{ not json", "utf-8")
    config._CACHE = None
    assert config.load_config()["music_dir"] == config.DEFAULTS["music_dir"]


def test_get_dotpath():
    assert config.get("dj.sent_max") == config.DEFAULTS["dj"]["sent_max"]
    assert config.get("nope.nothing", "fallback") == "fallback"


# --- presets ---------------------------------------------------------------

def test_preset_roundtrip():
    db.save_preset("chill", {"playback": {"genres": ["Jazz"]}})
    assert db.get_preset("chill") == {"playback": {"genres": ["Jazz"]}}
    assert [p["name"] for p in db.list_presets()] == ["chill"]


def test_preset_name_is_unique_and_updates():
    db.save_preset("x", {"a": 1})
    db.save_preset("x", {"a": 2})
    assert len(db.list_presets()) == 1
    assert db.get_preset("x") == {"a": 2}


def test_delete_preset():
    db.save_preset("gone", {})
    assert db.delete_preset("gone") is True
    assert db.delete_preset("gone") is False


# --- DJ script hygiene -----------------------------------------------------

def test_fallback_script_uses_available_metadata():
    text = dj.fallback_script({"title": "T", "artist": "A", "year": "1999"})
    assert "T" in text and "A" in text
    # Years are spelled out so TTS reads them as dates, not numerals.
    assert "nineteen ninety-nine" in text


def test_fallback_script_handles_missing_metadata():
    text = dj.fallback_script({})
    assert text.strip()


def test_clean_script_strips_reasoning_and_markdown():
    raw = "<think>hmm let me think</think>**Hello** there. Second one. Third. Fourth."
    out = dj._clean_script(raw, max_sentences=2)
    assert "think" not in out.lower()
    assert "*" not in out
    assert out.startswith("Hello there.")
    assert "Third" not in out


def test_clean_script_respects_sentence_cap():
    raw = "One. Two. Three. Four. Five."
    assert dj._clean_script(raw, 3) == "One. Two. Three."
    assert dj._clean_script(raw, 1) == "One."


def test_clean_script_on_empty_input():
    assert dj._clean_script("", 3) == ""
    assert dj._clean_script("<think>only reasoning</think>", 3) == ""


def test_facts_block_marks_unknown_fields():
    """Missing tags must be declared, otherwise small models invent them."""
    block = dj._facts_block({"title": "T", "artist": "A"}, {})
    assert "UNKNOWN" in block
    assert "album" in block and "year" in block


def test_facts_block_omits_unknown_line_when_complete():
    block = dj._facts_block(
        {"title": "T", "artist": "A", "album": "Al", "genre": "G", "year": "2000"}, {}
    )
    assert "UNKNOWN" not in block


def test_facts_block_excludes_misleading_musicbrainz_release_fields():
    """release_title / first_release_date caused confident false claims."""
    block = dj._facts_block(
        {"title": "T", "artist": "A"},
        {"release_title": "Live In Nowhere", "first_release_date": "2007-07-11"},
    )
    assert "Live In Nowhere" not in block
    assert "2007" not in block


def test_synthesize_rejects_empty_text():
    assert dj.synthesize("") is None
    assert dj.synthesize("   ") is None
