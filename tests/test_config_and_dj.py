"""Config, presets and DJ-logic tests (no network, no LLM)."""

from __future__ import annotations

from app import config, db, dj, icecast_server


def test_icecast_template_resolves_outside_the_image(tmp_path, monkeypatch):
    """A checkout must be able to render the managed Icecast config.

    Regression: the template path was pinned to ``/app/icecast.xml.tmpl``, the
    layout the Dockerfile creates — so bare metal, ``run.sh`` and the systemd
    unit blew up with FileNotFoundError while the container worked, which hid it.
    """
    assert icecast_server.ManagedIcecast.template_path().endswith("icecast.xml.tmpl")
    import os
    assert os.path.exists(icecast_server.ManagedIcecast.template_path())


def test_rendered_icecast_config_is_bounded_and_substituted(tmp_path, monkeypatch):
    monkeypatch.setattr(icecast_server, "_RENDERED_PATH", str(tmp_path / "icecast.xml"))
    monkeypatch.setattr(config, "_CACHE", {
        "icecast": {"enabled": True, "port": 8123, "mount": "virtualdj",
                    "hostname": "boombox", "source_password": "srcpw",
                    "public_port": 8123, "public_host": ""},
        "stream": {"station_name": "Virtual DJ"},
    })
    text = open(icecast_server.SERVER.render_config(), encoding="utf-8").read()
    assert "<source-password>srcpw</source-password>" in text
    assert "<hostname>boombox</hostname>" in text
    # The mount block must describe the mount the PUSHER streams to: it used to
    # be hardcoded to /virtualdj, so a configured mount was declared nowhere and
    # lost its stream-name/mount-type while the pusher used it anyway.
    assert "<mount-name>/virtualdj</mount-name>" in text
    # Icecast rotates its own access/error logs; without logsize they would grow
    # forever in /tmp/icecast next to the rendered config.
    assert "<logsize>10000</logsize>" in text


def test_rendered_icecast_config_follows_the_configured_mount(tmp_path, monkeypatch):
    monkeypatch.setattr(icecast_server, "_RENDERED_PATH", str(tmp_path / "icecast.xml"))
    monkeypatch.setattr(config, "_CACHE", {
        "icecast": {"enabled": True, "port": 8123, "mount": "/boombox & co",
                    "hostname": "boombox", "source_password": "srcpw"},
        "stream": {"station_name": "Radio & Roll"},
    })
    text = open(icecast_server.SERVER.render_config(), encoding="utf-8").read()
    # Mount and station name are escaped: one "&" in the config used to render
    # an icecast.xml that icecast could not parse at all.
    assert "<mount-name>/boombox &amp; co</mount-name>" in text
    assert "<stream-name>Radio &amp; Roll</stream-name>" in text


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


def test_the_fallback_intro_never_announces_a_program():
    """The templated intro introduces a SONG, not the station's scheduling.

    It used to open with the themed program the track belonged to ("Next up, a
    run of Rock", "Звучит подборка в стиле Rock") — the user directive is that
    the program is never spoken in either language.
    """
    track = {"id": 1, "title": "T", "artist": "A", "year": "1999"}
    for language in (None, "english", "russian"):
        text = dj.fallback_script(track, language=language)
        assert "T" in text and "A" in text, text
        low = text.lower()
        for word in ("program", "run of", "spotlight", "подборка", "шоу",
                     "блок", "программ"):
            assert word not in low, f"{language}: {word!r} in {text!r}"


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


def test_config_holding_a_non_object_does_not_take_the_app_down():
    """A JSON array or scalar in config.json must not crash the merge.

    `_deep_merge` calls `.items()` on whatever was read, so a valid-JSON-but-not-
    an-object file used to raise AttributeError at startup — a much worse failure
    than falling back to the defaults.
    """
    config.CONFIG_PATH.write_text("[1, 2, 3]", "utf-8")
    config._CACHE = None
    assert config.load_config()["stream"]["bitrate_kbps"] == \
        config.DEFAULTS["stream"]["bitrate_kbps"]


def test_underscore_comment_keys_never_reject_a_write():
    """`config.example.json` documents every knob with `_`-prefixed comment keys
    and the README tells users to copy it to data/config.json — a stored comment
    must not turn a later config write into a 422."""
    assert config.unknown_paths({
        "_comment": "docs", "playback": {"_genres": "docs", "repeat_window": 40},
    }) == []
    # A genuine typo is still reported.
    assert config.unknown_paths({"playback": {"repeat_windows": 40}}) == \
        ["playback.repeat_windows"]
