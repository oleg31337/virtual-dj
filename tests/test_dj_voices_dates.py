"""Tests for DJ date spelling (TTS must read years as words) and voice catalogue."""

from __future__ import annotations

from app import dj


def test_year_spelled_out_as_words():
    assert dj._spell_dates("From 1984 by the band.") == "From nineteen eighty-four by the band."
    assert dj._spell_dates("Hits from 2023 and 1999.") == "Hits from twenty twenty-three and nineteen ninety-nine."
    # Years 2000-2009 must not read as "twenty seven" (sounds like 27).
    assert dj._spell_dates("A track from 2007.") == "A track from two thousand seven."
    assert dj._spell_dates("From 2000.") == "From two thousand."


def test_non_year_numbers_left_alone():
    # Counts / non-year numbers should not be mangled.
    assert dj._spell_dates("Top 10 of 1984.") == "Top 10 of nineteen eighty-four."
    assert dj._spell_dates("We have 3 tracks left.") == "We have 3 tracks left."


def test_clean_script_spells_dates():
    out = dj._clean_script("This one is from 1991, enjoy.", 3)
    assert "nineteen ninety-one" in out
    assert "1991" not in out


def test_fallback_script_year_is_spoken():
    track = {"artist": "The Band", "title": "Old Song", "year": "1977"}
    text = dj.fallback_script(track)
    assert "nineteen seventy-seven" in text
    assert "1977" not in text


def test_voice_profiles_marks_installed():
    profiles = dj.voice_profiles()
    ids = {p["id"] for p in profiles}
    # The five curated voices are present.
    assert "en_US-amy-medium" in ids
    assert "en_US-lessac-medium" in ids
    assert "en_US-libritts_r-medium" in ids
    assert "en_US-ryan-medium" in ids
    assert "en_US-bryce-medium" in ids
    # Each carries an intonation note and installed flag.
    for p in profiles:
        assert "note" in p
        assert "installed" in p
