"""Tests for tag guessing, text-quality filtering and web-confirmation plumbing.

These never touch the real music library; they build tiny temp trees and stub
the network with monkeypatched helpers, so they run in CI in well under a
second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app import textq  # noqa: E402
from app import websearch  # noqa: E402


# --------------------------------------------------------------------------
# text quality
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,usable", [
    ("Enter Sandman", True),
    ("Björk", True),
    ("Café Tacvba", True),
    ("- - - - Metallica - - - -", True),  # decoration stripped -> real name
    ("01", False),
    ("Track 03", False),
    ("Unknown", False),
    ("untitled", False),
    ("A", False),
    ("", False),
])
def test_is_usable(value, usable):
    assert textq.is_usable(value) is usable


def test_non_latin_scripts():
    assert textq.has_non_latin_script("Синагогальная музыка")
    assert textq.has_non_latin_script("東京")
    assert not textq.has_non_latin_script("Café Tacvba")


def test_mojibake_detected():
    assert textq.looks_like_mojibake("Ñèíàãîãàëüíàÿ")
    assert not textq.looks_like_mojibake("Café Tacvba")


def test_rejection_reason_priority():
    # Cyrillic wins over missing title.
    assert textq.rejection_reason("Синагогальная", None) == "non_latin"
    assert textq.rejection_reason("Metallica", None) == "no_title"
    assert textq.rejection_reason(None, "Fade To Black") == "no_artist"
    assert textq.rejection_reason("Metallica", "Fade To Black") is None


# --------------------------------------------------------------------------
# path guessing
# --------------------------------------------------------------------------

def _touch(tmp: Path, rel: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00\x00")
    return p


def test_guess_artist_title_filename(tmp_path):
    p = _touch(tmp_path, "Metallica - Fade To Black.mp3")
    g = textq.guess_from_path(p, tmp_path)
    assert g["artist"] == "Metallica"
    assert g["title"] == "Fade To Black"


def test_guess_folder_layout(tmp_path):
    p = _touch(tmp_path, "Caravan Palace/Clash/03 Lone Digger.mp3")
    g = textq.guess_from_path(p, tmp_path)
    assert g["artist"] == "Caravan Palace"
    assert g["title"] == "Lone Digger"
    assert g["album"] == "Clash"


def test_guess_skips_container_folders(tmp_path):
    p = _touch(tmp_path, "Red Hot Chili Peppers/Mp3/1987-Album/04 Me & My Friends.mp3")
    g = textq.guess_from_path(p, tmp_path)
    assert g["artist"] == "Red Hot Chili Peppers"
    assert g["title"] == "Me & My Friends"


def test_guess_underscore_separator(tmp_path):
    p = _touch(tmp_path, "Snap/CD2_-_08_-_Snap_-_Do_You_See_The_Light.mp3")
    g = textq.guess_from_path(p, tmp_path)
    assert g["artist"] == "Snap"
    assert g["title"] == "Do You See The Light"


def test_guess_cyrillic_folder_not_usable(tmp_path):
    p = _touch(tmp_path, "Russian/Игорек - Подождем.mp3")
    g = textq.guess_from_path(p, tmp_path)
    # Path has a usable title but a non-Latin artist -> reportable, not a guess.
    assert g["title"] == "Подождем"


# --------------------------------------------------------------------------
# web confirmation (network stubbed)
# --------------------------------------------------------------------------

def test_canonical_genre_buckets():
    assert websearch.canonical_genre("heavy metal") == "Metal"
    assert websearch.canonical_genre("synth-pop") == "Pop"
    assert websearch.canonical_genre("Seen Live") is None
    assert websearch.canonical_genre("") is None


def test_confirm_track_uses_cache(monkeypatch):
    # Stub the provider so we can assert caching without the network. The stub
    # records how many real lookups it performed; a second call with the same
    # key must hit the SQLite cache instead.
    state = {"calls": 0}

    def fake_provider(client, artist, title):
        state["calls"] += 1
        return {"artist": artist, "title": title, "genre": "Pop",
                "score": 0.9, "provider": "stub"}

    monkeypatch.setattr(websearch, "_itunes_search", fake_provider)

    first = websearch.confirm_track("Pink", "Just Give Me A Reason")
    second = websearch.confirm_track("Pink", "Just Give Me A Reason")
    assert first["confirmed"] is True
    assert second["confirmed"] is True
    # One real provider call, the second served from cache.
    assert state["calls"] == 1, state


def test_confirm_track_disabled(monkeypatch):
    monkeypatch.setattr(websearch.config, "get",
                        lambda k, d=None: False if k == "websearch.enabled" else d)
    out = websearch.confirm_track("Metallica", "Fade To Black")
    assert out["confirmed"] is False
    assert out["confidence"] == 0.0
