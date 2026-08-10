"""Integration test: scanning a small temp library applies the three-stage
metadata resolution and excludes unidentifiable / non-Latin tracks.

The web-confirmation stage is stubbed so the test never touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import make_mp3
from app import db, library, websearch

# A stubbed web search: confirms anything with a plausible artist+title, and
# invents a genre so we can assert the genre plumbing end-to-end.
def _stub_confirm(artist, title, use_cache=True):
    if artist and title and title != "Garbage":
        return {"confirmed": True, "genre": "Rock", "confidence": 0.9,
                "sources": ["stub"], "artist": artist, "title": title}
    return {"confirmed": False, "genre": None, "confidence": 0.0, "sources": []}


@pytest.fixture
def temp_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(websearch, "confirm_track", _stub_confirm)
    music = tmp_path / "mp3"
    music.mkdir()

    # Tagged correctly -> from tags.
    make_mp3(music / "Tagged" / "Real Artist - Real Song.mp3",
             tags={"title": "Real Song", "artist": "Real Artist", "genre": "Jazz"})

    # No tags, named "Artist - Title" -> path guess, confirmed on web.
    (music / "Guess Artist - Guess Song.mp3").write_bytes(b"x")

    # No usable title -> excluded (no_title).
    (music / "Just A Folder").mkdir()
    (music / "Just A Folder" / "Track 03.mp3").write_bytes(b"x")

    # Non-Latin artist -> excluded (non_latin), never even looked up.
    (music / "Russian").mkdir()
    (music / "Russian" / "Игорек - Подождем.mp3").write_bytes(b"x")

    # "Bad Band - Garbage.mp3" -> path guess with a title the stub web rejects
    # -> excluded as unconfirmed.
    (music / "Bad Band - Garbage.mp3").write_bytes(b"x")

    # MusicBrainz-ish folder layout.
    (music / "FolderBand" / "FolderAlbum").mkdir(parents=True)
    (music / "FolderBand" / "FolderAlbum" / "07 Folder Title.mp3").write_bytes(b"x")

    monkeypatch.setenv("VDJ_MUSIC_DIR", str(music))
    monkeypatch.setenv("VDJ_DB_PATH", str(tmp_path / "lib.db"))
    db.DB_PATH = str(tmp_path / "lib.db")
    db.init_db()
    # Fresh, isolated DB. Drop any rows from a prior run.
    db.connect().executescript("DELETE FROM tracks; DELETE FROM web_lookups;")
    return music


def test_scan_classifies_and_excludes(temp_lib, monkeypatch):
    snap = library.scan_library(str(temp_lib), full=True, use_web=True)
    assert snap["error"] is None
    assert snap["scanned"] == 6

    stats = library.library_stats()
    # playable: tagged(1) + path-guess confirmed(1) + folder layout(1) = 3
    assert stats["playable"] == 3, stats
    assert stats["excluded"] == 3, stats
    assert stats["unknown_reasons"].get("non_latin") == 1
    assert stats["unknown_reasons"].get("unconfirmed") == 1

    # The web-confirmed path-guess got its genre from the stub.
    rows = library.query_tracks(genres=["Rock"])
    titles = {r["title"] for r in rows}
    assert "Guess Song" in titles

    # Excluded tracks are reachable only via excluded_tracks().
    exc = library.excluded_tracks()
    assert len(exc) == 3
    reasons = {e["exclude_reason"] for e in exc}
    assert {"no_title", "non_latin", "unconfirmed"} <= reasons

    # And never enter a normal playlist query.
    ids_exc = {e["id"] for e in exc}
    playlist_ids = {r["id"] for r in library.query_tracks(limit=1000)}
    assert not (ids_exc & playlist_ids)


def test_scan_sources_recorded(temp_lib):
    library.scan_library(str(temp_lib), full=True, use_web=True)
    sources = library.library_stats()["meta_sources"]
    # At least one track was identified from tags and one from the web.
    assert sources.get("tags", 0) >= 1
    assert sources.get("web", 0) >= 1
