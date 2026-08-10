"""Integration test: scanning a small temp library applies the three-stage
metadata resolution, keeps non-Latin tracks (Cyrillic is now playable), and
excludes only corrupt / unidentifiable ones.

The web-confirmation and local-AI stages are stubbed so the test never
touches the network or the real model.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_mp3
from app import db, library, websearch, ai_meta


def _stub_confirm(artist, title, use_cache=True):
    if artist and title and title != "Garbage":
        return {"confirmed": True, "genre": "Rock", "confidence": 0.9,
                "sources": ["stub"], "artist": artist, "title": title}
    return {"confirmed": False, "genre": None, "confidence": 0.0, "sources": []}


def _stub_infer_genre(artist, title, album=None, path=None, use_cache=True):
    # The AI is the guaranteed last resort for genre; in tests it just tags
    # anything the web left empty as "Rock" so we can assert the plumbing.
    return "Rock"


@pytest.fixture
def temp_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(websearch, "confirm_track", _stub_confirm)
    monkeypatch.setattr(ai_meta, "infer_genre", _stub_infer_genre)
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

    # Non-Latin (Cyrillic) artist -> now KEPT and playable, web-confirmed.
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
    db.connect().executescript("DELETE FROM tracks; DELETE FROM web_lookups;")
    return music


def test_scan_classifies_and_excludes(temp_lib, monkeypatch):
    snap = library.scan_library(str(temp_lib), full=True, use_web=True)
    assert snap["error"] is None
    assert snap["scanned"] == 6

    stats = library.library_stats()
    # playable: tagged(1) + path-guess confirmed(1) + Cyrillic kept(1)
    #           + folder layout(1) = 4
    assert stats["playable"] == 4, stats
    assert stats["excluded"] == 2, stats
    # Cyrillic is no longer excluded; only no_title + unconfirmed remain.
    assert "non_latin" not in stats["unknown_reasons"], stats
    assert stats["unknown_reasons"].get("no_title") == 1
    assert stats["unknown_reasons"].get("unconfirmed") == 1

    # The web-confirmed path-guess got its genre from the stub.
    rows = library.query_tracks(genres=["Rock"])
    titles = {r["title"] for r in rows}
    assert "Guess Song" in titles
    # The Cyrillic track is also playable and genre-tagged.
    cyr = library.query_tracks(search="Подождем")
    assert cyr, "Cyrillic track should be playable"
    assert cyr[0]["artist"] == "Игорек"

    # Excluded tracks are reachable only via excluded_tracks().
    exc = library.excluded_tracks()
    assert len(exc) == 2
    reasons = {e["exclude_reason"] for e in exc}
    assert reasons == {"no_title", "unconfirmed"}

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
