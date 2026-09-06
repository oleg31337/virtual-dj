"""Library scanner tests: real mp3 files, real tags, real SQLite."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app import db, library, textq
from tests.conftest import make_mp3


def test_scan_indexes_all_files(music_dir, has_ffmpeg):
    result = library.scan_library(str(music_dir))
    assert result["error"] is None
    assert result["added"] == 3
    assert library.library_stats()["total"] == 3


def test_tags_are_read_from_id3(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    rows = library.query_tracks(search="Alpha")
    assert len(rows) == 1
    track = rows[0]
    assert track["title"] == "Alpha"
    assert track["artist"] == "Band One"
    assert track["album"] == "First"
    assert track["genre"] == "Rock"
    assert track["year"] == "1999"
    assert track["duration"] > 0.5


def test_untagged_file_falls_back_to_filename(music_dir, has_ffmpeg, monkeypatch):
    # Path-guessed tracks are confirmed on the web before they become playable.
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": True,
                        "genre": "Rock", "confidence": 0.9, "sources": ["stub"],
                        "artist": a, "title": t})
    library.scan_library(str(music_dir))
    rows = library.query_tracks(search="Gamma")
    assert len(rows) == 1
    assert rows[0]["artist"] == "Band Three"
    assert rows[0]["title"] == "Gamma"


def test_clean_filename_guess_stays_playable_without_web(music_dir, has_ffmpeg, monkeypatch):
    # A clean "<Artist> - <Title>" filename guess is trustworthy on its own:
    # even when the web cannot confirm it, the track stays playable (the DJ can
    # announce it). Only genuinely corrupt/unusable names are excluded.
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": False,
                        "genre": None, "confidence": 0.0, "sources": []})
    library.scan_library(str(music_dir))
    assert library.library_stats()["excluded"] == 0
    titles = {t["title"] for t in library.query_tracks()}
    assert "Gamma" in titles
    # And the genre gets filled by the local AI fallback when web can't supply one.
    gamma = next(t for t in library.query_tracks() if t["title"] == "Gamma")
    assert gamma["artist"] == "Band Three"


def test_guess_from_filename_variants():
    assert library.guess_from_filename(Path("Artist - Title.mp3")) == ("Artist", "Title")
    assert library.guess_from_filename(Path("01 - A - B.mp3")) == ("A", "B")
    assert library.guess_from_filename(Path("03 JustTitle.mp3")) == (None, "JustTitle")


def test_leading_track_number_and_year_are_stripped(tmp_path):
    # Real-world noise: "040.URIAH HEEP - LADY IN BLAC.mp3" and a year-prefixed
    # album folder must not leak into the artist/title guess.
    p = tmp_path / "X" / "040.URIAH HEEP - LADY IN BLAC.mp3"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    g = textq.guess_from_path(p, tmp_path)
    assert g["artist"] == "URIAH HEEP"
    assert g["title"] == "LADY IN BLAC"

    p2 = tmp_path / "Y" / "1987 One Second" / "04 Le Secret Farida.mp3"
    p2.parent.mkdir(parents=True)
    p2.write_bytes(b"x")
    g2 = textq.guess_from_path(p2, tmp_path)
    assert g2["artist"] == "Y"
    assert g2["title"] == "Le Secret Farida"


def test_rescan_is_incremental(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    second = library.scan_library(str(music_dir))
    assert second["added"] == 0
    assert second["updated"] == 0
    assert second["total_seen"] == 3


def test_new_file_is_picked_up_on_rescan(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    make_mp3(music_dir / "c.mp3", tags={"title": "Delta", "artist": "Band Four"})
    result = library.scan_library(str(music_dir))
    assert result["added"] == 1
    assert library.library_stats()["total"] == 4


def test_deleted_file_is_removed_from_library_on_rescan(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    deleted = music_dir / "a.mp3"
    deleted.unlink()
    result = library.scan_library(str(music_dir))
    assert result["removed"] == 1
    # The vanished file's row is dropped so the library mirrors the folder on
    # disk: total drops to 2 and the deleted track no longer appears anywhere.
    assert library.library_stats()["total"] == 2
    assert not any(t["title"] == "Alpha" for t in library.query_tracks())
    assert db.connect().execute(
        "SELECT COUNT(*) AS n FROM tracks WHERE path = ?",
        (str(deleted),),
    ).fetchone()["n"] == 0


def test_genre_filter(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    rock = library.query_tracks(genres=["Rock"])
    assert [t["title"] for t in rock] == ["Alpha"]


def test_playable_tracks_appear_in_playlist_queries(music_dir, has_ffmpeg, monkeypatch):
    # Gamma is untagged but has a clean filename guess, which stays playable
    # (web confirmation no longer excludes a usable guess), so it appears in
    # normal playlist queries alongside the two tagged files.
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": False,
                        "genre": None, "confidence": 0.0, "sources": []})
    library.scan_library(str(music_dir))
    titles = {t["title"] for t in library.query_tracks()}
    assert "Gamma" in titles
    assert "Alpha" in titles and "Beta" in titles


def test_pagination_counts_playable_only(music_dir, has_ffmpeg, monkeypatch):
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": True,
                        "genre": "Rock", "confidence": 0.9, "sources": ["stub"],
                        "artist": a, "title": t})
    library.scan_library(str(music_dir))
    page1 = library.query_tracks(limit=2, offset=0)
    page2 = library.query_tracks(limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 1
    assert {t["id"] for t in page1}.isdisjoint({t["id"] for t in page2})


def test_missing_directory_records_error():
    result = library.scan_library("/nonexistent/path/xyz")
    assert result["error"] is not None
    assert result["running"] is False


def test_empty_directory_records_message(tmp_path):
    result = library.scan_library(str(tmp_path))
    assert result["total_seen"] == 0
    assert result["error"] is not None
    assert "No audio files found" in result["error"]
    assert result["running"] is False


def test_corrupt_file_does_not_break_scan(music_dir, has_ffmpeg):
    (music_dir / "broken.mp3").write_bytes(b"this is definitely not audio")
    result = library.scan_library(str(music_dir))
    assert result["error"] is None
    assert library.library_stats()["total"] == 4


def test_truncated_paren_tag_is_cleaned(music_dir, has_ffmpeg):
    """'Success (Thievery Corporation' (cut-off tag) must not keep the '('."""
    make_mp3(music_dir / "t.mp3", tags={"title": "Success (Thievery Corporation",
                                        "artist": "Thievery Corporation"})
    library.scan_library(str(music_dir))
    track = library.query_tracks(search="Success")[0]
    assert track["title"] == "Success"


# --- .wma / ASF tag reading ------------------------------------------------

def _make_wma(path: Path, tags: dict[str, str] | None = None) -> Path:
    """Render a real tiny .wma with ffmpeg and write ASF tags via mutagen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "wmav2", "-ar", "44100", "-ac", "1", str(path)],
        check=True, capture_output=True, timeout=60)
    if tags:
        from mutagen.asf import ASF
        asf = ASF(str(path))
        for key, value in tags.items():
            asf.tags[key] = [value]
        asf.save()
    return path


def test_wma_tags_are_read_from_asf(tmp_path, has_ffmpeg):
    # mutagen has no easy mode for ASF, so a .wma's tags must be read through
    # the raw ASF keys ("Title", "Author", "WM/...").
    p = _make_wma(
        tmp_path / "w" / "sample.wma",
        {"Title": "Alpha W", "Author": "Wma Band", "WM/AlbumTitle": "First",
         "WM/Genre": "Jazz", "WM/Year": "2001"},
    )
    m = library.read_metadata(p, root=tmp_path, allow_web=False)
    assert m["title"] == "Alpha W"
    assert m["artist"] == "Wma Band"
    assert m["album"] == "First"
    assert m["genre"] == "Jazz"
    assert m["year"] == "2001"
    assert m["meta_source"] == "tags"
    assert m["excluded"] == 0
    assert m["duration"] and m["duration"] > 0.5


def test_wma_file_is_scanned_and_playable(tmp_path, has_ffmpeg, monkeypatch):
    _make_wma(tmp_path / "wma" / "01 - Wma Song.wma",
              {"Title": "Wma Song", "Author": "Wma Band"})
    _make_wma(tmp_path / "wma" / "02 - Another.wma",
              {"Title": "Another", "Author": "Wma Band"})
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": True,
                        "genre": "Rock", "confidence": 0.9, "sources": ["stub"],
                        "artist": a, "title": t})
    result = library.scan_library(str(tmp_path / "wma"))
    assert result["error"] is None
    assert result["added"] == 2
    rows = library.query_tracks(search="Wma Song")
    assert len(rows) == 1
    assert rows[0]["artist"] == "Wma Band"
    assert rows[0]["title"] == "Wma Song"


def test_asf_tag_aliases_map_without_real_file(monkeypatch):
    """The ASF key aliases work against a raw ASF-style tags object."""
    class FakeAudio:
        tags = {"Title": ["stub title"], "Author": ["Stub Artist"],
                "WM/AlbumTitle": ["Stub Album"], "WM/Genre": ["Blues"],
                "WM/Year": ["1977"]}
        info = type("Info", (), {"length": 12.0})()

    monkeypatch.setattr(library, "MutagenFile",
                        lambda path, easy=True: FakeAudio())
    m = library.read_metadata(Path("/tmp/x/Stub File.wma"),
                              allow_web=False)
    assert m["title"] == "stub title"
    assert m["artist"] == "Stub Artist"
    assert m["album"] == "Stub Album"
    assert m["genre"] == "Blues"
    assert m["year"] == "1977"
    assert m["meta_source"] == "tags"


# --- unidentifiable files are registered, never excluded -------------------

def test_unidentifiable_file_registered_by_filename(monkeypatch, tmp_path):
    # No tags at all and a filename that yields no usable artist/title:
    # the file must stay playable as "Unknown - <file name>".
    monkeypatch.setattr(library, "MutagenFile", lambda path, easy=True: None)
    p = tmp_path / "Just A Folder" / "Track 03.wma"
    m = library.read_metadata(p, root=tmp_path, allow_web=False)
    assert m["excluded"] == 0
    assert m["exclude_reason"] is None
    assert m["artist"] == "Unknown"
    assert m["title"] == "Track 03"
    assert m["meta_source"] == "path"


def test_short_numeric_artist_name_is_kept(monkeypatch):
    # A real artist like "311", "U2" or "A1" fails the placeholder heuristics
    # (pure numbers / single letter) but must NEVER be overwritten with
    # "Unknown": only genuinely empty fields get the fallback.
    class FakeAudio:
        tags = {"title": ["Down"], "artist": ["311"]}
        info = type("Info", (), {"length": 200.0})()

    monkeypatch.setattr(library, "MutagenFile",
                        lambda path, easy=True: FakeAudio())
    monkeypatch.setattr(library.ai_meta, "recover_names",
                        lambda *a, **k: {"confident": False})
    m = library.read_metadata(Path("/mnt/mp3/311 - Down.mp3"),
                              root=Path("/mnt/mp3"), allow_web=False)
    assert m["excluded"] == 0
    assert m["artist"] == "311"
    assert m["title"] == "Down"


def test_junk_artist_label_becomes_unknown(monkeypatch, tmp_path):
    # CD rippers tag anonymous tracks with artist="artist" — that literal
    # label must not survive as an artist name; the file registers under
    # "Unknown" with its (junk) title kept so it stays playable.
    class FakeAudio:
        tags = {"title": ["Track 01"], "artist": ["artist"]}
        info = type("Info", (), {"length": 200.0})()

    monkeypatch.setattr(library, "MutagenFile",
                        lambda path, easy=True: FakeAudio())
    monkeypatch.setattr(library.ai_meta, "recover_names",
                        lambda *a, **k: {"confident": False})
    m = library.read_metadata(tmp_path / "01 - Track 01.mp3",
                              root=tmp_path, allow_web=False)
    assert m["excluded"] == 0
    assert m["artist"] == "Unknown"
    assert m["title"] == "Track 01"


def test_excluded_rows_self_heal_on_rescan(music_dir, has_ffmpeg, monkeypatch):
    # Simulate a row excluded by an older scan: an ordinary (incremental)
    # rescan must re-identify it instead of skipping the unchanged file.
    monkeypatch.setattr(library.websearch, "confirm_track",
                        lambda a, t, use_cache=True: {"confirmed": False,
                        "genre": None, "confidence": 0.0, "sources": []})
    library.scan_library(str(music_dir))
    conn = db.connect()
    row = conn.execute("SELECT id, path FROM tracks WHERE title='Gamma'").fetchone()
    assert row is not None
    conn.execute("UPDATE tracks SET excluded=1, exclude_reason='no_title', "
                 "title=NULL, artist=NULL WHERE id=?", (row["id"],))
    conn.commit()
    library.scan_library(str(music_dir))  # incremental, not full
    fixed = db.connect().execute(
        "SELECT title, artist, excluded FROM tracks WHERE title='Gamma'"
    ).fetchone()
    assert fixed, "previously-excluded row must be re-identified on rescan"
    assert fixed["excluded"] == 0
    assert fixed["artist"] == "Band Three"
