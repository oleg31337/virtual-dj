"""Library scanner tests: real mp3 files, real tags, real SQLite."""

from __future__ import annotations

from pathlib import Path

from app import config, db, library
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


def test_untagged_file_falls_back_to_filename(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    rows = library.query_tracks(search="Gamma")
    assert len(rows) == 1
    assert rows[0]["artist"] == "Band Three"
    assert rows[0]["title"] == "Gamma"


def test_guess_from_filename_variants():
    assert library.guess_from_filename(Path("Artist - Title.mp3")) == ("Artist", "Title")
    assert library.guess_from_filename(Path("01 - A - B.mp3")) == ("A", "B")
    assert library.guess_from_filename(Path("03 JustTitle.mp3")) == (None, "JustTitle")


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


def test_deleted_file_is_flagged_missing_not_dropped(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    (music_dir / "a.mp3").unlink()
    result = library.scan_library(str(music_dir))
    assert result["removed"] == 1
    # Row survives (history keeps resolving) but is excluded from queries.
    assert library.library_stats()["total"] == 3
    assert library.library_stats()["missing"] == 1
    assert not any(t["title"] == "Alpha" for t in library.query_tracks())


def test_genre_filter(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    rock = library.query_tracks(genres=["Rock"])
    assert [t["title"] for t in rock] == ["Alpha"]


def test_unknown_genre_bucket(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    genres = {g["genre"] for g in library.list_genres()}
    assert "Unknown" in genres
    unknown = library.query_tracks(genres=["Unknown"])
    assert [t["title"] for t in unknown] == ["Gamma"]


def test_missing_directory_records_error():
    result = library.scan_library("/nonexistent/path/xyz")
    assert result["error"] is not None
    assert result["running"] is False


def test_corrupt_file_does_not_break_scan(music_dir, has_ffmpeg):
    (music_dir / "broken.mp3").write_bytes(b"this is definitely not audio")
    result = library.scan_library(str(music_dir))
    assert result["error"] is None
    assert library.library_stats()["total"] == 4


def test_query_pagination_and_limit(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    page1 = library.query_tracks(limit=2, offset=0)
    page2 = library.query_tracks(limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 1
    assert {t["id"] for t in page1}.isdisjoint({t["id"] for t in page2})


def test_truncated_paren_tag_is_cleaned(music_dir, has_ffmpeg):
    """'Success (Thievery Corporation' (cut-off tag) must not keep the '('."""
    make_mp3(music_dir / "t.mp3", tags={"title": "Success (Thievery Corporation",
                                        "artist": "Thievery Corporation"})
    library.scan_library(str(music_dir))
    track = library.query_tracks(search="Success")[0]
    assert track["title"] == "Success"
