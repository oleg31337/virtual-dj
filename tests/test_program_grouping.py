"""Tests for themed program grouping in the scheduler.

These build a tiny in-memory library (via the real db module pointed at a temp
file) so the program builder runs against genuine SQL, not mocks.
"""

from __future__ import annotations

import pytest

from app import config, db, library, scheduler


@pytest.fixture
def tiny_library(tmp_path, monkeypatch):
    # Point the whole app at a temp DB and config dir.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vdj.sqlite3")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("VDJ_DATA_DIR", str(tmp_path))
    db.init_db()
    # Two genres, three tracks each, so each qualifies as a program of size>=2.
    rows = [
        ("/m/rock/A - One.mp3", "Rock A", "One", "Rock", "1990"),
        ("/m/rock/A - Two.mp3", "Rock A", "Two", "Rock", "1992"),
        ("/m/rock/A - Three.mp3", "Rock A", "Three", "Rock", "1991"),
        ("/m/pop/B - X.mp3", "Pop B", "X", "Pop", "2005"),
        ("/m/pop/B - Y.mp3", "Pop B", "Y", "Pop", "2006"),
        ("/m/pop/B - Z.mp3", "Pop B", "Z", "Pop", "2007"),
    ]
    conn = db.connect()
    for path, artist, title, genre, year in rows:
        conn.execute(
            "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
            "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
            (path, title, artist, "Album", genre, year, 210.0, 1, 1),
        )
    conn.commit()
    yield
    db.close()


def _set_program(monkeypatch, **overrides):
    prog = {"enabled": True, "size": 3, "strategy": "genre"}
    prog.update(overrides)
    monkeypatch.setattr(config, "_CACHE", {
        "music_dir": "/m", "playback": {"shuffle": True, "genres": [],
                                         "artists": [], "search": "",
                                         "program": prog},
        "dj": {"enabled": True, "every_n_tracks": 3},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })


def test_programs_group_by_genre(tiny_library, monkeypatch):
    _set_program(monkeypatch)
    sched = scheduler.Scheduler()
    items = sched._build_programs(2)
    # 2 programs x 3 tracks = 6 items.
    assert len(items) == 6
    # The first item of the second program carries a DJ-requested flag + theme.
    dj_items = [it for it in items if it.get("dj_requested")]
    assert len(dj_items) == 1
    theme = dj_items[0]["program"]
    assert theme["kind"] == "genre"
    # Contiguous blocks: all Rock first or all Pop first within a run.
    genres_in_order = [it["track"]["genre"] for it in items]
    assert genres_in_order.count("Rock") == 3
    assert genres_in_order.count("Pop") == 3
    # The run is not fully interleaved 1-by-1 (programs are grouped).
    assert "Rock" in "".join(genres_in_order) and "Pop" in "".join(genres_in_order)


def test_programs_respect_strategy_decade(tiny_library, monkeypatch):
    _set_program(monkeypatch, strategy="decade", size=2)
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    # 1990s has 3 tracks (>=size 2) and 2000s has 3 (>=size 2).
    assert len(items) >= 4
    for it in items:
        assert it["program"]["kind"] == "decade"
        assert it["program"]["label"] in ("1990s", "2000s")


def test_programs_disabled_falls_back(tiny_library, monkeypatch):
    _set_program(monkeypatch, enabled=False)
    sched = scheduler.Scheduler()
    # refill with programs disabled should still return a flat shuffle of all 6.
    n = sched.refill(10)
    assert n == 6
    # No program boundaries injected.
    assert all(it.get("program") is None for it in sched._queue)


def test_programs_too_small_library_returns_empty(tiny_library, monkeypatch):
    _set_program(monkeypatch, size=99)  # no theme has 99 tracks
    sched = scheduler.Scheduler()
    assert sched._build_programs(2) == []


def test_artist_strategy_not_killed_by_genre_filter(tiny_library, monkeypatch):
    # Regression: a global genre filter used to pre-filter the artist theme
    # list (which has no `genre` key) and silently fall back to a flat
    # shuffle. Now the filter is AND-ed into each theme's track query, so an
    # "Artist" program still builds (using only Rock tracks of that artist).
    prog = {"enabled": True, "size": 2, "strategy": "artist"}
    monkeypatch.setattr(config, "_CACHE", {
        "music_dir": "/m", "playback": {"shuffle": True,
                                         "genres": ["Rock"], "artists": [],
                                         "search": "", "program": prog},
        "dj": {"enabled": True, "every_n_tracks": 3},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    # Rock A (3 tracks, all Rock) qualifies; Pop B is excluded by the genre
    # filter, so we should still get a real artist-themed program, not empty.
    assert items, "artist strategy with a genre filter must still build programs"
    for it in items:
        assert it["program"]["kind"] == "artist"
        # Every queued track is Rock (genre filter respected).
        assert it["track"]["genre"] == "Rock"


def test_decade_strategy_with_genre_filter(tiny_library, monkeypatch):
    prog = {"enabled": True, "size": 2, "strategy": "decade"}
    monkeypatch.setattr(config, "_CACHE", {
        "music_dir": "/m", "playback": {"shuffle": True,
                                         "genres": ["Rock"], "artists": [],
                                         "search": "", "program": prog},
        "dj": {"enabled": True, "every_n_tracks": 3},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items, "decade strategy with a genre filter must still build programs"
    for it in items:
        assert it["track"]["genre"] == "Rock"
        assert it["program"]["kind"] == "decade"
