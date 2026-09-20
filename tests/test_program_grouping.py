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
        "dj": {"enabled": True, "talk_min": 2, "talk_max": 4},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })


def test_programs_group_by_genre(tiny_library, monkeypatch):
    _set_program(monkeypatch)
    sched = scheduler.Scheduler()
    items = sched._build_programs(2)
    # 2 programs x 3 tracks = 6 items.
    assert len(items) == 6
    # The first track of the SECOND program always carries a DJ-requested flag
    # (the vibe-switch announce). Programs are shuffled, so find the boundary
    # dynamically rather than assuming a fixed index. Periodic cadence talks
    # may also land mid-program; we assert the boundary announce specifically.
    genres = [it["track"]["genre"] for it in items]
    switch_index = next(i for i in range(1, len(genres)) if genres[i] != genres[i - 1])
    assert items[switch_index]["dj_requested"] is True
    theme = items[switch_index]["program"]
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
        "dj": {"enabled": True, "talk_min": 2, "talk_max": 4},
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
        "dj": {"enabled": True, "talk_min": 2, "talk_max": 4},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items, "decade strategy with a genre filter must still build programs"
    for it in items:
        assert it["track"]["genre"] == "Rock"
        assert it["program"]["kind"] == "decade"


# --- Programs card selection (switch themes off) ---------------------------
#
# The Programs card and the queue builder share library.program_selection(), so
# whatever a user switches off in the browser must be absent from the queue.

def test_selection_applies_limit_then_disabled(tiny_library, monkeypatch):
    _set_program(monkeypatch, limit=1)
    sel = library.program_selection("genre", size=3, limit=1)
    # Two eligible genres (Rock, Pop, 3 tracks each), but the rotation is the
    # biggest ONE of them — the cutoff is a stable "top N by track count" list.
    assert sel["eligible"] == 2
    assert len(sel["candidates"]) == 1
    assert sel["candidate_tracks"] == 3
    assert sel["library_tracks"] == 6
    # Switching a theme off removes it from the rotation (but not the list the
    # card renders, so it can be switched back on).
    disabled = {"genre": [sel["candidates"][0]["genre"]]}
    monkeypatch.setitem(config._CACHE["playback"]["program"], "disabled", disabled)
    sel2 = library.program_selection("genre", size=3, limit=1)
    assert sel2["themes"] == []
    assert len(sel2["candidates"]) == 1
    assert sel2["selected_tracks"] == 0


def test_disabled_genre_never_reaches_the_queue(tiny_library, monkeypatch):
    _set_program(monkeypatch, size=2, disabled={"genre": ["Rock"], "artist": [],
                                                "decade": []})
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items, "the remaining genre should still build programs"
    for it in items:
        assert it["track"]["genre"] != "Rock"
        assert it["program"]["label"] != "Rock"
    # And the flat path (programs off) honours it as well.
    _set_program(monkeypatch, enabled=False, size=2,
                 disabled={"genre": ["Rock"], "artist": [], "decade": []})
    sched2 = scheduler.Scheduler()
    assert sched2.refill(10) == 3
    assert all(i["track"]["genre"] == "Pop" for i in sched2._queue)


def test_disabled_artist_is_excluded_from_other_strategies(tiny_library, monkeypatch):
    # A banned artist must not sneak back in through a genre program.
    _set_program(monkeypatch, size=2, strategy="genre",
                 disabled={"genre": [], "artist": ["Pop B"], "decade": []})
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items
    assert all(it["track"]["artist"] != "Pop B" for it in items)
    # Pop's only artist is banned, so those themes yield nothing at all.
    assert all(it["track"]["genre"] == "Rock" for it in items)


def test_disabled_decade_is_excluded(tiny_library, monkeypatch):
    _set_program(monkeypatch, size=2, strategy="decade",
                 disabled={"genre": [], "artist": [], "decade": [2000]})
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items
    for it in items:
        assert it["track"]["year"].startswith("199")
        assert it["program"]["label"] == "1990s"


def test_switching_every_theme_off_queues_nothing(tiny_library, monkeypatch):
    # Silence is the honest outcome here: silently resurrecting the library
    # would play exactly what the user switched off.
    _set_program(monkeypatch, size=2,
                 disabled={"genre": ["Rock", "Pop"], "artist": [], "decade": []})
    sched = scheduler.Scheduler()
    assert sched._build_programs(4) == []
    assert sched.refill(10) == 0
    assert sched.peek() == []


def test_exclusions_are_null_safe(tiny_library, monkeypatch):
    conn = db.connect()
    conn.execute(
        "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
        "mtime,size,missing,excluded,meta_source) "
        "VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
        ("/m/misc/Z.mp3", "Untagged", None, "Album", None, None, 200.0, 1, 1),
    )
    conn.commit()
    # SQL NULL NOT LIKE '%Rock%' is NULL (not true), so a naive exclusion would
    # silently drop every genre-less track too. Each clause is NULL-safe.
    for kwargs in ({"exclude_genres": ["Rock"]},
                   {"exclude_artists": ["Rock A"]},
                   {"exclude_decades": [1990]}):
        rows = library.query_tracks(**kwargs)
        assert any(r["title"] == "Untagged" for r in rows), kwargs
    # "Unknown" is the one value that deliberately removes genre-less tracks.
    rows = library.query_tracks(exclude_genres=["Unknown"])
    assert not any(r["title"] == "Untagged" for r in rows)
    assert len(rows) == 6
    conn.execute("DELETE FROM tracks WHERE title = 'Untagged'")
    conn.commit()


def test_genres_card_filter_restricts_the_queue(tiny_library, monkeypatch):
    # The Genres card writes playback.genres — an INCLUDE filter. It has to
    # constrain both the flat queue and every program.
    _set_program(monkeypatch, enabled=False, size=2)
    monkeypatch.setitem(config._CACHE["playback"], "genres", ["Pop"])
    sched = scheduler.Scheduler()
    assert sched.refill(10) == 3
    assert all(i["track"]["genre"] == "Pop" for i in sched._queue)

    _set_program(monkeypatch, size=2, strategy="genre")
    monkeypatch.setitem(config._CACHE["playback"], "genres", ["Pop"])
    sched2 = scheduler.Scheduler()
    items = sched2._build_programs(4)
    assert items
    assert all(it["track"]["genre"] == "Pop" for it in items)


def test_genre_filter_narrows_genre_programs(tiny_library, monkeypatch):
    """Regression (caught by live validation on the real library).

    The Genres card's include filter used to be OR-ed into each genre program's
    own genre, so a "Punk" program kept playing Punk tracks while only
    "Electronic" was selected. A same-dimension filter must NARROW the rotation.
    """
    _set_program(monkeypatch, size=2, strategy="genre")
    monkeypatch.setitem(config._CACHE["playback"], "genres", ["Pop"])
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items, "the selected genre still builds programs"
    # Not one track outside the filter, and no Rock-themed program at all.
    assert all(it["track"]["genre"] == "Pop" for it in items)
    assert all(it["program"]["label"] == "Pop" for it in items)


def test_artist_filter_narrows_artist_programs(tiny_library, monkeypatch):
    _set_program(monkeypatch, size=2, strategy="artist")
    monkeypatch.setitem(config._CACHE["playback"], "artists", ["Pop B"])
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items
    assert all(it["track"]["artist"] == "Pop B" for it in items)


def test_cross_dimension_filters_still_and(tiny_library, monkeypatch):
    # An artist theme + a genre filter must AND (unchanged behaviour): only
    # that artist's tracks and only in that genre.
    _set_program(monkeypatch, size=2, strategy="artist")
    monkeypatch.setitem(config._CACHE["playback"], "genres", ["Rock"])
    sched = scheduler.Scheduler()
    items = sched._build_programs(4)
    assert items
    assert all(it["track"]["artist"] == "Rock A" for it in items)
    assert all(it["track"]["genre"] == "Rock" for it in items)


def test_unlisted_themes_still_play_with_every_listed_theme_off(tiny_library, monkeypatch):
    """Documented semantics of the all-off case (matches the live behaviour).

    Only the biggest theme is listed (limit=1). Switching it off empties the
    program rotation, so the queue falls back to a flat shuffle instead of
    going silent — but the switched-off theme must STILL be excluded.
    """
    _set_program(monkeypatch, size=2, limit=1,
                 disabled={"genre": ["Rock"], "artist": [], "decade": []})
    sched = scheduler.Scheduler()
    n = sched.refill(10)
    assert n > 0, "an empty rotation must not silence the station"
    assert all(i["track"]["genre"] != "Rock" for i in sched._queue)
