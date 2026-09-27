"""Artist spacing ("no artist twice within N tracks") + proportional programs.

Three rules now shape a queue, in this order:

* no more than ``max_consecutive_artist`` songs in a row by one band,
* no artist at all within ``artist_gap`` tracks of its last song,
* no song within ``repeat_window`` plays of its last play.

This file pins the second one plus the program-length scaling that keeps a small
genre audible instead of skipped.
"""

from __future__ import annotations

import pytest

from app import config, db, library, scheduler


@pytest.fixture
def wide_library(tmp_path, monkeypatch):
    """One genre, 8 artists x 3 tracks — enough bands to space properly."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "vdj.sqlite3")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("VDJ_DATA_DIR", str(tmp_path))
    db.init_db()
    conn = db.connect()
    rows = []
    for artist_index in range(8):
        artist = f"Band {artist_index}"
        for track_index in range(3):
            rows.append((f"/m/rock/{artist} - {track_index}.mp3", artist,
                         f"Song {track_index}", "Rock", "1990"))
    for path, artist, title, genre, year in rows:
        conn.execute(
            "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
            "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
            (path, title, artist, "Album", genre, year, 210.0, 1, 1),
        )
    conn.commit()
    yield
    db.close()


def _use(monkeypatch, **playback):
    base = {"shuffle": True, "genres": [], "artists": [], "search": "",
            "program": {"enabled": True, "size": 4, "strategy": "genre"}}
    base.update(playback)
    monkeypatch.setattr(config, "_CACHE", {
        "music_dir": "/m", "playback": base,
        "dj": {"enabled": True, "talk_min": 2, "talk_max": 4},
        "llm": {"enabled": True}, "ai": {"free_text_genre": True},
        "websearch": {"enabled": True},
    })


def _tracks(prefix: str, count: int, artist: str, genre: str = "Rock"):
    return [{"id": i, "path": f"/m/{artist} - {i}.mp3", "title": f"{prefix}{i}",
             "artist": artist, "genre": genre} for i in range(count)]


# --- the rule itself ---------------------------------------------------------

def test_no_artist_returns_within_the_gap():
    tracks = []
    for artist in ("A", "B", "C", "D"):
        tracks += _tracks(artist, 3, artist)
    report: dict = {}
    ordered = library.interleave_artists(
        tracks, max_consecutive=2, previous_artist=None, previous_run=0,
        limit=len(tracks), min_gap=4, last_positions={}, report=report)
    assert len(ordered) == len(tracks)
    assert report["gap_breaks"] == 0, "4 artists can honour a 4-track gap"
    last: dict[str, int] = {}
    for index, track in enumerate(ordered):
        key = library.artist_key(track)
        assert index - last.get(key, -99) >= 4, "artist came back too early"
        last[key] = index


def test_positions_from_history_and_queue_are_honoured():
    """An artist that played 2 songs ago must wait out the rest of the gap.

    Four artists, not two: with only two bands the no-long-runs cap leaves the
    recent artist as the sole legal choice and the gap MUST give way — that case
    is ``test_a_one_artist_pool_plays_on_and_reports_the_break``. Here there is
    enough material to defer, so deferring is what must happen.
    """
    tracks = _tracks("x", 6, "Recent")
    others = [f"Other {index}" for index in range(12)]
    for name in others:
        tracks += _tracks("y", 6, name)
    # "Recent" played 3 tracks before the batch; everyone else long ago. A gap of
    # 10 needs 10+ bands to fill it, which is why there are 12 others here.
    positions = {"Recent": -3}
    positions.update({name: -40 for name in others})
    report: dict = {}
    ordered = library.interleave_artists(
        tracks, max_consecutive=2, previous_artist="Recent", previous_run=1,
        limit=24, min_gap=10, last_positions=positions, report=report)
    keys = [library.artist_key(t) for t in ordered]
    # Deterministic: the artist is 3 tracks back, so its next song may only come
    # at index 7 or later — the first slot is off limits however the draws fall.
    assert keys[0] != "Recent", "an artist 3 tracks back took the very next slot"
    last: dict[str, int] = dict(positions)     # queue coordinates: 0 = next song
    for index, key in enumerate(keys):
        assert index - last.get(key, -99) >= 10, (
            f"{key} returned {index - last.get(key)} tracks later (gap 10)")
        last[key] = index
    assert report["gap_breaks"] == 0, "13 artists can fill a 10-track gap"


def test_a_one_artist_pool_plays_on_and_reports_the_break():
    """A one-band theme cannot be spaced — play it and admit it, never stall."""
    tracks = _tracks("solo", 6, "Solo")
    report: dict = {}
    ordered = library.interleave_artists(
        tracks, max_consecutive=6, previous_artist=None, previous_run=0,
        limit=6, min_gap=10, last_positions={}, report=report)
    assert len(ordered) == 6, "never stall and drop songs over the spacing rule"
    assert report["gap_breaks"] == 5, "every repeat after the first is a break"


def test_the_gap_is_off_when_set_to_zero():
    tracks = _tracks("a", 4, "A") + _tracks("b", 4, "B")
    report: dict = {}
    ordered = library.interleave_artists(
        tracks, max_consecutive=4, previous_artist=None, previous_run=0,
        limit=8, min_gap=0, last_positions={}, report=report)
    assert report["gap_breaks"] == 0
    assert len(ordered) == 8


# --- proportional programs ---------------------------------------------------

def test_program_length_scales_with_the_theme_size():
    # Biggest theme takes the ceiling, a quarter-size theme a quarter of it,
    # and anything below the floor is clamped to the floor.
    assert library.program_size_for(1000, 6, biggest=1000, min_size=2) == 6
    assert library.program_size_for(250, 6, biggest=1000, min_size=2) == 2
    assert library.program_size_for(500, 6, biggest=1000, min_size=2) == 3
    assert library.program_size_for(3, 6, biggest=1000, min_size=2) == 2


def test_a_small_genre_still_gets_a_program(wide_library, monkeypatch, tmp_path):
    """A 3-track genre plays a SHORT program instead of being skipped.

    Three tracks by three different artists: the theme is small, so its program
    scales down to the floor — but it still gets airtime.
    """
    conn = db.connect()
    for index in range(3):
        conn.execute(
            "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
            "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
            (f"/m/jazz/J {index}.mp3", f"Jazz {index}", f"Jazz Band {index}",
             "Album", "Jazz", "1995", 210.0, 1, 1))
    conn.commit()
    _use(monkeypatch, program={"enabled": True, "size": 6, "strategy": "genre"})
    sched = scheduler.Scheduler()
    items = sched._build_programs(6)
    labels = [it["program"]["label"] for it in items]
    assert "Jazz" in labels, "the small genre was skipped entirely"
    jazz_tracks = [it for it in items if it["program"]["label"] == "Jazz"]
    # 3 tracks against Rock's 24 -> scaled to the floor of 2, not 6.
    assert len(jazz_tracks) == 2, f"Jazz ran {len(jazz_tracks)} tracks, wanted 2"


def test_a_one_band_theme_is_skipped_rather_than_repeating_itself(wide_library, monkeypatch):
    """A genre with a single band cannot host a program without breaking the gap.

    Oleg's rule: "if there are not enough songs for this genre/band, just skip to
    another program". With Rock (8 bands) available, the one-band genre must be
    skipped rather than play the same band twice inside its own program.
    """
    conn = db.connect()
    for index in range(4):
        conn.execute(
            "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
            "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
            (f"/m/folk/F {index}.mp3", f"Folk {index}", "Solo Act", "Album",
             "Folk", "1995", 210.0, 1, 1))
    conn.commit()
    _use(monkeypatch, artist_gap=6,
         program={"enabled": True, "size": 4, "strategy": "genre"})
    sched = scheduler.Scheduler()
    items = sched._build_programs(6)
    assert items, "the rotation must still fill with the other themes"
    labels = [it["program"]["label"] for it in items]
    assert "Folk" not in labels, (
        "a one-band genre played a program against the artist gap: "
        f"{[it['track']['artist'] for it in items if it['program']['label'] == 'Folk']}")
    keys = [library.artist_key(it["track"]) for it in items]
    last: dict[str, int] = {}
    for index, key in enumerate(keys):
        assert index - last.get(key, -99) >= 6
        last[key] = index


def test_a_bigger_theme_runs_longer_than_a_smaller_one(wide_library, monkeypatch):
    conn = db.connect()
    for index in range(9):
        conn.execute(
            "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
            "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
            (f"/m/jazz/J {index}.mp3", f"Jazz {index}", "Tune", "Album",
             "Jazz", "1995", 210.0, 1, 1))
    conn.commit()
    _use(monkeypatch, program={"enabled": True, "size": 6, "strategy": "genre"})
    sel = library.program_selection("genre", 6, 20)
    sizes = {t["genre"]: t["program_size"] for t in sel["themes"]}
    assert sizes["Rock"] == 6, "the biggest theme must take the full ceiling"
    # 9 tracks against 24 -> ceil(6 * 9 / 24) = 3.
    assert sizes["Jazz"] == 3, "a 9-track theme against 24 must run short"
    assert sizes["Rock"] > sizes["Jazz"]


# --- the queue builder -------------------------------------------------------

def test_the_queue_respects_the_gap_across_programs_and_refills(wide_library, monkeypatch):
    _use(monkeypatch, artist_gap=6,
         program={"enabled": True, "size": 4, "strategy": "genre"})
    sched = scheduler.Scheduler()
    # One genre means one program per refill (the theme list is the rotation), so
    # build in several passes — the gap has to hold across all of them.
    for _ in range(10):
        sched.refill(20)
    items = sched.peek(200)
    assert len(items) >= 20, "the queue should be built"
    last: dict[str, int] = {}
    for index, item in enumerate(items):
        key = library.artist_key(item["track"])
        assert index - last.get(key, -99) >= 6, (
            f"artist {key} returned after {index - last.get(key)} tracks "
            f"(gap 6) at queue position {index}")
        last[key] = index
    assert sched.gap_breaks() == 0, "8 artists can satisfy a 6-track gap"


def test_the_gap_survives_a_cleared_queue_via_history(wide_library, monkeypatch):
    """Spacing must hold against what already PLAYED, not just what is queued."""
    _use(monkeypatch, artist_gap=6,
         program={"enabled": True, "size": 4, "strategy": "genre"})
    sched = scheduler.Scheduler()
    sched.refill(20)
    played = [item["track"] for item in sched.peek(20)]
    # Pretend the first 12 of them played, then the queue is rebuilt.
    for track in played[:12]:
        db.record_play(track["id"])
    sched.clear()
    sched.refill(20)
    items = sched.peek(40)
    history_keys = [library.artist_key(t) for t in played[11::-1]]  # newest first
    last = {key: -(index + 1) for index, key in enumerate(history_keys)}
    for index, item in enumerate(items):
        key = library.artist_key(item["track"])
        assert index - last.get(key, -99) >= 6, (
            f"{key} came back {index - last.get(key)} tracks after its last play")
        last[key] = index
