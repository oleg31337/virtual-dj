"""The no-repeat window: a song must not come back within N plays.

"I don't want same song to repeat too often, let's say don't repeat for at
least 50 songs."

The queue builder keeps the last ``playback.repeat_window`` plays (plus
everything already queued) out of the playlist, in every mode. These tests run
against real SQL on a temp DB, and drive the real ``record_play``/``refill``
path the broadcaster uses.
"""

from __future__ import annotations

import pytest

from app import config, db, library, scheduler


@pytest.fixture
def library_of(tmp_path, monkeypatch):
    """Build a temp library of ``n`` tracks, two artists per genre."""
    def build(n=12):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "DB_PATH", tmp_path / "vdj.sqlite3")
        monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
        monkeypatch.setenv("VDJ_DATA_DIR", str(tmp_path))
        db.init_db()
        conn = db.connect()
        for i in range(n):
            conn.execute(
                "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
                "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,0,0,'tags')",
                (f"/m/g{i % 2}/Artist {i % 4} - Song {i}.mp3", f"Song {i}",
                 f"Artist {i % 4}", "Album", f"Genre {i % 2}", "1999", 200.0, 1, 1))
        conn.commit()
        return n
    yield build
    db.close()


def set_playback(monkeypatch, **over):
    playback = {"shuffle": True, "genres": [], "artists": [], "search": "",
                "repeat_window": 50,
                "program": {"enabled": False, "size": 6, "strategy": "genre",
                            "limit": 20, "max_consecutive_artist": 2,
                            "disabled": {"genre": [], "artist": [], "decade": []}}}
    playback.update(over)
    monkeypatch.setattr(config, "_CACHE", {
        "music_dir": "/m", "playback": playback,
        "dj": {"enabled": True, "talk_min": 2, "talk_max": 4},
        "llm": {"enabled": True}, "websearch": {"enabled": True},
    })
    return playback


def play(ids):
    for i in ids:
        db.record_play(i)


def ids_of(items_or_tracks):
    return [t["id"] if "id" in t else t["track"]["id"] for t in items_or_tracks]


# --- the history query -------------------------------------------------------

def test_recent_track_ids_returns_the_last_plays_newest_first(library_of):
    library_of(12)
    play([3, 5, 7, 9])
    assert db.recent_track_ids(2) == [9, 7]
    assert set(db.recent_track_ids(4)) == {3, 5, 7, 9}
    assert db.recent_track_ids(0) == []


def test_recent_track_ids_keeps_repeats(library_of):
    library_of(12)
    play([4, 4, 4])
    # The window counts PLAYS, not distinct songs: three plays of track 4 fill
    # it, so nothing older can sneak back in.
    assert db.recent_track_ids(3) == [4, 4, 4]


def test_recent_track_ids_ignores_history_without_a_track(library_of):
    library_of(12)
    db.record_play(None)
    play([2])
    assert db.recent_track_ids(10) == [2]


# --- the rule itself ---------------------------------------------------------

def test_played_songs_stay_out_of_the_next_refill(library_of, monkeypatch):
    library_of(12)
    set_playback(monkeypatch, repeat_window=5)
    play([1, 2, 3, 4, 5])
    sched = scheduler.Scheduler()
    sched.refill(7)
    assert len(sched._queue) == 7, "only fresh tracks should fill the queue"
    assert not ({1, 2, 3, 4, 5} & set(ids_of(sched._queue)))


def test_song_older_than_the_window_may_play_again(library_of, monkeypatch):
    library_of(12)
    set_playback(monkeypatch, repeat_window=3)
    play([1, 2, 3, 4])          # 1 is the 4th-last -> outside a 3-song window
    sched = scheduler.Scheduler()
    sched.refill(8)
    queues = set(ids_of(sched._queue))
    assert not ({2, 3, 4} & queues), "the last 3 plays must not return"
    # 1 is allowed back; not guaranteed in one random draw of 8, so check the
    # query directly instead of the luck of the shuffle.
    pool = library.query_tracks(limit=20, exclude_ids=set(db.recent_track_ids(3)))
    assert 1 in ids_of(pool)


def test_no_repeats_within_the_queue_or_between_refills(library_of, monkeypatch):
    library_of(12)
    set_playback(monkeypatch, repeat_window=50, program={"enabled": True, "size": 4,
                 "strategy": "genre", "limit": 20, "max_consecutive_artist": 2,
                 "disabled": {"genre": [], "artist": [], "decade": []}})
    sched = scheduler.Scheduler()
    sched.refill(4)
    sched.refill(4)
    queued = ids_of(sched._queue)
    assert len(queued) == len(set(queued)), "a song must not be queued twice"


def test_honouring_the_window_over_many_songs(library_of, monkeypatch):
    """Simulate the broadcaster: queue, play, queue again — no early repeats."""
    n = library_of(12)
    set_playback(monkeypatch, repeat_window=6)
    sched = scheduler.Scheduler()
    played: list[int] = []
    for _ in range(24):                      # 24 songs from a 12-track library
        if not sched._queue:
            sched.refill(3)
        item = sched.pop_next()
        if item is None:
            break
        track_id = item[0]["id"]          # pop_next -> (track, dj_break, program)
        played.append(track_id)
        db.record_play(track_id)             # exactly what the broadcaster does
    assert len(played) >= 12
    # No song may reappear within 6 plays of itself.
    for i, tid in enumerate(played):
        window = played[max(0, i - 6):i]
        assert tid not in window, f"{tid} repeated within the window at {i}"
    # And it is not degenerate: with a 12-track library and a 6-song window we
    # still cycle through everything.
    assert len(set(played)) == n


def test_window_of_zero_disables_the_rule(library_of, monkeypatch):
    library_of(12)
    set_playback(monkeypatch, repeat_window=0)
    play([1, 2, 3, 4, 5])
    sched = scheduler.Scheduler()
    sched.refill(6)
    assert len(sched._queue) == 6
    assert library.recent_played_ids() == set()


def test_tiny_library_never_goes_silent(library_of, monkeypatch):
    """A 4-track library with a 50-song window must still play something.

    The window shrinks (never below REPEAT_WINDOW_FLOOR) rather than being
    dropped: yielding to "anything goes" let a song repeat immediately.
    """
    library_of(4)
    set_playback(monkeypatch, repeat_window=50)
    play([1, 2, 3, 4])                        # everything just played
    sched = scheduler.Scheduler()
    assert sched.refill(3) >= 1, "the station must never be left with nothing"
    # The floor is respected: the two most recent plays are still kept out even
    # though the configured window cannot be honoured.
    assert {3, 4}.isdisjoint(set(ids_of(sched._queue)))


def test_programs_mode_honours_the_window_too(library_of, monkeypatch):
    library_of(12)
    set_playback(monkeypatch, repeat_window=50, program={
        "enabled": True, "size": 2, "strategy": "genre", "limit": 20,
        "max_consecutive_artist": 2,
        "disabled": {"genre": [], "artist": [], "decade": []}})
    play(list(range(1, 9)))                   # 8 of 12 played
    sched = scheduler.Scheduler()
    sched.refill(4)
    assert not ({1, 2, 3, 4, 5, 6, 7, 8} & set(ids_of(sched._queue)))


def test_query_tracks_exclude_ids_is_a_real_filter(library_of):
    library_of(12)
    rows = library.query_tracks(limit=20, exclude_ids={1, 2, 3})
    assert not ({1, 2, 3} & set(ids_of(rows)))
    assert len(rows) == 9


def test_window_ladder_shrinks_but_never_below_the_floor(library_of, monkeypatch):
    library_of(6)
    set_playback(monkeypatch, repeat_window=50)
    assert scheduler.Scheduler._repeat_ladder() == [50, 25, 12, 6, 3, 2, 0]
    set_playback(monkeypatch, repeat_window=0)
    assert scheduler.Scheduler._repeat_ladder() == [0], "0 means the rule is off"
    set_playback(monkeypatch, repeat_window=3)
    assert scheduler.Scheduler._repeat_ladder() == [3, 2, 0]
