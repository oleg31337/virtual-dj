"""Scheduler: queue mechanics and DJ-break cadence."""

from __future__ import annotations

from app import config, library
from app.scheduler import Scheduler


def _fresh(music_dir):
    library.scan_library(str(music_dir))
    return Scheduler()


def test_refill_pulls_from_library(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    assert sched.refill(10) > 0
    # All three are playable: the two tagged files, plus the untagged
    # "Band Three - Gamma.mp3" whose clean filename guess is kept (web
    # confirmation no longer excludes a usable guess).
    assert len(sched.peek()) == 3


def test_pop_next_returns_tracks(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(10)
    track, _, _ = sched.pop_next()
    assert track["path"]


def test_queue_auto_refills_when_drained(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(3)
    for _ in range(10):
        assert sched.pop_next() is not None  # never runs dry


def test_enqueue_specific_tracks(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    ids = [t["id"] for t in library.query_tracks()]
    assert sched.enqueue_track_ids(ids[:2]) == 2
    assert [i["track"]["id"] for i in sched.peek()] == ids[:2]


def test_enqueue_next_jumps_the_line(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    ids = [t["id"] for t in library.query_tracks()]
    sched.enqueue_track_ids([ids[0]])
    sched.enqueue_track_ids([ids[1]], position="next")
    assert sched.peek()[0]["track"]["id"] == ids[1]


def test_enqueue_ignores_unknown_ids(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    assert sched.enqueue_track_ids([999999]) == 0


def test_remove_and_clear(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(10)
    uid = sched.peek()[0]["uid"]
    assert sched.remove(uid) is True
    assert sched.remove(uid) is False
    sched.clear()
    assert sched.peek() == []


def test_move_reorders(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(10)
    items = sched.peek()
    last_uid = items[-1]["uid"]
    assert sched.move(last_uid, 0) is True
    assert sched.peek()[0]["uid"] == last_uid


def test_replace_swaps_the_whole_queue(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(10)
    ids = [t["id"] for t in library.query_tracks()][:1]
    assert sched.replace(ids) == 1
    assert len(sched.peek()) == 1


def test_dj_cadence_every_track():
    config.save_config({"dj": {"enabled": True, "every_n_tracks": 1}})
    sched = Scheduler()
    item = sched._wrap({"id": 1})
    assert all(sched._dj_due(item, i) for i in range(4))


def test_dj_cadence_every_third():
    config.save_config({"dj": {"enabled": True, "every_n_tracks": 3}})
    sched = Scheduler()
    item = sched._wrap({"id": 1})
    due = [sched._dj_due(item, i) for i in range(6)]
    assert due == [True, False, False, True, False, False]


def test_dj_cadence_zero_means_never():
    config.save_config({"dj": {"enabled": True, "every_n_tracks": 0}})
    sched = Scheduler()
    assert not sched._dj_due(sched._wrap({"id": 1}), 0)


def test_dj_disabled_overrides_cadence():
    config.save_config({"dj": {"enabled": False, "every_n_tracks": 1}})
    sched = Scheduler()
    assert not sched._dj_due(sched._wrap({"id": 1}), 0)


def test_genre_filter_applies_to_refill(music_dir, has_ffmpeg):
    library.scan_library(str(music_dir))
    config.save_config({"playback": {"genres": ["Rock"], "shuffle": False}})
    sched = Scheduler()
    sched.refill(10)
    assert [i["track"]["title"] for i in sched.peek()] == ["Alpha"]


def test_impossible_filter_falls_back_to_whole_library(music_dir, has_ffmpeg):
    """The stream must never go silent because a filter matched nothing."""
    library.scan_library(str(music_dir))
    config.save_config({"playback": {"genres": ["NoSuchGenreAnywhere"]}})
    sched = Scheduler()
    assert sched.refill(10) > 0


def test_status_reports_counts(music_dir, has_ffmpeg):
    sched = _fresh(music_dir)
    sched.refill(10)
    status = sched.status()
    assert status["queue_length"] == 3
    assert status["tracks_played"] == 0
    sched.pop_next()
    assert sched.status()["tracks_played"] == 1
