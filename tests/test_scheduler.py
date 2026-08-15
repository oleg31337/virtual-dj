"""Scheduler: queue mechanics and DJ-break cadence."""

from __future__ import annotations

from app import config, library
from app import language as lang_mod
from app.scheduler import Scheduler

LANGUAGES = lang_mod.LANGUAGES


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


def test_dj_cadence_fixed_when_min_equals_max():
    # When talk_min == talk_max the interval is deterministic (no randomness).
    config.save_config({"dj": {"enabled": True, "talk_min": 3, "talk_max": 3}})
    sched = Scheduler()
    # Simulate filling the queue: each wrap stamps a dj_requested decision.
    items = [sched._wrap({"id": i}) for i in range(9)]
    due = [bool(it["dj_requested"]) for it in items]
    # First talk lands on index 3, then every 3 tracks -> indices 3 and 6.
    assert due == [False, False, False, True, False, False, True, False, False]


def test_dj_cadence_zero_means_never():
    config.save_config({"dj": {"enabled": True, "talk_min": 0, "talk_max": 0}})
    sched = Scheduler()
    items = [sched._wrap({"id": i}) for i in range(6)]
    assert not any(it["dj_requested"] for it in items)


def test_dj_disabled_overrides_cadence():
    config.save_config({"dj": {"enabled": False, "talk_min": 1, "talk_max": 1}})
    sched = Scheduler()
    items = [sched._wrap({"id": i}) for i in range(4)]
    assert not any(it["dj_requested"] for it in items)


def test_dj_random_interval_stays_in_range():
    # Over many rolls the stamped gaps should all fall inside [talk_min, talk_max].
    config.save_config({"dj": {"enabled": True, "talk_min": 2, "talk_max": 5}})
    sched = Scheduler()
    items = [sched._wrap({"id": i}) for i in range(600)]
    talks = [i for i, it in enumerate(items) if it["dj_requested"]]
    gaps = [talks[i + 1] - talks[i] for i in range(len(talks) - 1)]
    assert gaps, "expected some talks to occur"
    assert all(2 <= g <= 5 for g in gaps)


def test_prefetch_and_consumer_agree_on_dj_decision():
    # Both paths read the same stamped dj_requested flag, so they can never
    # disagree (no double-talk, no missed gap).
    config.save_config({"dj": {"enabled": True, "talk_min": 1, "talk_max": 2}})
    sched = Scheduler()
    items = [sched._wrap({"id": i}) for i in range(50)]
    for it in items:
        assert sched._dj_due(it, 0) == bool(it["dj_requested"])


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


def test_language_strategy_groups_into_language_runs(monkeypatch):
    # The 4th program strategy groups the queue into per-language runs
    # (English/French/Spanish/German/Russian). Verify the scheduler's wiring
    # (kind + label) without depending on the (empty in tests) library DB.
    config.save_config({"playback": {
        "genres": [], "artists": [], "shuffle": True,
        "program": {"enabled": True, "size": 5, "strategy": "language"}}})

    languages = ["russian", "french", "spanish", "german", "english"]
    monkeypatch.setattr(
        library, "program_themes",
        lambda strategy: [{"language": l, "n": 20} for l in languages]
        if strategy == "language" else [],
    )
    # Each language query returns two distinct fake tracks so the block is kept.
    def fake_query(languages=None, search="", limit=200, random_order=False, **kw):
        if not languages:
            return []
        return [
            {"id": 1, "path": "/x/1.mp3", "title": f"t1-{languages[0]}",
             "artist": "A", "album": "Al", "genre": "G", "year": "2000",
             "duration": 10.0},
            {"id": 2, "path": "/x/2.mp3", "title": f"t2-{languages[0]}",
             "artist": "A", "album": "Al", "genre": "G", "year": "2000",
             "duration": 10.0},
        ]

    monkeypatch.setattr(library, "query_tracks", fake_query)

    sched = Scheduler()
    items = sched._build_programs(3)
    assert items, "expected at least one program block"
    kinds = {it["program"]["kind"] for it in items if it.get("program")}
    assert kinds == {"language"}
    labels = [it["program"]["label"] for it in items if it.get("program")]
    assert all(label in LANGUAGES for label in labels)
