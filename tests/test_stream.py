"""Broadcast engine: real ffmpeg encoding, real MP3 bytes, real pacing."""

from __future__ import annotations

import subprocess
import threading
import time

from app import config
from app import stream as stream_mod
from app.stream import CLIENT_BUFFER_CHUNKS, Broadcaster, Listener


def test_listener_registry():
    b = Broadcaster()
    assert b.listener_count() == 0
    l1, l2 = b.add_listener(), b.add_listener()
    assert b.listener_count() == 2
    b.remove_listener(l1)
    assert b.listener_count() == 1
    b.remove_listener(l2)
    assert b.listener_count() == 0


def test_broadcast_reaches_every_listener():
    b = Broadcaster()
    a, c = b.add_listener(), b.add_listener()
    b._broadcast(b"hello")
    assert a.queue.get_nowait() == b"hello"
    assert c.queue.get_nowait() == b"hello"


def test_slow_listener_is_drained_not_blocking():
    """A stalled client must never freeze the transmitter for everyone else."""
    listener = Listener(1)
    for i in range(CLIENT_BUFFER_CHUNKS + 50):
        listener.put(b"x")  # must not raise or hang
    assert listener.dropped is True
    assert listener.queue.qsize() <= CLIENT_BUFFER_CHUNKS


def test_encode_args_reflect_config():
    config.save_config({"stream": {"bitrate_kbps": 192, "sample_rate": 48000}})
    args = Broadcaster()._encode_args("/tmp/x.mp3")
    assert "192k" in args
    assert "48000" in args
    assert args[-1] == "pipe:1"
    assert "-f" in args and "mp3" in args


def test_play_file_emits_valid_mp3(music_dir, has_ffmpeg):
    """The bytes we push to clients must actually decode as MP3 audio."""
    b = Broadcaster()
    listener = b.add_listener()
    track = {"path": str(music_dir / "a.mp3"), "title": "Alpha"}

    done = threading.Event()
    threading.Thread(
        target=lambda: (b._play_file(track["path"], "track", {"track": track}),
                        done.set()),
        daemon=True,
    ).start()

    chunks = []
    deadline = time.time() + 20
    while time.time() < deadline and not (done.is_set() and listener.queue.empty()):
        try:
            item = listener.queue.get(timeout=0.5)
        except Exception:
            continue
        if item is None:
            break
        chunks.append(item)
    done.wait(timeout=10)

    data = b"".join(chunks)
    assert len(data) > 2000, "no audio was broadcast"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=format_name:stream=codec_name,sample_rate,channels",
         "-of", "default=nw=1", "-"],
        input=data, capture_output=True, timeout=30,
    )
    out = probe.stdout.decode()
    assert "mp3" in out, f"stream did not decode as mp3: {probe.stderr.decode()[:300]}"
    assert "codec_name=mp3" in out


def test_play_file_missing_path_returns_false():
    assert Broadcaster()._play_file("/no/such/file.mp3", "track", {}) is False


def test_now_playing_state_updates(music_dir, has_ffmpeg):
    b = Broadcaster()
    b.add_listener()
    track = {"path": str(music_dir / "a.mp3"), "title": "Alpha", "artist": "Band One"}
    threading.Thread(
        target=b._play_file,
        args=(track["path"], "track", {"track": track, "duration": 1.0}),
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline and b.state()["track"] is None:
        time.sleep(0.1)
    state = b.state()
    assert state["track"]["title"] == "Alpha"
    assert state["kind"] == "track"
    assert state["listeners"] == 1
    b.stop()


def test_skip_interrupts_playback(music_dir, has_ffmpeg):
    from tests.conftest import make_mp3

    long_track = make_mp3(music_dir / "long.mp3", seconds=25)
    b = Broadcaster()
    b.add_listener()
    finished = threading.Event()

    def run():
        b._play_file(str(long_track), "track", {"track": {"path": str(long_track)}})
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    time.sleep(2.0)
    assert not finished.is_set(), "25s track ended too early — pacing is broken"
    b.skip()
    assert finished.wait(timeout=10), "skip did not interrupt playback"


def test_playback_is_paced_in_real_time(music_dir, has_ffmpeg):
    """A 25s track must not be blasted out in one burst."""
    from tests.conftest import make_mp3

    track = make_mp3(music_dir / "paced.mp3", seconds=25)
    b = Broadcaster()
    listener = b.add_listener()
    threading.Thread(
        target=b._play_file,
        args=(str(track), "track", {"track": {"path": str(track)}}),
        daemon=True,
    ).start()

    start = time.time()
    total = 0
    while time.time() - start < 3.0:
        try:
            item = listener.queue.get(timeout=0.5)
        except Exception:
            continue
        if item is None:
            break
        total += len(item)
    b.skip()

    bps = b._bytes_per_second()
    seconds_of_audio = total / bps
    # ~3s of wall clock should yield roughly 3-5s of audio (1s look-ahead),
    # definitely not the whole 25s file.
    assert seconds_of_audio < 12, f"stream ran away: {seconds_of_audio:.1f}s in 3s"
    assert seconds_of_audio > 0.5, "stream produced almost nothing"


def test_pause_and_resume_flags():
    b = Broadcaster()
    assert b.paused is False
    b.pause()
    assert b.paused is True
    b.resume()
    assert b.paused is False


def test_state_shape_is_stable():
    state = Broadcaster().state()
    for key in ("kind", "track", "dj_text", "elapsed", "duration",
                "paused", "listeners", "scheduler"):
        assert key in state


def test_dj_text_shows_for_announced_track_then_clears(music_dir, has_ffmpeg):
    """Per requirement: the 'DJ SAID' panel shows the phrase for the track it
    introduced, and clears when that song ends (the next track starts)."""
    from tests.conftest import make_mp3
    make_mp3(music_dir / "a.mp3", seconds=2)   # DJ break source (any audio)
    announced = make_mp3(music_dir / "b.mp3", seconds=2)
    next_one = make_mp3(music_dir / "c.mp3", seconds=2)
    dj_track = {"path": str(music_dir / "a.mp3"), "title": "Alpha",
                "artist": "Band One", "id": 101}
    announced = {"path": str(announced), "title": "Beta",
                  "artist": "Band Two", "id": 102}
    next_one = {"path": str(next_one), "title": "Gamma",
                "artist": "Band Three", "id": 103}

    b = Broadcaster()
    b.add_listener()

    def play(path, kind, meta):
        b._play_file(path, kind, meta)

    # 1) DJ break introducing track id 102.
    threading.Thread(
        target=play,
        args=(dj_track["path"], "dj",
              {"track": announced, "dj_text": "Next up, a run of Rock.",
               "duration": 1.0}),
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline and b.state().get("dj_text") is None:
        time.sleep(0.1)
    assert b.state()["dj_text"] == "Next up, a run of Rock."

    # 2) The announced track (id 102) plays -> phrase still shown.
    threading.Thread(
        target=play, args=(announced["path"], "track", {"track": announced}),
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline and b.state()["kind"] != "track":
        time.sleep(0.1)
    assert b.state()["dj_text"] == "Next up, a run of Rock."

    # 3) The next track (id 103) starts -> the song has ended, panel clears.
    threading.Thread(
        target=play, args=(next_one["path"], "track", {"track": next_one}),
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline and (b.state().get("track") or {}).get("id") != 103:
        time.sleep(0.1)
    assert b.state()["dj_text"] is None
    b.stop()


# --- a library that cannot be played must not spin ---------------------------
#
# Live incident: /mnt/mp3 was unmounted, so every queued file was gone. Each
# miss returned instantly, so the broadcast loop consumed ~34,000 queue items in
# 45 s, refilled forever (each refill stamping new DJ talks -> LLM calls at
# ~13/min) while the UI showed a silent IDLE. These tests pin the guards.

def test_missing_track_is_flagged_and_leaves_the_pool(music_dir, has_ffmpeg):
    from app import db, library
    library.scan_library(str(music_dir))
    track = library.query_tracks(limit=1)[0]
    b = Broadcaster()
    assert b._play_file("/gone/" + track["path"].split("/")[-1], "track",
                        {"track": track}) is False
    # The row is flagged, so query_tracks (which filters missing = 0) never
    # hands it out again — one attempt per dead file, not thousands.
    assert db.unplayable_count() >= 1
    assert track["id"] not in {t["id"] for t in library.query_tracks(limit=50)}


def test_dj_clip_failure_does_not_touch_the_library(music_dir, has_ffmpeg):
    from app import db, library
    library.scan_library(str(music_dir))
    track = library.query_tracks(limit=1)[0]
    before = db.unplayable_count()
    Broadcaster()._play_file("/gone/break.mp3", "dj", {"track": track})
    assert db.unplayable_count() == before   # a missing DJ clip is not a library row


def test_failure_breaker_backs_off_and_reports_the_error():
    b = Broadcaster()
    assert b.state()["error"] is None
    delays = [b._register_failure("/music/dead.mp3") for _ in range(12)]
    # Below the threshold there is no delay (a few bad files are just skipped)…
    assert delays[:stream_mod.FAILURE_BREAKER - 1] == [0.0] * (stream_mod.FAILURE_BREAKER - 1)
    # …then it grows and is capped, so a mass-missing library cannot spin the
    # CPU or re-trigger refills thousands of times per minute.
    past = delays[stream_mod.FAILURE_BREAKER - 1:]
    assert all(d > 0 for d in past)
    assert past == sorted(past), "the backoff must never shrink while failing"
    assert max(past) == 30.0, "the backoff must be capped at 30 s"
    err = b.state()["error"]
    assert err and "could not be played" in err and "mounted" in err
    # Playing something successfully clears it.
    b._register_success()
    assert b.state()["error"] is None
