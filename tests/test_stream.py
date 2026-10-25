"""Broadcast engine: real ffmpeg encoding, real MP3 bytes, real pacing."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

import pytest

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
    # The folder is there, this one file is not: the track really is gone.
    gone = str(music_dir / "deleted-since-the-last-scan.mp3")
    assert b._play_file(gone, "track", {"track": track}) is False
    # The row is flagged, so query_tracks (which filters missing = 0) never
    # hands it out again — one attempt per dead file, not thousands.
    assert db.unplayable_count() >= 1
    assert track["id"] not in {t["id"] for t in library.query_tracks(limit=50)}


def test_an_unmounted_library_never_flags_tracks_missing(music_dir, has_ffmpeg):
    """A share that is away must not drain the pool it flagged on the way out.

    Live: one unmounted night flagged 142 rows. ``missing = 1`` excludes a row
    until a rescan, so a few such nights take the playable pool to zero and the
    station goes silent even once the share is back — playback "stops on its
    own" days later. The file's own folder tells the two apart: /gone is not
    there at all, so nothing there can be judged missing.
    """
    from app import db, library
    library.scan_library(str(music_dir))
    track = library.query_tracks(limit=1)[0]
    before = db.unplayable_count()
    b = Broadcaster()
    assert b._play_file("/gone/" + track["path"].split("/")[-1], "track",
                        {"track": track}) is False
    assert db.unplayable_count() == before
    assert track["id"] in {t["id"] for t in library.query_tracks(limit=200)}


def test_a_library_that_comes_back_returns_its_flagged_tracks(music_dir, has_ffmpeg):
    """The first file that plays after a stall heals what the stall flagged."""
    from app import db, library
    library.scan_library(str(music_dir))
    track = library.query_tracks(limit=1)[0]
    db.mark_missing(track["id"])
    assert db.unplayable_count() >= 1
    b = Broadcaster()
    b._consecutive_failures = 5          # the run of failures this recovery ends
    b._register_success()
    flagged = lambda: {track_id for track_id, _ in db.missing_rows()}
    deadline = time.monotonic() + 15     # the heal runs off-thread
    while track["id"] in flagged() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert track["id"] not in flagged()
    assert track["id"] in {t["id"] for t in library.query_tracks(limit=200)}


def test_the_empty_queue_heal_runs_at_most_once_a_minute(monkeypatch):
    """A drained pool must heal, but the 5 s idle loop must not hammer the DB."""
    b = Broadcaster()
    calls: list[int] = []

    def fake_heal() -> None:
        calls.append(1)
        b._healed_at = time.monotonic()   # the real heal stamps the clock

    monkeypatch.setattr(b, "_heal_missing_flags", fake_heal)
    b._heal_missing_flags_if_flagged()
    b._heal_missing_flags_if_flagged()
    assert calls == [1]                 # throttled
    b._healed_at -= 61
    b._heal_missing_flags_if_flagged()
    assert calls == [1, 1]


# --- dead air: a track that goes silent must not hold the station ----------------


def test_the_silence_rule_counts_seconds_actually_on_the_air():
    config.save_config({"playback": {"silence_skip_s": 10, "silence_db": -50}})
    b = Broadcaster()
    assert b._silence_due(0.0) is False        # nothing reported as silent yet
    b._silence_start = 30.0
    assert b._silence_due(35.0) is False       # 5 s of it
    assert b._silence_due(40.0) is True        # 10 s on the air
    b._silence_start = None                    # the music came back
    assert b._silence_due(400.0) is False
    config.save_config({"playback": {"silence_skip_s": 0}})
    b._silence_start = 0.0
    assert b._silence_due(999.0) is False      # switched off


def test_the_encoder_reports_silence_only_when_the_feature_is_on():
    config.save_config({"playback": {"silence_skip_s": 10, "silence_db": -45}})
    args = Broadcaster()._encode_args("/tmp/x.mp3")
    chain = args[args.index("-af") + 1]
    assert chain.startswith("silencedetect=noise=-45dB:")
    # silencedetect reports at INFO; with the old error level it would say
    # nothing at all.
    assert args[args.index("-loglevel") + 1] == "info"
    config.save_config({"playback": {"silence_skip_s": 0}})
    args = Broadcaster()._encode_args("/tmp/x.mp3")
    assert "silencedetect" not in args[args.index("-af") + 1]
    assert args[args.index("-loglevel") + 1] == "error"


def _gap_file(path, tone_s: float, gap_s: float, tail_s: float) -> str:
    """An mp3 that is music, then a stretch of absolute silence, then music."""
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={tone_s}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={gap_s}",
        "-f", "lavfi", "-i", f"sine=frequency=660:duration={tail_s}",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "128k",
        "-write_xing", "0", "-id3v2_version", "0", str(path),
    ], check=True)
    return str(path)


def test_a_hidden_silence_is_skipped(tmp_path, has_ffmpeg):
    """The Offspring case: a track with a long silent stretch inside it.

    "Smash" ends "Come Out and Play" with a hidden gap, so the station sat in
    silence for minutes with a listener connected. The skip has to be a
    DELIBERATE one (the same path as the Skip button), not a playback failure —
    otherwise the failure breaker would treat silence as an unmounted library.
    """
    config.save_config({"playback": {"silence_skip_s": 2, "silence_db": -50}})
    path = _gap_file(tmp_path / "gap.mp3", tone_s=1, gap_s=8, tail_s=1)
    b = Broadcaster()
    assert b._play_file(path, "track", {"track": {"id": 1, "gain_db": 0.0}}) is True
    assert b._skip.is_set() is True            # cut, not played to the end
    assert b.state()["error"] is None          # and not counted as a failure


def test_a_short_gap_is_not_dead_air(tmp_path, has_ffmpeg):
    """A breath between two parts of a song is not silence worth skipping."""
    config.save_config({"playback": {"silence_skip_s": 2, "silence_db": -50}})
    path = _gap_file(tmp_path / "breath.mp3", tone_s=1, gap_s=0.4, tail_s=1)
    b = Broadcaster()
    assert b._play_file(path, "track", {"track": {"id": 2, "gain_db": 0.0}}) is True
    assert b._skip.is_set() is False           # played through


def test_a_silent_file_does_not_hold_the_station(tmp_path, has_ffmpeg):
    """A file that decodes to nothing (a broken rip) must not park the air."""
    config.save_config({"playback": {"silence_skip_s": 2, "silence_db": -50}})
    path = _gap_file(tmp_path / "silent.mp3", tone_s=0.1, gap_s=8, tail_s=0.1)
    b = Broadcaster()
    assert b._play_file(path, "track", {"track": {"id": 3, "gain_db": 0.0}}) is True
    assert b._skip.is_set() is True


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


# --- the station must never go quiet -------------------------------------------
#
# Icecast drops a source that sends nothing for <source-timeout> (30 s in the
# icecast.xml the app renders for the bundled server, 10 s in a standalone
# deploy/icecast.xml) and a dropped source takes the mount with it: an external
# player that tunes in while the mount is gone gets nothing at all and has to be
# stopped and started by hand. Every stall in the broadcast loop used to do this
# — the pause, the backoff after a run of unplayable files (measured against the
# real server: the pusher's socket was broken off ~30 s after the library was
# unmounted), the LLM writing a DJ break and piper speaking it, which is exactly
# what a resuming listener waits for. Broadcaster._keepalive sends silence
# whenever nothing else is playing, so there is always a stream on the mount.

def test_a_paused_station_keeps_sending_audio(has_ffmpeg):
    b = Broadcaster()
    listener = b.add_listener()
    b.pause()
    worker = threading.Thread(target=b._pause_hold, daemon=True)
    air = threading.Thread(target=b._keepalive, daemon=True)
    worker.start()
    air.start()
    try:
        data = listener.queue.get(timeout=15)
        assert data, "a paused station must still send bytes"
        assert data[0] == 0xFF and (data[1] & 0xE0) == 0xE0, (
            "the stream must stay whole MPEG frames")
    finally:
        b._stop.set()
        b.resume()
        worker.join(timeout=5)
        air.join(timeout=5)
    assert b.paused is False


def test_a_stalled_station_never_goes_quiet(has_ffmpeg):
    """Nothing playable at all — the state that dropped the mount for real.

    Before the keep-alive, this stream carried no bytes whatsoever while the
    loop backed off, so the mount vanished and a player connecting to it got
    nothing until it was restarted by hand.
    """
    b = Broadcaster()
    listener = b.add_listener()
    b._silence_frames()  # warm the cache: no ffmpeg spawn in the middle here
    b._last_audio_at = time.monotonic()
    b._next_silence_at = b._last_audio_at + stream_mod.KEEPALIVE_GAP_S
    air = threading.Thread(target=b._keepalive, daemon=True)
    air.start()
    arrivals: list[float] = []
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            try:
                listener.queue.get(timeout=2.0)
            except queue.Empty:
                pytest.fail("a stalled station went quiet for 2 s")
            arrivals.append(time.monotonic())
    finally:
        b._stop.set()
        air.join(timeout=5)
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    assert len(arrivals) >= 3, f"only {len(arrivals)} chunks of air in 4 s"
    assert max(gaps) < 2.0, f"a listener heard {max(gaps):.1f}s of nothing"


def test_the_pause_hold_itself_sends_nothing(has_ffmpeg):
    """One owner of the air: the hold waits, the keep-alive sends.

    Two senders would put two seconds of audio on the wire per second — a
    latency cushion on every listener that never comes back down.
    """
    b = Broadcaster()
    sent: list[bytes] = []
    b._broadcast = lambda data: sent.append(data)  # type: ignore[method-assign]
    b.pause()
    done = threading.Event()
    threading.Thread(target=lambda: (b._pause_hold(), done.set()),
                     daemon=True).start()
    time.sleep(0.5)
    assert not sent, "the hold must leave the air to the keep-alive"
    b.resume()
    assert done.wait(5.0)


def test_the_keepalive_adds_nothing_while_audio_is_flowing():
    """A track in flight must not have silence injected on top of it."""
    b = Broadcaster()
    listener = b.add_listener()
    b._silence_frames = lambda: b"\xff\xfb\x90\x00"  # type: ignore[method-assign]
    b._broadcast(b"\xff\xfb\x90\x00" * 20)  # audio just went out
    assert listener.queue.get(timeout=5)
    assert b._keepalive_tick(time.monotonic()) is False
    assert listener.queue.empty(), "silence on top of live audio"


def test_the_keepalive_fills_a_stall_in_real_time():
    """One second of silence per second: no drift, no doubling.

    A station stalled for hours must send exactly as much audio as time passes;
    anything less slowly drains every listener's cushion.
    """
    b = Broadcaster()
    b._silence_frames = lambda: b"\xff\xfb\x90\x00" * 8  # type: ignore[method-assign]
    b._last_audio_at = b._next_silence_at = 0.0
    times: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < 5.0:
        now = time.monotonic()
        if b._keepalive_tick(now):
            times.append(now)
        time.sleep(0.05)
    assert 4 <= len(times) <= 6, f"expected ~1 chunk/s, got {len(times)} in 5 s"
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert max(gaps) < 2.0, f"a gap of {max(gaps):.1f}s is air the source lost"


def test_the_keepalive_says_so_when_it_cannot_fill_the_air(caplog):
    b = Broadcaster()
    b._silence_frames = lambda: b""  # type: ignore[method-assign]
    b._last_audio_at = b._next_silence_at = 0.0
    assert b._keepalive_tick(time.monotonic()) is False
    assert b._keepalive_warned is True
    assert any("on air" in record.message for record in caplog.records)


def test_starting_the_station_starts_the_keepalive(has_ffmpeg):
    b = Broadcaster()
    b.start()
    try:
        assert b._keepalive_thread is not None and b._keepalive_thread.is_alive()
        assert b._keepalive_thread.name == "vdj-keepalive"
    finally:
        b.stop()
    assert b._keepalive_thread is None, "stop() must own the thread's lifetime"


def test_the_pause_silence_is_really_silent(has_ffmpeg):
    """Not just valid frames — frames that decode to (near) silence."""
    frames = Broadcaster()._silence_frames()
    assert frames, "silence must be generated"
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-f", "mp3", "-i", "pipe:0",
         "-af", "volumedetect", "-f", "null", "-"],
        input=frames, capture_output=True, timeout=60,
    )
    text = probe.stderr.decode("utf-8", "replace")
    assert "mean_volume" in text, f"ffmpeg could not decode the silence: {text[:300]}"
    mean_db = float(text.split("mean_volume:")[1].split("dB")[0])
    assert mean_db < -60, f"pause audio is not silent (mean {mean_db} dB)"


def test_pause_silence_follows_the_stream_format(has_ffmpeg):
    """A format change must re-encode rather than mix sample rates."""
    config.save_config({"stream": {"bitrate_kbps": 192, "sample_rate": 48000}})
    b = Broadcaster()
    frames = b._silence_frames()
    assert b._silence_key == (192, 48000, 2)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
         "-of", "json", "-"], input=frames, capture_output=True, timeout=30,
    )
    streams = json.loads(probe.stdout or b"{}").get("streams") or []
    assert streams and int(streams[0]["sample_rate"]) == 48000


# --- presence wiring ----------------------------------------------------------

def test_listener_changes_notify_subscribers():
    b = Broadcaster()
    hits: list[int] = []
    b.on_listeners_changed(lambda: hits.append(1))
    listener = b.add_listener()
    b.remove_listener(listener)
    assert len(hits) == 2, "connect and disconnect must both notify"


def test_the_apps_own_relay_is_served_but_never_counts_as_a_listener():
    """The Icecast pusher holds a /stream.mp3 connection for the whole lifetime
    of the app. If it counted, presence could never reach zero and the station
    would never pause itself — exactly what the first live run showed (web=1
    forever, nobody listening). It must still receive audio."""
    b = Broadcaster()
    relay = b.add_listener(audience=False)
    assert b.listener_count() == 0
    b._broadcast(b"audio")
    assert relay.queue.get_nowait() == b"audio", "the relay must still be fed"

    real = b.add_listener()
    assert b.listener_count() == 1
    b.remove_listener(real)
    assert b.listener_count() == 0
    b.remove_listener(relay)


def test_state_reports_the_presence_watcher():
    class Watcher:
        def status(self):
            return {"web": 1, "icecast": 0, "total": 1, "auto_paused": False}

    b = Broadcaster()
    assert b.state()["listeners_detail"] is None   # nothing attached yet
    b.set_listener_watcher(Watcher())
    detail = b.state()["listeners_detail"]
    assert detail["total"] == 1 and detail["auto_paused"] is False


def test_progress_measures_broadcast_audio_and_stops_when_the_audio_stops(
        music_dir, has_ffmpeg):
    """The UI's progress bar used to be `time.time() - started_at` — a wall clock
    that kept running while the station was paused, so a stopped song looked like
    it was still playing. Progress must be the audio actually broadcast, which
    stands still exactly when the audio does."""
    from tests.conftest import make_mp3

    path = make_mp3(music_dir / "progress.mp3", seconds=20)
    track = {"path": str(path), "title": "Progress", "duration": 20.0}
    b = Broadcaster()
    player = threading.Thread(
        target=b._play_file, args=(str(path), "track", {"track": track}),
        daemon=True)
    player.start()

    def wait_for(cond, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return True
            time.sleep(0.2)
        return False

    try:
        assert wait_for(lambda: b.state()["elapsed"] >= 1.0), (
            f"progress never started: {b.state()}")
        b.pause()
        frozen = b.state()["elapsed"]
        time.sleep(4)                        # wall-clock time passes...
        assert b.state()["elapsed"] == frozen, (
            "the paused station still reported progress "
            f"({frozen} -> {b.state()['elapsed']})")
        b.resume()
        assert wait_for(lambda: b.state()["elapsed"] > frozen + 1.0), (
            f"progress did not resume: {b.state()}")
    finally:
        b.stop()
        player.join(timeout=10)


def test_the_measured_length_replaces_a_wrong_catalogue_one(music_dir, has_ffmpeg):
    """The catalogue length is mutagen's header estimate, which lies about VBR
    files without a Xing header — the bar then hits 100% mid-song. Playing a file
    measures it exactly, so the wrong number is corrected at the natural end."""
    from app import db
    from tests.conftest import make_mp3

    path = make_mp3(music_dir / "mismeasured.mp3", seconds=3)
    conn = db.connect()
    conn.execute("INSERT INTO tracks(path, title, duration) VALUES(?, ?, ?)",
                 (str(path), "Mismeasured", 90.0))
    conn.commit()
    track_id = conn.execute("SELECT id FROM tracks WHERE path = ?",
                            (str(path),)).fetchone()["id"]

    b = Broadcaster()
    assert b._play_file(str(path), "track",
                        {"track": {"id": track_id, "path": str(path),
                                   "duration": 90.0}}) is True
    stored = db.connect().execute("SELECT duration FROM tracks WHERE id = ?",
                                  (track_id,)).fetchone()["duration"]
    assert 2.0 <= stored <= 4.5, (
        f"the catalogue still claims {stored}s for a 3s file")


def test_a_skipped_track_is_not_measured(music_dir, has_ffmpeg):
    """Only a track that played to its end measures the whole file; a skip must
    not rewrite the catalogue with how far it happened to get."""
    from app import db
    from tests.conftest import make_mp3

    path = make_mp3(music_dir / "cut.mp3", seconds=20)
    conn = db.connect()
    conn.execute("INSERT INTO tracks(path, title, duration) VALUES(?, ?, ?)",
                 (str(path), "Cut", 90.0))
    conn.commit()
    track_id = conn.execute("SELECT id FROM tracks WHERE path = ?",
                            (str(path),)).fetchone()["id"]

    b = Broadcaster()
    player = threading.Thread(
        target=b._play_file,
        args=(str(path), "track", {"track": {"id": track_id, "path": str(path),
                                             "duration": 90.0}}),
        daemon=True)
    player.start()
    time.sleep(2.5)
    b.skip()
    player.join(timeout=10)

    stored = db.connect().execute("SELECT duration FROM tracks WHERE id = ?",
                                  (track_id,)).fetchone()["duration"]
    assert stored == 90.0, f"a skipped play overwrote the length with {stored}s"


def test_a_skip_with_no_audio_yet_is_not_a_playback_failure(music_dir, monkeypatch):
    """Skip must not be counted as a failed file.

    Five quick skips (each landing before the encoder produced a chunk) used to
    reach the failure breaker: the loop backed off and the UI showed "5 queued
    files in a row could not be played — is the music library mounted?".
    """
    import io

    track = {"path": str(music_dir / "a.mp3"), "title": "Alpha"}
    b = Broadcaster()

    class _MuteOutput(io.BytesIO):
        """The encoder produces nothing: the user hits Skip while waiting."""

        def read(self, size: int | None = -1):
            b.skip()
            return b""

    class _SilentProcess:
        returncode = 0

        def __init__(self):
            self.stdout = _MuteOutput()
            self.stderr = io.BytesIO()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(stream_mod.subprocess, "Popen",
                        lambda *a, **k: _SilentProcess())
    assert b._play_file(track["path"], "track", {"track": track}) is True
    assert b._consecutive_failures == 0
    assert b._playback_error is None


def test_the_ffmpeg_stderr_sink_drops_the_banner_but_keeps_errors():
    """At INFO level the head is the banner; the error must still be readable."""
    sink = stream_mod._stderr_sink()
    sink.write(b"Input #0, mp3, from 'x.mp3':\n"
               b"  Duration: 00:03:21.00, start: 0.000000, bitrate: 128 kb/s\n"
               b"  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s\n"
               b"Stream mapping:\n"
               b"[silencedetect @ 0x55] silence_start: 12.0\n"
               b"size=     329KiB time=00:00:21.02 bitrate= 128.2kbits/s\n"
               b"[mp3 @ 0x55] Header missing\n"
               b"Error while decoding stream #0:0: Invalid data found\n")
    sink.seek(0)
    text = stream_mod._stderr_text(sink)
    assert "Header missing" in text
    assert "Invalid data found" in text
    assert "Stream mapping" not in text and "Duration:" not in text
    assert "silencedetect" not in text and "size=" not in text


def test_the_ffmpeg_stderr_sink_reads_and_closes():
    """ffmpeg's stderr goes to a file, not a pipe: nothing reads a pipe until
    the file is over, so a full one blocks the child mid-write and the whole
    broadcast hangs."""
    sink = stream_mod._stderr_sink()
    sink.write(b"boom " * 200)
    text = stream_mod._stderr_text(sink)
    assert text.startswith("boom")
    assert len(text) <= 300
    assert sink.closed
    assert stream_mod._stderr_text(None) == ""


def test_a_skip_while_the_station_is_paused_never_spins(has_ffmpeg):
    """A pause holds the line while nobody listens. A skip arriving in that
    state must not turn the hold into a 100 % CPU loop that sends no audio —
    that would idle the source out and Icecast would drop the mount."""
    b = Broadcaster()
    b.pause()
    b.skip()  # e.g. a skip that landed before the pause began
    done = threading.Event()
    threading.Thread(target=lambda: (b._pause_hold(), done.set()),
                     daemon=True).start()
    air = threading.Thread(target=b._keepalive, daemon=True)
    listener = b.add_listener()
    air.start()
    time.sleep(1.3)
    assert not done.is_set(), "the hold lasts as long as the pause does"
    b.resume()
    assert done.wait(5.0), "the hold must end when the pause does"
    b._stop.set()
    air.join(timeout=5)
    assert not listener.queue.empty(), "a paused station still has to send audio"


def test_a_skip_stops_the_idle_tone(has_ffmpeg):
    """The idle tone is a 5 s item: skip() must cut it short and reap the
    encoder instead of leaving it running (and unreaped) to the end."""
    b = Broadcaster()
    sent: list[int] = []
    b._broadcast = lambda data: sent.append(len(data))  # type: ignore[method-assign]
    done = threading.Event()
    threading.Thread(target=lambda: (b._idle_tone(), done.set()),
                     daemon=True).start()
    time.sleep(0.4)
    proc = b._proc
    b.skip()
    assert done.wait(6.0)
    before = list(sent)
    time.sleep(0.3)
    assert sent == before, "silence must not keep playing after a skip"
    assert proc is not None, "the encoder must be reachable from skip()"
    assert proc.poll() is not None, "the encoder is killed and reaped"


def test_the_frame_aligner_follows_the_mpeg_version():
    """MPEG-2/2.5 frames use half-stepped bitrates, their own sample rates and
    72 bytes per frame instead of 144: sizing an MPEG-2 stream with MPEG-1
    numbers mis-slices every frame (the whole point of this class)."""
    size = stream_mod.FrameAligner._frame_size
    assert size(bytes([0xFF, 0xFB, 0x90, 0x00])) == 417      # MPEG-1 128k/44.1k
    assert size(bytes([0xFF, 0xF2, 0x80, 0x00])) == 208      # MPEG-2 64k/22.05k
    assert size(bytes([0xFF, 0xE6, 0x90, 0x00])) is None     # reserved version
    assert size(bytes([0xFF, 0xFD, 0x90, 0x00])) is None     # Layer II, not III


def test_the_aligner_slices_mpeg2_streams_whole():
    frame = bytes([0xFF, 0xF2, 0x80, 0x00]) + b"\x00" * 204  # 208 bytes
    out = stream_mod.FrameAligner().feed(frame * 3 + b"\x00\x00")
    assert len(out) == 208 * 3          # three whole frames, no partial tail
