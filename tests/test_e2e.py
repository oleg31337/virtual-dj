"""End-to-end: boot a real uvicorn server, tune in like VLC would, and prove
the bytes coming off the wire are continuous, correctly-paced, decodable audio.

This is the test that actually answers "does the radio station work".
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import make_mp3

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real server process with its own data dir and a real music library."""
    tmp = tmp_path_factory.mktemp("e2e")
    data_dir = tmp / "data"
    data_dir.mkdir()
    music = tmp / "music"
    for i, name in enumerate(["one", "two", "three"]):
        make_mp3(music / f"{name}.mp3", seconds=20, freq=330 + i * 110,
                 tags={"title": name.title(), "artist": f"Artist {i}",
                       "genre": "Rock"})

    # DJ off: this test isolates the audio transport. The DJ pipeline has its
    # own live test below.
    (data_dir / "config.json").write_text(json.dumps({
        "music_dir": str(music),
        "dj": {"enabled": False},
        "enrich": {"enabled": False},
        "llm": {"enabled": False},
    }))

    port = _free_port()
    env = {
        **dict(__import__("os").environ),
        "VDJ_DATA_DIR": str(data_dir),
        "VDJ_DB_PATH": str(data_dir / "e2e.sqlite3"),
        "VDJ_LOG_LEVEL": "warning",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 45
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", "replace")
            pytest.fail(f"server died on startup:\n{output[-2000:]}")
        try:
            if httpx.get(f"{base}/api/status", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.4)
    else:
        proc.kill()
        pytest.fail("server did not become ready in 45s")

    # Wait for the library scan to finish so the queue has content.
    deadline = time.time() + 60
    while time.time() < deadline:
        status = httpx.get(f"{base}/api/status", timeout=5).json()
        if not status["scan"]["running"] and status["library"]["total"] >= 3:
            break
        time.sleep(0.5)

    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_server_is_up_and_library_scanned(live_server):
    status = httpx.get(f"{live_server}/api/status", timeout=10).json()
    assert status["library"]["total"] == 3


def test_stream_headers_are_icecast_compatible(live_server):
    """What Winamp / VLC / Sonos look at when opening the URL."""
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=20) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/mpeg")
        assert resp.headers["icy-name"]
        assert resp.headers["icy-br"]
        assert "content-length" not in resp.headers  # unbounded stream
        next(resp.iter_bytes())  # confirm data actually flows
        resp.close()


def test_stream_bytes_decode_as_continuous_audio(live_server):
    """Pull ~8 seconds like a real player and decode the result with ffprobe."""
    data = bytearray()
    start = time.time()
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as resp:
        for chunk in resp.iter_bytes(4096):
            data.extend(chunk)
            if time.time() - start > 8:
                break
        resp.close()

    assert len(data) > 50_000, f"only got {len(data)} bytes in 8s"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "json", "-"],
        input=bytes(data), capture_output=True, timeout=30,
    )
    info = json.loads(probe.stdout or b"{}")
    streams = info.get("streams") or []
    assert streams, f"ffprobe could not decode the stream: {probe.stderr[:400]}"
    assert streams[0]["codec_name"] == "mp3"
    assert int(streams[0]["sample_rate"]) == 44100
    assert int(streams[0]["channels"]) == 2

    # And it must decode to real PCM, not just parse as a header.
    decode = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-"],
        input=bytes(data), capture_output=True, timeout=60,
    )
    pcm_seconds = len(decode.stdout) / (44100 * 2 * 2)
    assert pcm_seconds > 3, f"only {pcm_seconds:.1f}s of decodable audio"


def test_stream_is_paced_not_dumped(live_server):
    """~6s of listening should yield roughly 6s of audio, not a whole file."""
    bitrate_bytes = 128 * 1000 / 8
    data = bytearray()
    start = time.time()
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as resp:
        for chunk in resp.iter_bytes(4096):
            data.extend(chunk)
            if time.time() - start > 6:
                break
    seconds_of_audio = len(data) / bitrate_bytes
    assert 1.5 < seconds_of_audio < 15, (
        f"expected ~6s of audio in 6s, got {seconds_of_audio:.1f}s"
    )
    resp.close()


def test_two_listeners_receive_the_same_broadcast(live_server):
    """Radio semantics: everyone hears the same thing at the same time."""
    a, b = bytearray(), bytearray()
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as ra, \
         httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as rb:
        ia, ib = ra.iter_bytes(4096), rb.iter_bytes(4096)
        start = time.time()
        while time.time() - start < 5:
            a.extend(next(ia))
            b.extend(next(ib))
        ra.close()
        rb.close()

    assert len(a) > 10_000 and len(b) > 10_000
    # Both tuned in at the same moment, so the payloads should overlap heavily.
    shorter = min(len(a), len(b))
    assert bytes(a[:shorter])[:8192] == bytes(b[:shorter])[:8192]


def test_listener_count_tracks_connections(live_server):
    """A connected player must be counted; a closed one must be dropped.

    NOTE: uses curl (a real streaming client) instead of httpx — httpx's
    HTTP/1.1 implementation closes/resets unbounded responses, which makes
    it unsuitable for holding a radio stream open.
    """
    # Give any disconnect from a previous test time to settle so the count
    # below is deterministic (server-side cleanup is async).
    deadline = time.time() + 15
    while time.time() < deadline:
        if httpx.get(f"{live_server}/api/status",
                     timeout=10).json()["now_playing"]["listeners"] == 0:
            break
        time.sleep(0.3)
    before = httpx.get(f"{live_server}/api/status", timeout=10).json()
    assert before["now_playing"]["listeners"] == 0

    player = subprocess.Popen(
        ["curl", "-s", "-m", "20", "-o", os.devnull,
         f"{live_server}/stream.mp3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 10
        during = 0
        while time.time() < deadline:
            during = httpx.get(f"{live_server}/api/status",
                               timeout=10).json()["now_playing"]["listeners"]
            if during == 1:
                break
            time.sleep(0.2)
        assert during == 1, "connected player was not counted"
    finally:
        player.terminate()
        player.wait(timeout=5)

    deadline = time.time() + 15
    while time.time() < deadline:
        after = httpx.get(f"{live_server}/api/status", timeout=10).json()
        if after["now_playing"]["listeners"] == 0:
            break
        time.sleep(0.5)
    assert after["now_playing"]["listeners"] == 0, "listener was not cleaned up"


def test_no_decode_errors_across_forced_track_switches(live_server):
    """The choppiness regression guard: with encoders killed mid-track, the
    byte stream must still contain only complete MPEG frames. ffmpeg used as
    a player reports 'Header missing' / 'Invalid data' otherwise — exactly
    what VLC hears as choppiness."""
    import threading

    stop = threading.Event()

    def skipper():
        while not stop.is_set():
            time.sleep(2.0)
            try:
                httpx.post(f"{live_server}/api/transport/skip", timeout=5)
            except Exception:
                pass

    th = threading.Thread(target=skipper, daemon=True)
    th.start()
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i",
             f"{live_server}/stream.mp3", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(25)  # ~12 forced track switches
        proc.kill()
        out = proc.stderr.read()
    finally:
        stop.set()

    problems = [l for l in out.splitlines()
                if any(k in l.lower() for k in
                       ("invalid", "header", "underflow", "underrun",
                        "corrupt", "non-monotonous", "error"))]
    assert not problems, f"decoder complained during track switches:\n{problems[:6]}"


def test_now_playing_reports_a_real_track(live_server):
    deadline = time.time() + 30
    while time.time() < deadline:
        state = httpx.get(f"{live_server}/api/status", timeout=10).json()["now_playing"]
        if state["track"]:
            assert state["track"]["title"] in {"One", "Two", "Three"}
            assert state["kind"] == "track"
            return
        time.sleep(1)
    pytest.fail("no track ever started playing")


def test_skip_advances_to_another_track(live_server):
    def current():
        return httpx.get(f"{live_server}/api/status",
                         timeout=10).json()["now_playing"]["track"]

    deadline = time.time() + 30
    while time.time() < deadline and not current():
        time.sleep(0.5)
    first = current()
    assert first is not None

    httpx.post(f"{live_server}/api/transport/skip", timeout=10)
    deadline = time.time() + 20
    while time.time() < deadline:
        now = current()
        if now and now["path"] != first["path"]:
            return
        time.sleep(0.5)
    pytest.fail("skip did not advance the stream")


def test_stream_survives_a_client_disconnecting_mid_track(live_server):
    """A player closing abruptly must not take the station down."""
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as resp:
        next(resp.iter_bytes())
        resp.close()
    time.sleep(1)
    data = bytearray()
    start = time.time()
    with httpx.stream("GET", f"{live_server}/stream.mp3", timeout=30) as resp:
        for chunk in resp.iter_bytes(4096):
            data.extend(chunk)
            if time.time() - start > 3:
                break
        resp.close()
    assert len(data) > 10_000, "stream stopped after a client disconnect"


def test_queue_and_history_populate_over_time(live_server):
    queue = httpx.get(f"{live_server}/api/queue", timeout=10).json()
    assert len(queue) > 0
    assert queue[0]["track"]["path"]

    deadline = time.time() + 40
    while time.time() < deadline:
        history = httpx.get(f"{live_server}/api/history", timeout=10).json()
        if history:
            assert history[0]["title"]
            return
        httpx.post(f"{live_server}/api/transport/skip", timeout=10)
        time.sleep(2)
    pytest.fail("nothing was ever recorded in play history")
