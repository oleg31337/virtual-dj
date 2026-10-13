"""Presence, end to end against a real server.

The station must stop when the last listener leaves and start again the moment
somebody connects — that is the whole feature, so it is checked over HTTP with
real bytes, not by calling the decision function.

Icecast listener counts are covered by tests/test_listeners.py and by the live
container check; this file pins the app's own /stream.mp3 path.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from tests.conftest import make_mp3

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def presence_server(tmp_path_factory):
    """A real server that pauses when nobody listens (2s grace: quick tests)."""
    tmp = tmp_path_factory.mktemp("presence")
    data_dir = tmp / "data"
    data_dir.mkdir()
    music = tmp / "music"
    for i, name in enumerate(["one", "two", "three"]):
        make_mp3(music / f"{name}.mp3", seconds=20, freq=330 + i * 110,
                 tags={"title": name.title(), "artist": f"Artist {i}",
                       "genre": "Rock"})
    (data_dir / "config.json").write_text(json.dumps({
        "music_dir": str(music),
        "dj": {"enabled": False},
        "enrich": {"enabled": False},
        "llm": {"enabled": False},
        "playback": {"pause_without_listeners": True,
                     "listener_grace_s": 2, "listener_poll_s": 1},
    }))

    port = _free_port()
    env = {**dict(os.environ), "VDJ_DATA_DIR": str(data_dir),
           "VDJ_DB_PATH": str(data_dir / "presence.sqlite3"),
           "VDJ_LOG_LEVEL": "warning"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 45
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            pytest.fail(f"server died on startup:\n{output[-2000:]}")
        try:
            if httpx.get(f"{base}/api/status", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.4)
    else:
        proc.kill()
        pytest.fail("server did not become ready in 45s")

    deadline = time.time() + 60
    while time.time() < deadline:
        if httpx.get(f"{base}/api/status", timeout=5).json()["library"]["total"] >= 3:
            break
        time.sleep(0.5)

    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def now_playing(base: str) -> dict[str, Any]:
    return httpx.get(f"{base}/api/status", timeout=10).json()["now_playing"]


def wait_for(base: str, predicate: Callable[[dict], bool],
             timeout: float = 25.0) -> dict[str, Any]:
    """Poll the live state until it satisfies ``predicate``."""
    deadline = time.time() + timeout
    state = now_playing(base)
    while time.time() < deadline:
        state = now_playing(base)
        if predicate(state):
            return state
        time.sleep(0.5)
    return state


def plays_recorded(base: str) -> int:
    history = httpx.get(f"{base}/api/history?limit=50", timeout=10).json()
    return len(history if isinstance(history, list) else history.get("history", []))


def test_the_station_pauses_itself_with_nobody_listening(presence_server):
    state = wait_for(presence_server, lambda s: s["paused"] is True)
    assert state["paused"] is True, f"never paused with no listeners: {state}"
    detail = state["listeners_detail"]
    assert detail["auto_paused"] is True, f"the pause must be ours: {detail}"
    assert detail["total"] == 0
    assert detail["enabled"] is True and detail["grace_s"] == 2


def test_a_paused_station_plays_nothing_but_stays_reachable(presence_server):
    """Paused means the queue is not consumed — while the stream still answers."""
    base = presence_server
    before = plays_recorded(base)
    time.sleep(6)                        # several grace windows and ticks
    assert plays_recorded(base) == before, "a paused station consumed the queue"
    assert now_playing(base)["paused"] is True

    with httpx.stream("GET", f"{base}/stream.mp3", timeout=20) as resp:
        assert resp.status_code == 200
        next(resp.iter_bytes())           # silence keeps the mount answering
        resp.close()


def test_connecting_resumes_and_disconnecting_pauses_again(presence_server):
    base = presence_server
    with httpx.stream("GET", f"{base}/stream.mp3", timeout=40) as resp:
        assert resp.status_code == 200
        state = wait_for(base, lambda s: s["paused"] is False, timeout=15)
        assert state["paused"] is False, f"a connected listener must restart it: {state}"
        assert state["listeners"] == 1
        assert state["listeners_detail"]["auto_paused"] is False
        got = 0
        for chunk in resp.iter_bytes(4096):
            got += len(chunk)
            if got > 20_000:
                break
        assert got > 20_000, f"only {got} bytes reached a connected listener"
    state = wait_for(base, lambda s: s["paused"] is True)
    assert state["paused"] is True, f"never paused after the listener left: {state}"
    assert state["listeners_detail"]["auto_paused"] is True


def test_the_stations_own_relay_does_not_keep_it_awake(presence_server):
    """The Icecast pusher reads this stream too (?internal=1). It must be served
    — and must not count as an audience, or the station could never pause."""
    base = presence_server
    state = wait_for(base, lambda s: s["paused"] is True)
    assert state["paused"] is True, f"precondition: paused, got {state}"

    with httpx.stream("GET", f"{base}/stream.mp3?internal=1", timeout=40) as resp:
        assert resp.status_code == 200
        bytes_seen = 0
        for chunk in resp.iter_bytes(4096):
            bytes_seen += len(chunk)
            if bytes_seen > 10_000:
                break
        assert bytes_seen > 10_000, "the relay must still be fed the audio"

        for _ in range(6):                 # several watcher ticks
            time.sleep(1)
            state = now_playing(base)
            assert state["listeners"] == 0, (
                f"the station's own relay counted as a listener: {state}")
            assert state["paused"] is True, (
                f"the relay kept the station playing: {state}")
