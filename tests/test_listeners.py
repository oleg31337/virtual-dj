"""Presence: stop the station when nobody listens, start it when somebody does.

No threads and no sockets here — the rule is a decision function, so the whole
matrix is exercised directly, and the one place that talks to Icecast is faked.
"""

from __future__ import annotations

import time
from typing import Any

from app import config, listeners


class FakeBroadcaster:
    """Records transport calls; stands in for ``stream.Broadcaster``."""

    def __init__(self, web: int = 0) -> None:
        self.web = web
        self.calls: list[str] = []
        self.watcher = None
        self.callbacks: list[Any] = []

    def listener_count(self) -> int:
        return self.web

    def pause(self) -> None:
        self.calls.append("pause")

    def resume(self) -> None:
        self.calls.append("resume")

    def set_listener_watcher(self, watcher) -> None:
        self.watcher = watcher

    def on_listeners_changed(self, callback) -> None:
        self.callbacks.append(callback)


def presence(web: int = 0, ice: int | None = 0, known: bool = True) -> dict[str, Any]:
    return {"web": web, "icecast": ice, "total": web + (ice or 0),
            "icecast_known": known, "icecast_error": None}


def tick(watcher: listeners.ListenerWatcher, pres: dict[str, Any], now: float,
         enabled: bool = True, grace: int = 30) -> str:
    """One watcher iteration: decide, then apply (what the thread does)."""
    action = watcher.decide(pres, enabled, grace, now)
    watcher._apply(action, grace)
    return action


# --- the rule -----------------------------------------------------------------

def test_somebody_listening_keeps_the_music_going():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    assert tick(w, presence(web=1), 100.0) == "hold"
    assert tick(w, presence(ice=2), 100.0) == "hold"
    assert b.calls == []


def test_nobody_listening_pauses_after_the_grace():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    assert tick(w, presence(), 100.0) == "hold"      # grace starts here
    assert tick(w, presence(), 129.0) == "hold"      # still inside it
    assert tick(w, presence(), 130.0) == "pause"     # expired
    assert b.calls == ["pause"]
    # It must not re-pause on every tick while it stays empty.
    assert tick(w, presence(), 400.0) == "hold"
    assert b.calls == ["pause"]


def test_a_listener_connecting_resumes_playback():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    tick(w, presence(), 0.0)
    assert tick(w, presence(), 31.0) == "pause"
    assert tick(w, presence(web=1), 32.0) == "resume"
    assert b.calls == ["pause", "resume"]
    assert w.status()["auto_paused"] is False


def test_a_brief_dropout_inside_the_grace_never_stops_the_music():
    """A player that reconnects must not make the station stop and start."""
    b = FakeBroadcaster(web=1)
    w = listeners.ListenerWatcher(b)
    tick(w, presence(), 0.0)          # last listener left
    tick(w, presence(web=1), 10.0)    # back again
    tick(w, presence(), 20.0)         # and gone: the grace restarts here
    assert tick(w, presence(), 40.0) == "hold"
    assert tick(w, presence(), 51.0) == "pause"
    assert b.calls == ["pause"]


def test_zero_grace_pauses_immediately():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    assert tick(w, presence(), 5.0, grace=0) == "pause"


def test_icecast_that_cannot_be_asked_is_never_treated_as_empty():
    """A stats outage must not take the station off air."""
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    config.save_config({"icecast": {"enabled": True}})
    for now in (0.0, 100.0, 5000.0):
        assert tick(w, presence(known=False), now) == "hold"
    assert b.calls == []
    # ...and if we had already paused on a known-zero count, an Icecast that
    # stops answering puts the music back on rather than leaving it stopped.
    assert tick(w, presence(known=False), 5001.0) == "hold"
    w._auto_paused = True
    assert tick(w, presence(known=False), 5002.0) == "resume"


def test_with_icecast_off_only_web_listeners_count():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    config.save_config({"icecast": {"enabled": False}})
    assert tick(w, presence(known=False), 0.0) == "hold"
    assert tick(w, presence(known=False), 31.0) == "pause"


def test_icecast_listeners_count_as_presence():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    tick(w, presence(ice=0), 0.0)
    assert tick(w, presence(ice=3), 5.0) == "hold"
    assert b.calls == []


def test_switching_the_feature_off_puts_the_music_back_on():
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    tick(w, presence(), 0.0)
    assert tick(w, presence(), 31.0) == "pause"
    assert tick(w, presence(), 32.0, enabled=False) == "resume"
    assert b.calls == ["pause", "resume"]
    # And it stays off: no more pausing however long it is empty.
    assert tick(w, presence(), 1000.0, enabled=False) == "hold"
    assert b.calls == ["pause", "resume"]


def test_a_manual_pause_is_never_undone_by_the_watcher():
    """Only our own pause is ours to undo — the UI button still wins."""
    b = FakeBroadcaster(web=1)
    w = listeners.ListenerWatcher(b)
    b.calls.append("pause")          # the user pressed Pause, not the watcher
    assert tick(w, presence(web=1), 0.0) == "hold"
    assert tick(w, presence(web=0), 60.0) == "hold"   # already paused anyway
    assert b.calls == ["pause"]


def test_a_pause_taken_while_auto_paused_stays_the_users():
    """A listener arriving must not switch the station back on over the user.

    We had stopped the station because nobody was listening; the user then
    pressed Pause. Our flag said the stop was still ours, so the next listener
    connection used to undo a deliberate pause.
    """
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    tick(w, presence(), 0.0)
    assert tick(w, presence(), 31.0) == "pause"
    w.note_manual_transport()                          # the UI's Pause button
    assert tick(w, presence(web=1), 32.0) == "hold"
    assert b.calls == ["pause"]


def test_a_manual_play_does_not_disable_the_rule():
    """After the user presses Play, the next empty house must still pause.

    The reverse of the test above, and the same root cause: the stale flag made
    the watcher answer "hold" forever, so pause-without-listeners was silently
    off until the process restarted.
    """
    b = FakeBroadcaster()
    w = listeners.ListenerWatcher(b)
    tick(w, presence(), 0.0)
    assert tick(w, presence(), 31.0) == "pause"
    w.note_manual_transport()                          # the UI's Play button
    assert tick(w, presence(), 40.0) == "hold"         # a new grace starts
    assert tick(w, presence(), 71.0) == "pause"        # and it expires
    assert b.calls == ["pause", "pause"]


def test_a_leftover_auto_pause_flag_is_dropped_on_its_own():
    """Self-heal: believe the broadcaster, not our memory of a pause."""

    class RunningBroadcaster(FakeBroadcaster):
        paused = False          # the station is audible, whatever we remember

    b = RunningBroadcaster()
    w = listeners.ListenerWatcher(b)
    w._auto_paused = True
    assert tick(w, presence(), 0.0) == "hold"      # nothing to resume
    assert w.status()["auto_paused"] is False
    assert tick(w, presence(), 31.0) == "pause"    # the rule works again
    assert b.calls == ["pause"]


# --- presence sources ---------------------------------------------------------

class _Response:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> Any:
        return self._payload


def _icecast_on() -> None:
    config.save_config({"icecast": {"enabled": True, "host": "ice", "port": 8008,
                                    "mount": "virtualdj"}})


def test_icecast_status_is_parsed(monkeypatch):
    _icecast_on()
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(
        200, {"icestats": {"source": {"listeners": 3, "server_name": "Virtual DJ",
                                      "bitrate": 128}}}))
    st = listeners.icecast_status()
    assert st["known"] is True and st["connected"] is True and st["listeners"] == 3
    assert st["mount"] == "/virtualdj" and st["server_name"] == "Virtual DJ"


def test_icecast_source_list_picks_our_mount(monkeypatch):
    """Icecast ignores ?mount=: with two streams, only ours may be counted.

    Reading the first entry meant reading somebody else's listener count — and
    that count is what pauses the station, so it would have gone silent with a
    listener connected.
    """
    _icecast_on()
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(200, {"icestats": {"source": [
        {"listeners": 9, "listenurl": "http://ice:8008/relay"},
        {"listeners": 2, "listenurl": "http://ice:8008/virtualdj"},
    ]}}))
    st = listeners.icecast_status()
    assert st["known"] is True and st["listeners"] == 2


def test_an_icecast_without_our_mount_is_unknown_not_empty(monkeypatch):
    """No source on our mount is not "nobody is listening".

    The mount can be between sources, and with no mount nobody could connect to
    trigger a resume — so the station must keep playing rather than pause itself
    into a corner.
    """
    _icecast_on()
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(200, {"icestats": {"source": [
        {"listeners": 4, "listenurl": "http://ice:8008/relay"},
    ]}}))
    st = listeners.icecast_status()
    assert st["known"] is False and st["listeners"] == 0
    assert "/virtualdj" in st["error"]
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(
        200, {"icestats": {"icestats_source_missing": 1}}))
    assert listeners.icecast_status()["known"] is False


def test_a_single_source_without_a_listenurl_is_ours(monkeypatch):
    """Icecast can answer without listenurl; one stream can only be ours."""
    _icecast_on()
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(
        200, {"icestats": {"source": [{"listeners": 5}]}}))
    assert listeners.icecast_status()["listeners"] == 5


def test_an_unreachable_icecast_is_unknown_not_zero(monkeypatch):
    _icecast_on()

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("httpx.get", boom)
    st = listeners.icecast_status()
    assert st["known"] is False and st["connected"] is False
    assert st["listeners"] == 0                     # the UI keeps its old render
    assert "connection refused" in st["error"]


def test_an_icecast_error_page_is_unknown_too(monkeypatch):
    _icecast_on()
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response(500))
    st = listeners.icecast_status()
    assert st["known"] is False and "500" in st["error"]


def test_icecast_disabled_reports_off():
    config.save_config({"icecast": {"enabled": False}})
    assert listeners.icecast_status() == {"enabled": False, "connected": False,
                                          "known": False, "listeners": 0}


# --- wiring -------------------------------------------------------------------

def test_watcher_reports_presence_and_the_countdown():
    b = FakeBroadcaster(web=2)
    w = listeners.ListenerWatcher(b)
    w._icecast = 1
    st = w.status()
    assert (st["web"], st["icecast"], st["total"]) == (2, 1, 3)
    assert st["auto_paused"] is False and st["grace_left"] is None
    assert st["enabled"] is True and st["grace_s"] == 30 and st["poll_s"] == 5

    b.web, w._icecast = 0, 0
    w.decide(w.presence(), True, 30, time.monotonic())
    st = w.status()
    assert st["total"] == 0 and 0 <= st["grace_left"] <= 30


def test_settings_come_from_config_and_clamp_to_something_usable():
    config.save_config({"playback": {"pause_without_listeners": False,
                                     "listener_grace_s": 0, "listener_poll_s": 0}})
    w = listeners.ListenerWatcher(FakeBroadcaster())
    assert w.settings() == (False, 0, 1)          # a poll of 0s would spin


def test_start_watcher_attaches_itself_to_the_broadcaster(monkeypatch):
    monkeypatch.setattr(listeners, "WATCHER", None)
    b = FakeBroadcaster()
    w = listeners.start_watcher(b)
    try:
        assert b.watcher is w
        assert b.callbacks == [w.notify]
        assert w.status()["enabled"] is True
    finally:
        w.stop()
