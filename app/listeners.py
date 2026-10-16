"""Who is listening — and stopping the station when nobody is.

Two independent sources of presence:

* HTTP clients on the app's own ``/stream.mp3`` (the web player, ``curl``).
  The broadcaster knows them exactly and reports the moment one arrives.
* Listeners on the Icecast mount (Winamp, VLC, Sonos). Only Icecast knows
  about those, so its status page is polled.

Presence drives one decision: with nobody listening for ``listener_grace_s``
the station pauses itself, and it resumes the moment somebody connects.

Two rules make that safe, and both were measured against the real server:

* **Unknown is not zero.** If Icecast is enabled but its status cannot be read,
  presence is unknown and the station keeps playing. A stats outage must never
  take the station off air.
* **Pausing must keep sending.** The broadcaster's keep-alive thread sends
  silence whenever nothing else is playing, instead of letting the socket go
  quiet. Icecast drops a source that sends nothing for ``source-timeout``
  (30 s in the ``icecast.xml`` the app renders for the bundled server), which
  would remove the mount exactly when a new listener wants to connect — and
  with no mount nobody can ever trigger the resume. The same guarantee covers
  the failures and the DJ breaks, which stall the audio for just as long.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import config

log = logging.getLogger(__name__)

# How often the local (in-process, free) listener count is looked at. Keeping
# this short is what makes a browser connecting resume playback within a tick.
TICK_S = 0.5
# Defaults, used when the config value is missing or unusable.
DEFAULT_GRACE_S = 30
DEFAULT_POLL_S = 5
ICECAST_TIMEOUT_S = 4.0


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def icecast_status() -> dict[str, Any]:
    """Live Icecast mount info for the web UI and for presence.

    Never raises: an unreachable Icecast reports ``known: False`` (presence
    unknown) with an ``error`` string, and keeps ``listeners`` at 0 so the UI's
    "not connected" render is unchanged.
    """
    if not config.get("icecast.enabled", False):
        return {"enabled": False, "connected": False, "known": False, "listeners": 0}
    host = config.get("icecast.host", "127.0.0.1")
    port = _as_int(config.get("icecast.port", 8008), 8008)
    mount = "/" + str(config.get("icecast.mount", "virtualdj")).lstrip("/")
    try:
        import httpx
        r = httpx.get(f"http://{host}:{port}/status-json.xsl?mount={mount}",
                      timeout=ICECAST_TIMEOUT_S)
        if r.status_code != 200:
            return {"enabled": True, "connected": False, "known": False,
                    "mount": mount, "listeners": 0,
                    "error": f"icecast returned {r.status_code}"}
        src = r.json().get("icestats", {}).get("source", {})
        if isinstance(src, list):
            src = src[0] if src else {}
        return {
            "enabled": True,
            "connected": True,
            "known": True,
            "mount": mount,
            "listeners": _as_int(src.get("listeners"), 0),
            "server_name": src.get("server_name"),
            "bitrate": src.get("bitrate"),
        }
    except Exception as exc:  # noqa: BLE001 - report any connectivity problem
        return {"enabled": True, "connected": False, "known": False,
                "mount": mount, "listeners": 0, "error": str(exc)}


class ListenerWatcher:
    """Pause playback when nobody is listening; resume when somebody is."""

    def __init__(self, broadcaster: Any) -> None:
        self._broadcaster = broadcaster
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        # Last Icecast answer. None = unknown (never polled, or unreachable).
        self._icecast: int | None = None
        self._icecast_error: str | None = None
        self._icecast_checked_at = 0.0
        # When presence first dropped to zero (None = somebody is listening).
        self._empty_since: float | None = None
        # True only while WE are the ones holding playback paused, so a manual
        # pause is never undone and a manual resume can always win.
        self._auto_paused = False

    # --- settings ---------------------------------------------------------

    def settings(self) -> tuple[bool, int, int]:
        """(enabled, grace seconds, icecast poll seconds) — read live."""
        enabled = bool(config.get("playback.pause_without_listeners", True))
        grace = max(0, _as_int(config.get("playback.listener_grace_s"), DEFAULT_GRACE_S))
        poll = max(1, _as_int(config.get("playback.listener_poll_s"), DEFAULT_POLL_S))
        return enabled, grace, poll

    # --- presence ---------------------------------------------------------

    def _refresh_icecast(self, now: float, poll: int) -> None:
        if now - self._icecast_checked_at < poll:
            return
        self._icecast_checked_at = now
        status = icecast_status()
        if not status.get("enabled"):
            self._icecast, self._icecast_error = None, None
            return
        if status.get("known"):
            self._icecast, self._icecast_error = status["listeners"], None
        else:
            # Unreachable Icecast = unknown presence, never "zero listeners".
            self._icecast, self._icecast_error = None, status.get("error")

    def presence(self) -> dict[str, Any]:
        """Current presence, without touching Icecast (see ``run``)."""
        web = self._broadcaster.listener_count()
        ice = self._icecast
        return {
            "web": web,
            "icecast": ice,
            "total": web + (ice or 0),
            "icecast_known": ice is not None,
            "icecast_error": self._icecast_error,
        }

    # --- the rule ---------------------------------------------------------

    def decide(self, presence: dict[str, Any], enabled: bool, grace: int,
               now: float) -> str:
        """What to do now: ``"pause"``, ``"resume"`` or ``"hold"``.

        Keeps the grace timer, which is the only mutable part: presence going to
        zero starts it, any listener (or an unknown Icecast) clears it.
        """
        if not enabled:
            self._empty_since = None
            return "resume" if self._auto_paused else "hold"
        if presence["total"] > 0:
            self._empty_since = None
            return "resume" if self._auto_paused else "hold"
        if not presence["icecast_known"] and config.get("icecast.enabled", False):
            # Icecast is on but not answering: unknown, so keep playing.
            self._empty_since = None
            return "resume" if self._auto_paused else "hold"
        if self._empty_since is None:
            self._empty_since = now
        if now - self._empty_since >= grace:
            return "hold" if self._auto_paused else "pause"
        return "hold"

    def _apply(self, action: str, grace: int) -> None:
        if action == "pause":
            self._auto_paused = True
            self._broadcaster.pause()
            log.info("no listeners for %ss — playback paused; the station stays "
                     "on air with silence so a listener can reconnect", grace)
        elif action == "resume" and self._auto_paused:
            self._auto_paused = False
            self._broadcaster.resume()
            log.info("listener connected — playback resumed")

    # --- thread -----------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                enabled, grace, poll = self.settings()
                now = time.monotonic()
                self._refresh_icecast(now, poll)
                self._apply(self.decide(self.presence(), enabled, grace, now), grace)
            except Exception:
                log.exception("listener watcher iteration failed; continuing")
            # Wake early when an HTTP listener arrives/leaves, otherwise on the
            # tick, which is also the resolution of the grace timer.
            self._wake.wait(TICK_S)
            self._wake.clear()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="vdj-listeners",
                                        daemon=True)
        self._thread.start()
        enabled, grace, poll = self.settings()
        log.info("listener watch: %s (grace %ss, icecast poll %ss)",
                 "pausing when nobody listens" if enabled else "off", grace, poll)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def notify(self) -> None:
        """Check immediately (called when an HTTP listener connects/leaves)."""
        self._wake.set()

    # --- introspection ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Presence + the pause state, for the web UI."""
        enabled, grace, poll = self.settings()
        presence = self.presence()
        grace_left = None
        if self._empty_since is not None and enabled:
            grace_left = max(0, round(grace - (time.monotonic() - self._empty_since), 1))
        with self._lock:
            return {
                **presence,
                "enabled": enabled,
                "auto_paused": self._auto_paused,
                "grace_s": grace,
                "grace_left": grace_left,
                "poll_s": poll,
            }


WATCHER: ListenerWatcher | None = None


def start_watcher(broadcaster: Any) -> ListenerWatcher:
    """Create (once) and start the watcher for this process."""
    global WATCHER
    if WATCHER is None:
        WATCHER = ListenerWatcher(broadcaster)
        broadcaster.set_listener_watcher(WATCHER)
        broadcaster.on_listeners_changed(WATCHER.notify)
    WATCHER.start()
    return WATCHER
