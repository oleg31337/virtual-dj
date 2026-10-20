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


def _pick_mount(source: Any, mount: str) -> dict[str, Any] | None:
    """Our stream out of Icecast's ``icestats.source``, or None if it is absent.

    Icecast 2.4.4 ignores the ``?mount=`` parameter, so a server with a second
    stream (a relay, a fallback, an old mount still draining) answers with a
    list — and reading the first entry, as this used to, means reading somebody
    else's listener count. That count is what pauses the station, so the station
    would go silent with a listener connected. ``listenurl`` is the only field
    that says which mount an entry is, so the match is made there.
    """
    entries = source if isinstance(source, list) else [source]
    candidates = [entry for entry in entries if isinstance(entry, dict)]
    if not candidates:
        return None
    if len(candidates) == 1 and not candidates[0].get("listenurl"):
        return candidates[0]  # one stream, no url to match: it is ours
    wanted = mount.rstrip("/")
    for entry in candidates:
        url = str(entry.get("listenurl") or "").split("?")[0].rstrip("/")
        if url.endswith(wanted):
            return entry
    return None


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
        src = _pick_mount(r.json().get("icestats", {}).get("source"), mount)
        if src is None:
            # Our mount is not among Icecast's sources. That is NOT "zero
            # listeners": the mount may be between sources, or Icecast may be
            # listing a different stream. Pausing on it would take the station
            # off air — and, with no source on the mount, nobody could ever
            # connect to trigger the resume. Unknown presence keeps it playing.
            return {"enabled": True, "connected": True, "known": False,
                    "mount": mount, "listeners": 0,
                    "error": f"mount {mount} is not on air"}
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
        # Our own pause can be undone by something else (the UI's Play button).
        # A flag that outlives that pause is worse than useless: while it is set
        # the rule below answers "hold" for an empty house, so pause-without-
        # listeners silently stops working until the process restarts.
        live = getattr(self._broadcaster, "paused", None)
        if self._auto_paused and (live() if callable(live) else live) is False:
            self._auto_paused = False
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

    def note_manual_transport(self) -> None:
        """The user pressed Play or Pause in the UI.

        Either way our own pause is no longer the reason the station is stopped,
        so the flag has to go. Otherwise both directions break: a Pause taken
        while we had auto-paused is undone the moment a listener connects (the
        watcher believes the pause is still its own to end), and a Play leaves
        the flag set, which stops the next empty house from ever pausing.

        The grace timer restarts with it: if the user turns the music on over an
        empty house, ``grace`` is measured from their press, not from the moment
        the last listener left (which would pause it again on the next tick).
        """
        self._auto_paused = False
        self._empty_since = None

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
