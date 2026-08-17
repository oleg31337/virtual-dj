"""Managed Icecast2 server (bundled in the same container as the app).

The app owns Icecast end-to-end: it renders ``icecast.xml`` from the same
``data/config.json`` the rest of the UI uses, then spawns ``icecast2 -b`` as
root so Icecast can perform its <changeowner> privilege-drop to the
unprivileged ``nobody`` user. MP3 source mounts only register and serve when
that drop happens, which is why Icecast must be launched this way (a plain
non-root foreground start silently fails to serve source mounts).

The Icecast pusher (``icecast.py``) then relays the app's own ``/stream.mp3``
into the mount over loopback. External players (Winamp/VLC/Sonos) connect to
the published Icecast port; the web player keeps using ``/stream.mp3``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time

from . import config

log = logging.getLogger(__name__)

_TMPL_PATH = os.path.join(os.path.dirname(__file__), "..", "icecast.xml.tmpl")
_RENDERED_PATH = "/tmp/icecast/icecast.xml"
_LOGDIR = "/tmp/icecast"


class ManagedIcecast:
    """Render config and supervise an ``icecast2 -b`` process (managed mode)."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not config.get("icecast.enabled", False):
            log.info("icecast delivery disabled; managed server not started")
            return
        if shutil.which("icecast2") is None:
            log.error("icecast2 binary not found; cannot run managed icecast")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, name="vdj-icecast-server", daemon=True
        )
        self._thread.start()
        log.info("managed icecast server starting")

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                # icecast2 -b daemonizes; kill by PID we recorded.
                proc.terminate()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # --- rendering ---------------------------------------------------------

    def render_config(self) -> str:
        """Render icecast.xml from the app config. Returns the path written."""
        tmpl = open(_TMPL_PATH, "r", encoding="utf-8").read()
        enabled = bool(config.get("icecast.enabled", False))
        source_password = str(config.get("icecast.source_password", "hackme"))
        admin_password = os.environ.get("VDJ_ICECAST_ADMIN_PASSWORD", "admin")
        relay_password = os.environ.get("VDJ_ICECAST_RELAY_PASSWORD", source_password)
        port = int(config.get("icecast.port", 8008))
        hostname = str(config.get("icecast.hostname", "virtual-dj"))
        mount = str(config.get("icecast.mount", "virtualdj")).lstrip("/") or "virtualdj"
        stream_name = str(config.get("stream.station_name", "Virtual DJ"))
        os.makedirs(_LOGDIR, exist_ok=True)
        rendered = tmpl.replace("${ICECAST_SOURCE_PASSWORD}", source_password)
        rendered = rendered.replace("${ICECAST_ADMIN_PASSWORD}", admin_password)
        rendered = rendered.replace("${ICECAST_RELAY_PASSWORD}", relay_password)
        rendered = rendered.replace("${ICECAST_PORT}", str(port))
        rendered = rendered.replace("${ICECAST_HOSTNAME}", hostname)
        rendered = rendered.replace("${ICECAST_MOUNT}", mount)
        rendered = rendered.replace("${ICECAST_STREAM_NAME}", stream_name)
        with open(_RENDERED_PATH, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        log.info("rendered icecast config -> %s (mount /%s)", _RENDERED_PATH, mount)
        return _RENDERED_PATH

    # --- supervision -------------------------------------------------------

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cfg = self.render_config()
            cmd = ["icecast2", "-b", "-c", cfg]
            log.info("icecast server launching: %s", " ".join(cmd))
            try:
                # -b daemonizes, so Popen returns immediately with the daemon
                # still running. We record the PID icecast2 reports.
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except OSError as exc:
                log.error("failed to launch icecast server: %s", exc)
                if not self._stop.wait(min(backoff, 15)):
                    backoff = min(backoff * 2, 15)
                continue

            with self._lock:
                self._proc = proc

            # Give icecast a moment to come up; the daemon's PID is our handle.
            time.sleep(2.0)
            if proc.poll() is not None:
                # Daemon failed to start; surface its stderr tail.
                tail = b""
                if proc.stderr:
                    tail = proc.stderr.read() or b""
                log.warning("icecast server exited early: %s",
                            tail.decode("utf-8", "replace").rstrip()[:500])
                with self._lock:
                    self._proc = None
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 15)
                continue

            # Healthy: wait until stop is requested, then terminate the daemon.
            while not self._stop.is_set():
                if proc.poll() is not None:
                    # Daemon died unexpectedly; restart after backoff.
                    with self._lock:
                        self._proc = None
                    log.warning("icecast server died; restarting")
                    if self._stop.wait(backoff):
                        break
                    backoff = min(backoff * 2, 15)
                    break
                time.sleep(1.0)
            else:
                # stop requested
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                break


SERVER = ManagedIcecast()
