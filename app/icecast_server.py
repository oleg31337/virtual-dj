"""Managed Icecast2 server (bundled in the same container as the app).

The app owns Icecast end-to-end: it renders icecast.xml from the same
data/config.json the rest of the UI uses, then spawns `icecast2 -b` as root so
Icecast can perform its <changeowner> privilege-drop to the unprivileged
`nobody` user. MP3 source mounts only register and serve when that drop
happens, which is why Icecast must be launched this way (a plain non-root
foreground start silently fails to serve source mounts).

`icecast2 -b` *daemonizes*: the launcher process exits immediately (exit 0)
after forking the background daemon, which then drops to nobody and binds the
port. We therefore supervise the daemon by polling its listening socket (and
its pidfile), NOT by watching the launcher's exit code -- the launcher exiting
0 is success, not a crash. Treating the launcher's exit as a failure causes a
storm of "Could not create listener socket" errors (each relaunch collides
with the already-bound daemon).
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

from . import config

log = logging.getLogger(__name__)

_TMPL_PATH = os.path.join(os.path.dirname(__file__), "..", "icecast.xml.tmpl")
_RENDERED_PATH = "/tmp/icecast/icecast.xml"
_LOGDIR = "/tmp/icecast"
_PIDFILE = os.path.join(_LOGDIR, "icecast.pid")


class ManagedIcecast:
    """Render config and supervise an ``icecast2 -b`` daemon (managed mode)."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._daemon_pid: int | None = None

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
            pid = self._daemon_pid
            self._daemon_pid = None
        if pid and self._pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def restart(self) -> None:
        """Restart the Icecast daemon with the current (possibly changed) config.

        Kills the running daemon; the supervise loop notices the port stopped
        listening and re-renders icecast.xml (picking up a new port/mount/...) and
        rebinds. No-op if the server isn't currently up (the supervisor will
        launch it from the latest config on next start()).
        """
        with self._lock:
            pid = self._daemon_pid
            self._daemon_pid = None
        if pid and self._pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            # Wait briefly for the port to free so the relaunch can bind it.
            for _ in range(30):
                if not self._pid_alive(pid):
                    break
                time.sleep(0.1)

    # --- rendering ---------------------------------------------------------

    def render_config(self) -> str:
        """Render icecast.xml from the app config. Returns the path written."""
        tmpl = open(_TMPL_PATH, "r", encoding="utf-8").read()
        source_password = str(config.get("icecast.source_password", "hackme"))
        admin_password = os.environ.get("VDJ_ICECAST_ADMIN_PASSWORD", "admin")
        relay_password = os.environ.get("VDJ_ICECAST_RELAY_PASSWORD", source_password)
        port = int(config.get("icecast.port", 8008))
        hostname = str(config.get("icecast.hostname", "virtual-dj"))
        mount = str(config.get("icecast.mount", "virtualdj")).lstrip("/") or "virtualdj"
        stream_name = str(config.get("stream.station_name", "Virtual DJ"))
        os.makedirs(_LOGDIR, exist_ok=True)
        rendered = tmpl
        for ph, val in (
            ("${ICECAST_SOURCE_PASSWORD}", source_password),
            ("${ICECAST_ADMIN_PASSWORD}", admin_password),
            ("${ICECAST_RELAY_PASSWORD}", relay_password),
            ("${ICECAST_PORT}", str(port)),
            ("${ICECAST_HOSTNAME}", hostname),
            ("${ICECAST_MOUNT}", mount),
            ("${ICECAST_STREAM_NAME}", stream_name),
        ):
            rendered = rendered.replace(ph, val)
        with open(_RENDERED_PATH, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        log.info(
            "rendered icecast config -> %s (mount /%s, port %s)",
            _RENDERED_PATH, mount, port,
        )
        return _RENDERED_PATH

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _is_listening(port: int, host: str = "127.0.0.1") -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _read_daemon_pid(self) -> int | None:
        try:
            with open(_PIDFILE, "r", encoding="utf-8") as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    # --- supervision -------------------------------------------------------

    def _wait_listening(self, port: int, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            if self._is_listening(port):
                return True
            time.sleep(0.5)
        return False

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            # Drop any stale pidfile from a previous daemon before relaunching.
            if os.path.exists(_PIDFILE):
                try:
                    os.remove(_PIDFILE)
                except OSError:
                    pass
            cfg = self.render_config()
            port = int(config.get("icecast.port", 8008))
            cmd = ["icecast2", "-b", "-c", cfg]
            log.info("icecast server launching: %s", " ".join(cmd))
            try:
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

            # `icecast2 -b` daemonizes: the launcher exits (code 0) once forked.
            # We confirm the daemon is actually up by polling the listening
            # socket -- never by the launcher's exit code.
            if not self._wait_listening(port):
                tail = b""
                if proc.stderr:
                    try:
                        tail = proc.stderr.read() or b""
                    except OSError:
                        pass
                log.warning(
                    "icecast server failed to start: %s",
                    tail.decode("utf-8", "replace").rstrip()[:600],
                )
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 15)
                continue

            # Reap the launcher (it exited) and record the daemon pid.
            proc.poll()
            with self._lock:
                self._daemon_pid = self._read_daemon_pid()
            log.info("icecast server up (daemon pid %s)", self._daemon_pid)
            backoff = 1.0

            # Healthy: stay up until the daemon stops serving, then relaunch;
            # or stop requested, in which case terminate the daemon.
            while not self._stop.is_set():
                if not self._is_listening(port):
                    log.warning("icecast daemon stopped serving; restarting")
                    break
                time.sleep(1.0)
            else:
                with self._lock:
                    pid = self._daemon_pid
                    self._daemon_pid = None
                if pid and self._pid_alive(pid):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                break


SERVER = ManagedIcecast()
