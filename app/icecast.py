"""Icecast delivery: relay the internal MP3 feed to an Icecast server.

The broadcaster already produces a clean, real-time MP3 stream and serves it
from ``/stream.mp3``. Winamp / VLC / Sonos consume *Icecast* natively (a real
SHOUTcast/Icecast stream with proper client-side buffering); they stutter on
the app's custom HTTP sender. So when Icecast delivery is enabled, this module
runs a background ``ffmpeg`` ("the pusher") that ingests the app's own
``/stream.mp3`` over loopback and relays it to Icecast. The web player keeps
using ``/stream.mp3`` directly; external players use the Icecast mount.

The pusher is supervised: if ffmpeg exits (Icecast restart, transient error)
it is relaunched after a short backoff, so the relay is self-healing.
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

# Within the container / on the host, the app reaches its own HTTP server on
# loopback. The pusher reads the MP3 from there and pushes it to Icecast.
_PUSH_SOURCE_URL = "http://127.0.0.1:{port}/stream.mp3"


class IcecastPusher:
    """Supervise a long-running ffmpeg relay from /stream.mp3 to Icecast."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not config.get("icecast.enabled", False):
            log.info("icecast delivery disabled; pusher not started")
            return
        if not shutil.which("ffmpeg"):
            log.error("ffmpeg not found; cannot run icecast pusher")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, name="vdj-icecast-pusher", daemon=True
        )
        self._thread.start()
        log.info("icecast pusher started (target %s:%s/%s)",
                 config.get("icecast.host"), config.get("icecast.port"),
                 config.get("icecast.mount"))

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # --- internals --------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        host = config.get("icecast.host", "127.0.0.1")
        port = int(config.get("icecast.port", 8008))
        mount = config.get("icecast.mount", "virtualdj")
        password = config.get("icecast.source_password", "hackme")
        bitrate = int(config.get("stream.bitrate_kbps", 128))
        app_port = int(os.environ.get("VDJ_PORT", "8420"))
        source = _PUSH_SOURCE_URL.format(port=app_port)
        # Icecast requires the source to declare a Content-Type. ffmpeg's
        # icecast muxer sends audio/mpeg for mp3; we make it explicit.
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            # Read the live feed at its native (real-time) rate.
            "-re",
            "-i", source,
            "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            "-content_type", "audio/mpeg",
            "-f", "mp3",
            f"icecast://source:{password}@{host}:{port}/{mount}",
        ]

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cmd = self._build_cmd()
            log.info("icecast pusher launching: %s", " ".join(cmd))
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except OSError as exc:
                log.error("failed to launch icecast pusher: %s", exc)
                if not self._stop.wait(min(backoff, 15)):
                    backoff = min(backoff * 2, 15)
                continue

            with self._lock:
                self._proc = proc

            # Wait for it to exit (or for stop). Drain stderr so the pipe
            # doesn't block the child, and log ffmpeg's complaints.
            stderr = []
            while not self._stop.is_set():
                line = proc.stderr.readline() if proc.stderr else b""
                if line:
                    stderr.append(line)
                    if len(stderr) <= 20:
                        log.warning("icecast pusher: %s",
                                    line.decode("utf-8", "replace").rstrip())
                if proc.poll() is not None:
                    break
                time.sleep(0.1)

            with self._lock:
                if self._proc is proc:
                    self._proc = None

            if self._stop.is_set():
                if proc.poll() is None:
                    proc.kill()
                break

            # Unexpected exit (Icecast down, auth error, ...). Log and retry.
            code = proc.wait() if proc.poll() is None else proc.returncode
            tail = b"".join(stderr[-5:]).decode("utf-8", "replace").rstrip()
            log.warning("icecast pusher exited (code %s); restarting. %s",
                        code, tail)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15)


PUSHER = IcecastPusher()
