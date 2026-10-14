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

import collections
import logging
import re
import shutil
import subprocess
import threading
import time

from . import config

log = logging.getLogger(__name__)

# Within the container / on the host, the app reaches its own HTTP server on
# loopback. The pusher reads the MP3 from there and pushes it to Icecast.
#
# `?internal=1` marks this connection as the station's own plumbing: it must be
# served like any other client, but it must NOT count as a listener, or presence
# could never reach zero and the station would never pause itself.
_PUSH_SOURCE_URL = "http://127.0.0.1:{port}/stream.mp3?internal=1"


def _as_int(value: object, default: int) -> int:
    """A config-file integer that never raises: empty/junk -> ``default``."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _mask(cmd: list[str]) -> str:
    """Command line for the log with any URL userinfo replaced by ``***``.

    The pusher's Icecast URL carries the source password, so the raw command
    must never be logged.
    """
    return " ".join(
        re.sub(r"://([^:/@\s]+):[^@\s]*@", r"://\1:***@", part) for part in cmd
    )


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
        port = _as_int(config.get("icecast.port", 8008), 8008)
        mount = config.get("icecast.mount", "virtualdj")
        password = config.get("icecast.source_password", "hackme")
        # env_int, not int(os.environ[...]): compose forwards an unset var as
        # the empty string, and int("") used to raise ValueError inside the
        # supervisor thread -- killing the pusher for good.
        app_port = config.env_int("VDJ_PORT", 8420)
        source = _PUSH_SOURCE_URL.format(port=app_port)
        # Icecast requires the source to declare a Content-Type. ffmpeg's
        # icecast muxer sends audio/mpeg for mp3; we make it explicit.
        #
        # The broadcaster already emits a clean, constant-bitrate MP3 stream
        # (CBR, no Xing/ID3 frames), so we relay it byte-for-byte with
        # "-c:a copy" -- there is NO reason to re-encode it here. Re-encoding
        # (the old "-c:a libmp3lame -b:a 128k") plus the "-re" read throttle
        # was the root cause of Icecast timing out the source ("socket
        # timeout") roughly every minute: the encode stalled ffmpeg's output to
        # Icecast long enough to trip Icecast's source-timeout, Icecast dropped
        # the source, every listener was disconnected (Winamp "Buffer 0%", 0
        # listeners), and the pusher had to reconnect.
        #
        # Without "-re", ffmpeg reads /stream.mp3 as fast as it is produced and
        # buffers it; its own buffer smooths the tiny (sub-second) gaps at track
        # boundaries so the Icecast socket is never idle long enough to time out.
        # The reconnect flags make the input side self-healing if /stream.mp3
        # ever hiccups.
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            # Self-heal if the app's /stream.mp3 feed ever stalls/drops.
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
            "-i", source,
            # Relay the already-encoded MP3 without re-encoding.
            "-c:a", "copy",
            "-content_type", "audio/mpeg",
            "-f", "mp3",
            f"icecast://source:{password}@{host}:{port}/{mount}",
        ]

    def _supervise(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            # Build INSIDE the loop's guard: a bad config value (or a missing
            # ffmpeg) must be logged and retried, never allowed to escape and
            # kill this supervisor thread for the life of the process.
            try:
                cmd = self._build_cmd()
            except Exception as exc:  # noqa: BLE001 - keep the supervisor alive
                log.error("cannot build the icecast pusher command: %s", exc)
                if self._stop.wait(min(backoff, 15)):
                    break
                backoff = min(backoff * 2, 15)
                continue
            # The command embeds the source password; log it masked.
            log.info("icecast pusher launching: %s", _mask(cmd))
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
            # doesn't block the child, and log ffmpeg's complaints. The buffer
            # is bounded: with -reconnect ffmpeg stays alive for days, and only
            # the tail is ever reported.
            stderr: collections.deque[bytes] = collections.deque(maxlen=20)
            logged = 0
            while not self._stop.is_set():
                line = proc.stderr.readline() if proc.stderr else b""
                if line:
                    stderr.append(line)
                    if logged < 20:
                        logged += 1
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
            tail = b"".join(list(stderr)[-5:]).decode("utf-8", "replace").rstrip()
            log.warning("icecast pusher exited (code %s); restarting. %s",
                        code, tail)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15)


PUSHER = IcecastPusher()
