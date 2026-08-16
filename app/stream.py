"""The always-on broadcast engine.

A single background thread ("the transmitter") decodes each queue item through
ffmpeg into a constant-bitrate MP3 byte stream and paces it out in real time.
Every connected HTTP listener gets its own bounded buffer fed from that one
stream, so 1 listener or 20 listeners cost the same CPU — and everybody hears
the same thing at the same moment, exactly like a real radio station.

Output is plain MP3 over an infinite chunked HTTP response, which is what
Winamp, VLC, Sonos and every other Icecast-compatible client expect.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from . import config, db
from .scheduler import SCHEDULER

log = logging.getLogger(__name__)

CHUNK = 2048
# How much audio a slow client may fall behind before we drop it.
CLIENT_BUFFER_CHUNKS = 512
# How far ahead of real time we let the pipeline run. Icecast keeps several
# seconds of cushion so players (VLC's default 1s network cache) never
# underrun on jitter, track boundaries or encoder spawns.
MAX_AHEAD_SECONDS = 3.0
# On connect we replay this much of the recent broadcast to the new listener
# (Icecast's "burst"), so its player can fill its buffer instantly.
BURST_SECONDS = 3.0


class FrameAligner:
    """Splits a raw byte feed into complete MPEG audio frames.

    The broadcast stream must never contain partial frames: a frame cut in
    half by a killed encoder or by the burst replay makes every decoder
    (VLC, ffmpeg) emit "Header missing" / "Invalid data" glitches at that
    point — heard as choppiness. Every byte we broadcast is therefore a
    whole number of MPEG frames; a trailing partial frame at end-of-stream
    is discarded (the next track's first frame is a clean resync point).
    """

    _BITRATES = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    _SAMPLE_RATES = [44100, 48000, 32000]

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _frame_size(header: bytes) -> int | None:
        b0, b1, b2, b3 = header
        if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
            return None
        bitrate_idx = (b2 >> 4) & 0x0F
        sr_idx = (b2 >> 2) & 0x03
        if bitrate_idx in (0, 15) or sr_idx == 3:
            return None  # free-format / reserved
        bitrate = FrameAligner._BITRATES[bitrate_idx] * 1000
        sample_rate = FrameAligner._SAMPLE_RATES[sr_idx]
        pad = (b2 >> 1) & 0x01
        return 144 * bitrate // sample_rate + pad

    def feed(self, data: bytes) -> bytes:
        """Append ``data``; return all complete frames contained in it."""
        self._buf.extend(data)
        out = bytearray()
        while True:
            if len(self._buf) < 4:
                break
            size = self._frame_size(self._buf[:4])
            if size is None:
                del self._buf[0]  # not a frame start: resync one byte at a time
                continue
            if len(self._buf) < size:
                break
            out.extend(self._buf[:size])
            del self._buf[:size]
        return bytes(out)

    def discard_leftover(self) -> None:
        """Drop the trailing partial frame (called at end-of-stream)."""
        self._buf.clear()


class Listener:
    """One connected client."""

    __slots__ = ("queue", "id", "connected_at", "dropped")

    def __init__(self, listener_id: int) -> None:
        self.id = listener_id
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=CLIENT_BUFFER_CHUNKS)
        self.connected_at = time.time()
        self.dropped = False

    def put(self, data: bytes) -> None:
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            # Client cannot keep up: drain half the buffer and continue rather
            # than blocking the transmitter for everyone else.
            self.dropped = True
            for _ in range(CLIENT_BUFFER_CHUNKS // 2):
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass

    def close(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


class Broadcaster:
    def __init__(self) -> None:
        self._listeners: dict[int, Listener] = {}
        self._listener_seq = 0
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._paused = threading.Event()
        self._proc: subprocess.Popen | None = None
        # Rolling history of recent broadcast bytes (Icecast-style burst
        # buffer): replayed to new listeners so players start with a full
        # jitter buffer instead of starving for the first few seconds.
        # 256 chunks x 4KB = 1MB (~64s at 128kbps); we only replay the last
        # BURST_SECONDS worth.
        self._history: deque[bytes] = deque(maxlen=256)
        self._now_playing: dict[str, Any] = {
            "kind": "idle", "track": None, "dj_text": None,
            "started_at": None, "duration": None,
        }
        # The last DJ phrase spoken, and the id of the track it announced.
        # The "DJ SAID" panel shows the phrase only while that announced track
        # is playing; when the next track starts (the relevant song has ended)
        # the panel clears. Without the id binding the phrase would either
        # vanish the instant the music began, or linger forever.
        self._last_dj_text: str | None = None
        self._dj_for_track_id: int | None = None
        self._listeners_changed: list[Callable[[], None]] = []
        self._on_change: list[Callable[[dict[str, Any]], None]] = []
        self._started_stream_at: float | None = None

    # --- listener registry ------------------------------------------------

    def add_listener(self) -> Listener:
        with self._lock:
            self._listener_seq += 1
            listener = Listener(self._listener_seq)
            burst_chunks = max(1, int(BURST_SECONDS * self._bytes_per_second()
                                      / CHUNK))
            for item in list(self._history)[-burst_chunks:]:
                listener.put(item)
            self._listeners[listener.id] = listener
        self._notify_change()
        return listener

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.pop(listener.id, None)
        listener.close()
        self._notify_change()

    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    def _broadcast(self, data: bytes) -> None:
        with self._lock:
            listeners = list(self._listeners.values())
            self._history.append(data)
        for listener in listeners:
            listener.put(data)

    # --- observers --------------------------------------------------------

    def on_change(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._on_change.append(callback)

    def _notify_change(self) -> None:
        state = self.state()
        for callback in list(self._on_change):
            try:
                callback(state)
            except Exception:
                log.debug("on_change callback failed", exc_info=True)

    # --- transport controls -----------------------------------------------

    def skip(self) -> None:
        self._skip.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def pause(self) -> None:
        """Pause = broadcast silence, keeping every client connected."""
        self._paused.set()
        self._notify_change()

    def resume(self) -> None:
        self._paused.clear()
        self._notify_change()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # --- encoding ---------------------------------------------------------

    def _encode_args(self, path: str, extra_filters: list[str] | None = None) -> list[str]:
        bitrate = int(config.get("stream.bitrate_kbps", 128))
        rate = int(config.get("stream.sample_rate", 44100))
        channels = int(config.get("stream.channels", 2))
        filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"] + (extra_filters or [])
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", path,
            "-vn", "-map", "0:a:0",
            "-af", ",".join(filters),
            "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            "-ar", str(rate), "-ac", str(channels),
            "-f", "mp3", "-write_xing", "0", "-id3v2_version", "0",
            "pipe:1",
        ]

    def _bytes_per_second(self) -> float:
        return int(config.get("stream.bitrate_kbps", 128)) * 1000.0 / 8.0

    def _play_file(self, path: str, kind: str, meta: dict[str, Any]) -> bool:
        """Stream one file to all listeners, paced in real time."""
        if not Path(path).exists():
            log.warning("missing media file: %s", path)
            return False

        self._skip.clear()
        args = self._encode_args(path)
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            log.error("failed to launch ffmpeg: %s", exc)
            return False

        self._proc = proc
        self._now_playing = {
            "kind": kind,
            "track": meta.get("track"),
            "dj_text": meta.get("dj_text"),
            "program": meta.get("program"),
            "started_at": time.time(),
            "duration": meta.get("duration"),
        }
        # A DJ break carries the spoken phrase and the id of the track it
        # introduces; remember both so the UI can show the phrase for exactly
        # the duration of that track. The binding is overwritten by the next
        # DJ break, and a non-matching track id simply stops showing it
        # (handled in state()), so we never need to clear it explicitly.
        if meta.get("dj_text") and meta.get("track"):
            self._last_dj_text = meta["dj_text"]
            self._dj_for_track_id = meta["track"].get("id")
        self._notify_change()

        bps = self._bytes_per_second()
        sent = 0
        start = time.monotonic()
        aligner = FrameAligner()
        try:
            while not self._stop.is_set() and not self._skip.is_set():
                raw = proc.stdout.read(CHUNK)
                if not raw:
                    break
                data = aligner.feed(raw)
                if not data:
                    continue

                # Pace: keep a MAX_AHEAD_SECONDS cushion of audio ahead of
                # real time so client-side jitter buffers (VLC's 1s network
                # cache) never run dry on scheduler jitter or track changes.
                while not self._stop.is_set() and not self._skip.is_set():
                    ahead = (sent / bps) - (time.monotonic() - start)
                    if ahead <= MAX_AHEAD_SECONDS:
                        break
                    time.sleep(min(ahead - MAX_AHEAD_SECONDS, 0.25))

                if self._paused.is_set():
                    # Hold the position; emit nothing (clients buffer/underrun
                    # gracefully) until resumed.
                    while self._paused.is_set() and not self._stop.is_set():
                        time.sleep(0.2)
                    start = time.monotonic() - (sent / bps)

                self._broadcast(data)
                sent += len(data)
        except Exception:
            log.exception("error while streaming %s", path)
        finally:
            aligner.discard_leftover()  # never broadcast a partial frame
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                stderr = proc.stderr.read() or b""
                if proc.wait(timeout=5) not in (0, -9) and stderr:
                    log.warning("ffmpeg: %s", stderr.decode("utf-8", "replace")[:300])
            except Exception:
                pass
            self._proc = None
        return sent > 0

    # --- main loop --------------------------------------------------------

    def _idle_tone(self) -> None:
        """Broadcast a short silence when the library yields nothing."""
        bitrate = int(config.get("stream.bitrate_kbps", 128))
        rate = int(config.get("stream.sample_rate", 44100))
        channels = int(config.get("stream.channels", 2))
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i",
            f"anullsrc=r={rate}:cl={'stereo' if channels == 2 else 'mono'}",
            "-t", "5", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            "-f", "mp3", "-write_xing", "0", "pipe:1",
        ]
        self._now_playing = {
            "kind": "idle", "track": None, "dj_text": None,
            "started_at": time.time(), "duration": 5.0,
        }
        self._notify_change()
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
        except OSError:
            time.sleep(5)
            return
        bps = self._bytes_per_second()
        sent, start = 0, time.monotonic()
        aligner = FrameAligner()
        while not self._stop.is_set():
            raw = proc.stdout.read(CHUNK)
            if not raw:
                break
            data = aligner.feed(raw)
            if not data:
                continue
            while (sent / bps) - (time.monotonic() - start) > MAX_AHEAD_SECONDS:
                time.sleep(0.1)
                if self._stop.is_set():
                    break
            self._broadcast(data)
            sent += len(data)
        aligner.discard_leftover()
        if proc.poll() is None:
            proc.kill()

    def _run(self) -> None:
        log.info("broadcaster started")
        while not self._stop.is_set():
            try:
                nxt = SCHEDULER.pop_next()
                if nxt is None:
                    self._idle_tone()
                    continue
                track, dj_break, program = nxt

                if dj_break and dj_break.get("audio_path"):
                    self._play_file(
                        dj_break["audio_path"], "dj",
                        {"track": track, "dj_text": dj_break.get("text"),
                         "duration": dj_break.get("duration"),
                         "program": program},
                    )

                if self._stop.is_set():
                    break

                played = self._play_file(
                    track["path"], "track",
                    {"track": track, "duration": track.get("duration"),
                     "program": program},
                )
                if played:
                    try:
                        db.record_play(track.get("id"))
                    except Exception:
                        log.debug("could not record play", exc_info=True)
            except Exception:
                log.exception("broadcast loop error; continuing")
                time.sleep(1)
        log.info("broadcaster stopped")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        self._stop.clear()
        self._started_stream_at = time.time()
        self._thread = threading.Thread(target=self._run, name="vdj-broadcast",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        with self._lock:
            listeners = list(self._listeners.values())
        for listener in listeners:
            listener.close()
        if self._thread:
            self._thread.join(timeout=5)

    # --- introspection ----------------------------------------------------

    def state(self) -> dict[str, Any]:
        now = self._now_playing
        elapsed = None
        if now.get("started_at"):
            elapsed = round(time.time() - now["started_at"], 1)
        # Show the remembered DJ phrase only for the track it introduced. Once
        # the next track starts (the announced song has ended) it clears.
        now_track = now.get("track") or {}
        show_dj = (self._last_dj_text is not None
                   and self._dj_for_track_id is not None
                   and now_track.get("id") == self._dj_for_track_id
                   and now.get("kind") in ("dj", "track"))
        dj_text = self._last_dj_text if show_dj else None
        return {
            "kind": now.get("kind"),
            "track": now.get("track"),
            "dj_text": dj_text,
            "elapsed": elapsed,
            "duration": now.get("duration"),
            "paused": self.paused,
            "listeners": self.listener_count(),
            "uptime": (round(time.time() - self._started_stream_at, 1)
                       if self._started_stream_at else None),
            "scheduler": SCHEDULER.status(),
        }


BROADCASTER = Broadcaster()
