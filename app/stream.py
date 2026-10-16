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
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from . import config, db, loudness
from .scheduler import SCHEDULER
from . import icecast_metadata

log = logging.getLogger(__name__)

# Size of one broadcast chunk. Small enough that clients' buffers are topped up
# continuously (no multi-second silence between clumps), large enough to avoid
# per-byte syscall overhead: at 128 kbps this is ~0.125 s of audio, well within
# any player's jitter buffer. This is what the whole delivery path hands out --
# `_get_chunk` in server.py relays one item at a time.
CHUNK = 2048
# How much audio a slow client may fall behind before we drop it.
CLIENT_BUFFER_CHUNKS = 512
# How far ahead of real time we let the pipeline run. Icecast keeps several
# seconds of cushion so players (VLC's default 1s network cache) never
# underrun on jitter, track boundaries or encoder spawns.
MAX_AHEAD_SECONDS = 3.0
# Consecutive unplayable files before the broadcast loop starts backing off.
# Below it a few bad files are just skipped; at/above it the station treats the
# library as unavailable (share not mounted) and slows to a crawl, so a mass of
# missing files can never spin the CPU or hammer the DJ LLM with refills.
FAILURE_BREAKER = 5
# On connect we replay this much of the recent broadcast to the new listener
# (Icecast's "burst"), so its player can fill its buffer instantly.
BURST_SECONDS = 3.0
# The station must never go quiet for longer than this. Icecast drops a source
# that sends nothing for <source-timeout> (30 s in the icecast.xml that
# app.icecast_server renders for the bundled server; a standalone
# deploy/icecast.xml uses 10 s) and a dropped source takes the mount with it:
# an external player that tunes in then gets nothing at all -- it plays for a
# moment, gives up, and has to be stopped and started again -- while a player
# that was already connected is cut off. Everything that interrupts the audio
# (a pause, the failure backoff when the library is not mounted, an LLM writing
# the next DJ break, piper speaking it, an encoder starting up) is covered by
# _keepalive, which sends silence whenever nothing else has gone out for this
# long.
KEEPALIVE_GAP_S = 1.0
# How often the keep-alive looks. A clock read per tick, and short enough that
# silence resumes within a fraction of a second of any stall.
KEEPALIVE_TICK_S = 0.25


class FrameAligner:
    """Splits a raw byte feed into complete MPEG audio frames.

    The broadcast stream must never contain partial frames: a frame cut in
    half by a killed encoder or by the burst replay makes every decoder
    (VLC, ffmpeg) emit "Header missing" / "Invalid data" glitches at that
    point — heard as choppiness. Every byte we broadcast is therefore a
    whole number of MPEG frames; a trailing partial frame at end-of-stream
    is discarded (the next track's first frame is a clean resync point).
    """

    # Layer III only (that is what the encoder is asked for). The tables differ
    # per MPEG version: MPEG-2/2.5 uses half-stepped bitrates, its own sample
    # rates, and 72 bytes per frame instead of 144. Guessing MPEG-1 numbers for
    # an MPEG-2 stream (any stream.sample_rate below 32000) mis-sized every
    # frame and cut the broadcast mid-frame — the one thing this class exists to
    # prevent.  keys: 1 = MPEG-1, 2 = MPEG-2, 25 = MPEG-2.5
    _BITRATES = {
        1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
        2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    }
    _SAMPLE_RATES = {
        1: [44100, 48000, 32000],
        2: [22050, 24000, 16000],
        25: [11025, 12000, 8000],
    }
    _VERSIONS = {0: 25, 2: 2, 3: 1}   # bit pattern -> MPEG version (1 is reserved)

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _frame_size(header: bytes) -> int | None:
        b0, b1, b2, _ = header
        if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
            return None
        if (b1 >> 1) & 0x03 != 1:
            return None  # Layer I/II — not what this app encodes
        version = FrameAligner._VERSIONS.get((b1 >> 3) & 0x03)
        if version is None:
            return None  # reserved version bits: a false sync, not a header
        bitrate_idx = (b2 >> 4) & 0x0F
        sr_idx = (b2 >> 2) & 0x03
        if bitrate_idx in (0, 15) or sr_idx == 3:
            return None  # free-format / reserved
        table = 1 if version == 1 else 2
        bitrate = FrameAligner._BITRATES[table][bitrate_idx] * 1000
        sample_rate = FrameAligner._SAMPLE_RATES[version][sr_idx]
        pad = (b2 >> 1) & 0x01
        return (144 if version == 1 else 72) * bitrate // sample_rate + pad

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


def _stderr_sink() -> Any:
    """A child's stderr sink that can never block the child.

    A pipe is the obvious choice and it is a trap: nothing reads it until the
    file is over, so once ffmpeg fills the ~64 KB pipe buffer it blocks inside
    write() — stdout stops and the broadcast hangs for every listener until
    somebody skips the track.
    """
    try:
        return tempfile.TemporaryFile()
    except OSError:
        return None


def _stderr_text(sink: Any, limit: int = 300) -> str:
    """The head of what a child wrote to `_stderr_sink`, then close it."""
    if sink is None:
        return ""
    try:
        sink.seek(0)
        return sink.read(64 * 1024).decode("utf-8", "replace")[:limit].strip()
    except (AttributeError, OSError, ValueError):
        return ""
    finally:
        try:
            sink.close()
        except (AttributeError, OSError):
            pass


class Listener:
    """One connected client.

    ``audience`` marks a real listener (a browser player, mpv, curl). The app's
    own Icecast relay also holds a /stream.mp3 connection — it must be served,
    but it must NOT count as somebody listening, or presence could never reach
    zero and the station would never pause itself.
    """

    __slots__ = ("queue", "id", "dropped", "audience")

    def __init__(self, listener_id: int, audience: bool = True) -> None:
        self.id = listener_id
        self.audience = audience
        self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=CLIENT_BUFFER_CHUNKS)
        self.dropped = False

    def put(self, data: bytes) -> None:
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            # Client cannot keep up: drain half the buffer and continue rather
            # than blocking the transmitter for everyone else.
            if not self.dropped:
                log.warning("listener %s cannot keep up: dropping buffered audio",
                            self.id)
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
        # Presence watcher (app/listeners.py), attached at startup so state()
        # can report who is listening and who paused the station.
        self._watcher: Any = None
        # Cached silence, encoded once per stream format: what the keep-alive
        # broadcasts whenever nothing else is playing (see _keepalive).
        self._silence: bytes | None = None
        self._silence_key: tuple[int, int, int] | None = None
        # When the last chunk went out, when the next silence frame is due, and
        # the thread that keeps the air filled (see _keepalive).
        self._last_audio_at = 0.0
        self._next_silence_at = 0.0
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_warned = False
        # How much audio of the current item has actually gone out, in seconds.
        # This — never a wall clock — is the progress the UI shows: it stops
        # when the audio stops (a pause holds it, a stalled decoder too), where
        # `time.time() - started_at` kept the progress bar crawling through a
        # paused, silent song.
        self._played_seconds = 0.0
        # Rolling history of recent broadcast bytes (Icecast-style burst
        # buffer): replayed to new listeners so players start with a full
        # jitter buffer instead of starving for the first few seconds.
        # 256 chunks x 2KB = 512KB (~32s at 128kbps); we only replay the last
        # BURST_SECONDS worth.
        self._history: deque[bytes] = deque(maxlen=256)
        self._now_playing: dict[str, Any] = {
            "kind": "idle", "track": None, "dj_text": None,
            "duration": None,
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
        # Circuit breaker for a library that cannot be played at all (share not
        # mounted, folder moved): consecutive failures + the message shown in
        # the UI. Without it the loop spins at ~1000 pops/s.
        self._consecutive_failures = 0
        self._playback_error: str | None = None

    # --- listener registry ------------------------------------------------

    def add_listener(self, audience: bool = True) -> Listener:
        """Register a client. ``audience=False`` for the app's own relay."""
        with self._lock:
            self._listener_seq += 1
            listener = Listener(self._listener_seq, audience=audience)
            burst_chunks = max(1, int(BURST_SECONDS * self._bytes_per_second()
                                      / CHUNK))
            for item in list(self._history)[-burst_chunks:]:
                listener.put(item)
            self._listeners[listener.id] = listener
        self._notify_change()
        self._notify_listeners_changed()
        return listener

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.pop(listener.id, None)
        listener.close()
        self._notify_change()
        self._notify_listeners_changed()

    def listener_count(self) -> int:
        """How many people are listening (the app's own relay does not count)."""
        with self._lock:
            return sum(1 for listener in self._listeners.values() if listener.audience)

    def on_listeners_changed(self, callback: Callable[[], None]) -> None:
        """Called when a listener connects or leaves (never on a track change).

        The presence watcher uses this to react the instant a browser connects
        instead of waiting for its next tick.
        """
        self._listeners_changed.append(callback)

    def _notify_listeners_changed(self) -> None:
        for callback in list(self._listeners_changed):
            try:
                callback()
            except Exception:
                log.debug("listeners-changed callback failed", exc_info=True)

    def set_listener_watcher(self, watcher: Any) -> None:
        """Attach the presence watcher so ``state()`` can report it."""
        self._watcher = watcher

    def _broadcast(self, data: bytes) -> None:
        with self._lock:
            # Stamped here, not by the callers: every path that puts audio on
            # the wire (a track, a DJ break, the idle tone, the keep-alive's
            # silence) goes through this method, so the keep-alive can never
            # send a second of silence on top of audio that just left.
            self._last_audio_at = time.monotonic()
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

    @staticmethod
    def _gain_filter(gain_db: float | None) -> str:
        """The audio filter that sets this track's level.

        A track measured by ``app/loudness.py`` plays at its own fixed gain —
        deterministic, transparent, and correct from the first second. Anything
        not measured yet (and everything with normalization switched off) keeps
        the legacy dynamic ``loudnorm`` chain, which normalizes on the fly:
        it rides the level inside a song and ramps in at every track start.
        """
        cfg = loudness.settings()
        if cfg["enabled"] and gain_db is not None:
            try:
                return f"volume={float(gain_db):.2f}dB"
            except (TypeError, ValueError):
                pass
        return (f"loudnorm=I={cfg['target_lufs']}:"
                f"TP={cfg['true_peak_ceiling']}:LRA=11")

    def _encode_args(self, path: str, gain_db: float | None = None) -> list[str]:
        bitrate = int(config.get("stream.bitrate_kbps", 128))
        rate = int(config.get("stream.sample_rate", 44100))
        channels = int(config.get("stream.channels", 2))
        filters = [self._gain_filter(gain_db)]
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

    def _silence_frames(self) -> bytes:
        """One second of MP3 silence as whole frames, cached per format.

        This is what a pause broadcasts (see ``_pause_hold``). Anullsrc goes
        through the same aligner as real audio so the stream never carries a
        partial frame, and the format follows the stream config, so changing
        bitrate/sample rate re-encodes instead of mixing formats.
        """
        bitrate = int(config.get("stream.bitrate_kbps", 128))
        rate = int(config.get("stream.sample_rate", 44100))
        channels = int(config.get("stream.channels", 2))
        key = (bitrate, rate, channels)
        if self._silence is not None and self._silence_key == key:
            return self._silence
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i",
            f"anullsrc=r={rate}:cl={'stereo' if channels == 2 else 'mono'}",
            "-t", "1", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            "-ar", str(rate), "-ac", str(channels),
            "-f", "mp3", "-write_xing", "0", "-id3v2_version", "0", "pipe:1",
        ]
        try:
            proc = subprocess.run(args, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, timeout=20)
            raw = proc.stdout or b""
        except Exception:
            log.warning("could not generate pause silence", exc_info=True)
            return b""
        frames = FrameAligner().feed(raw)  # whole frames only
        self._silence, self._silence_key = frames, key
        return frames

    def _pause_hold(self) -> None:
        """Wait out a pause; ``_keepalive`` keeps the station on air.

        This deliberately sends nothing itself. The audio a paused station
        broadcasts *is* silence, but it must not have two owners: a pause, an
        LLM writing the next DJ break, piper speaking it and the failure
        backoff all need the same guarantee, and two senders would put two
        seconds of audio on the wire per second — growing a latency cushion on
        every listener that never comes back down.
        """
        while self._paused.is_set() and not self._stop.is_set():
            time.sleep(0.05)

    # --- keep-alive -------------------------------------------------------

    def _keepalive_tick(self, now: float) -> bool:
        """Broadcast a second of silence if nothing has gone out recently.

        Split out of the thread loop so the guarantee is testable without
        sleeping. Returns True when frames were sent.
        """
        if now - self._last_audio_at < KEEPALIVE_GAP_S:
            # Something is playing: nothing is owed, so silence must never go
            # out on top of it. Anchor the next silence to that audio — with
            # max(), so that ticking inside the gap cannot walk the deadline
            # forward and halve the audio a stalled station sends.
            self._next_silence_at = max(
                self._next_silence_at, self._last_audio_at + KEEPALIVE_GAP_S)
            return False
        if now < self._next_silence_at:
            return False
        frames = self._silence_frames()
        if not frames:
            self._next_silence_at = now + KEEPALIVE_GAP_S
            if not self._keepalive_warned:
                self._keepalive_warned = True
                log.error("cannot keep the station on air: no silence frames to "
                          "send, so Icecast will time the source out and drop "
                          "the mount (external players then get no stream)")
            return False
        self._broadcast(frames)
        # Advance the deadline by exactly one gap, not from `now`: the audio a
        # stalled station sends has to be real time. Sending 1 s every 1.1 s
        # (a tick's jitter, each time) would slowly drain every listener's
        # cushion — the pause that ran for 32 h before the library was
        # unmounted would have leaked minutes of it.
        self._next_silence_at += KEEPALIVE_GAP_S
        if self._next_silence_at < now:  # fell behind: re-anchor, never burst
            self._next_silence_at = now + KEEPALIVE_GAP_S
        return True

    def _keepalive(self) -> None:
        """Never let the station go silent for longer than KEEPALIVE_GAP_S.

        The stream Icecast relays is what keeps the mount alive, and every
        stall in the broadcast loop used to leave it idle for long enough to
        trip ``source-timeout``: the backoff after a run of unplayable files (up
        to 30 s, measured against the real server — the pusher's socket was
        broken off exactly that long after the library went missing), the LLM
        writing a DJ break and piper speaking it (which is precisely the path a
        resuming listener waits on), an encoder starting up, a pause. When the
        source is dropped the mount disappears with it, so an external player
        connecting at that moment gets nothing and has to be restarted by hand
        once the source is back.

        Sending silence whenever nothing else is playing makes "there is always
        a stream on the mount" true for all of those at once.
        """
        while not self._stop.is_set():
            if self._stop.wait(KEEPALIVE_TICK_S):
                break
            try:
                self._keepalive_tick(time.monotonic())
            except Exception:  # never let the guarantee die with the thread
                log.debug("keep-alive failed", exc_info=True)

    def _note_true_duration(self, kind: str, meta: dict[str, Any], played: float,
                            returncode: int | None, reason: str) -> None:
        """Replace a track's guessed length with the one we just measured.

        The scanner stores mutagen's header estimate, which is wrong for VBR
        files without a Xing header — often by a minute or more. The UI's
        progress bar then reaches 100% while the song is still playing. A full
        decode measures the file exactly (bytes broadcast / bytes per second),
        so keep it: the scanner only rewrites `duration` for files whose size or
        mtime changed, so this survives a rescan.
        """
        if kind != "track" or reason != "eof" or returncode != 0 or played < 1.0:
            return
        track = meta.get("track") or {}
        track_id = track.get("id")
        if not track_id:
            return
        stored = track.get("duration")
        if stored and abs(played - float(stored)) <= max(3.0, 0.03 * float(stored)):
            return  # the header estimate was close enough
        try:
            db.set_duration(int(track_id), played)
            log.info("track %s measured at %.1fs (catalogue said %s)",
                     track_id, played, stored)
        except Exception:
            log.debug("could not store the measured duration", exc_info=True)

    def _note_unplayable(self, kind: str, meta: dict[str, Any]) -> None:
        """Take a track whose file is gone out of the playable pool.

        A missing file makes ``_play_file`` return instantly, so without this the
        broadcast loop consumes the whole queue in a fraction of a second,
        refills, and repeats — forever. Every refill also stamps new DJ talks,
        so the LLM gets hammered while the station plays nothing. Flagging the
        row (``missing = 1``) removes it from ``query_tracks`` so each dead file
        costs exactly one attempt; a rescan then deletes the row properly.
        """
        if kind != "track":
            return  # a missing DJ clip is a cache artifact, not a library row
        track_id = (meta.get("track") or {}).get("id")
        if not track_id:
            return
        try:
            db.mark_missing(int(track_id))
        except Exception:
            log.debug("could not flag missing track %s", track_id, exc_info=True)

    def _register_failure(self, path: str) -> float:
        """Count a consecutive playback failure; return how long to back off.

        Returns 0 while below the breaker threshold. Past it, the delay grows
        exponentially (capped), which stops a mass-missing library from
        spinning the CPU, spamming the log and re-triggering refills (and thus
        DJ-break LLM calls) thousands of times per minute.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures < FAILURE_BREAKER:
            return 0.0
        self._playback_error = (
            f"{self._consecutive_failures} queued files in a row could not be "
            f"played (last: {Path(path).name}) — is the music library mounted?"
        )
        log.error("playback stalled: %s", self._playback_error)
        return min(2.0 ** (self._consecutive_failures - FAILURE_BREAKER), 30.0)

    def _register_success(self) -> None:
        if self._consecutive_failures or self._playback_error:
            log.info("playback recovered after %d failed file(s)",
                     self._consecutive_failures)
        self._consecutive_failures = 0
        self._playback_error = None

    def _play_file(self, path: str, kind: str, meta: dict[str, Any]) -> bool:
        """Stream one file to all listeners, paced in real time."""
        if not Path(path).exists():
            log.warning("missing media file: %s", path)
            self._note_unplayable(kind, meta)
            return False

        self._skip.clear()
        # Music plays at its measured static gain; the DJ voice keeps the
        # dynamic chain (every clip comes from the same Piper model, so its
        # level is already consistent and dj.gain_db remains the trim).
        gain_db = None
        if kind == "track":
            gain_db = (meta.get("track") or {}).get("gain_db")
        args = self._encode_args(path, gain_db=gain_db)
        err_sink = _stderr_sink()
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if err_sink is None else err_sink,
                bufsize=0,
            )
        except OSError as exc:
            log.error("failed to launch ffmpeg: %s", exc)
            _stderr_text(err_sink, 0)  # closes the sink
            return False

        self._proc = proc
        with self._lock:  # one atomic switch-over for state()'s readers
            self._played_seconds = 0.0
            self._now_playing = {
                "kind": kind,
                "track": meta.get("track"),
                "dj_text": meta.get("dj_text"),
                "program": meta.get("program"),
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
        # Push the now-playing title to Icecast off-thread (best-effort): a
        # slow/unreachable Icecast admin API must never stall the broadcast
        # loop, which would underrun the pusher and can drop Winamp listeners.
        icecast_metadata.push_async(kind, meta)

        bps = self._bytes_per_second()
        sent = 0
        start = time.monotonic()
        aligner = FrameAligner()
        reason = "eof"
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
                    # Hold the decoder where it is (so resuming continues the
                    # same track). The air time is the keep-alive's job. The
                    # progress counter is deliberately NOT advanced here: the
                    # audio is not moving, so neither is the UI's bar.
                    self._pause_hold()
                    start = time.monotonic() - (sent / bps)

                self._broadcast(data)
                sent += len(data)
                with self._lock:
                    self._played_seconds = sent / bps
        except Exception:
            reason = "error"
            log.exception("error while streaming %s", path)
        finally:
            aligner.discard_leftover()  # never broadcast a partial frame
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
            text = _stderr_text(err_sink)  # also closes it
            if code is not None and code not in (0, -9) and text:
                log.warning("ffmpeg: %s", text)
            self._proc = None
        # A skipped or stopped item only got part way through, so its byte count
        # is not the file's length — only a natural end measures the whole file.
        if self._skip.is_set() or self._stop.is_set():
            reason = "cut"
        self._note_true_duration(kind, meta, sent / bps, proc.returncode, reason)
        # A deliberate skip is not a playback failure. Counting it as one means
        # five quick skips (each landing before the first chunk) look like an
        # unmounted library: the loop backs off and the UI shows "could not be
        # played — is the music library mounted?".
        if self._skip.is_set():
            return True
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
        self._skip.clear()  # a skip aimed at the previous item, not at silence
        with self._lock:  # one atomic switch-over for state()'s readers
            self._played_seconds = 0.0
            self._now_playing = {
                "kind": "idle", "track": None, "dj_text": None,
                "duration": 5.0,
            }
        self._notify_change()
        # Clear Icecast "now playing" while idle so players don't keep showing
        # the last song over silence. Off-thread so it can't stall the loop.
        icecast_metadata.push_async("idle", {})
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
        except OSError:
            time.sleep(5)
            return
        # Published like any other item, so skip()/stop() can kill the encoder
        # instead of waiting out the full 5 s of silence.
        self._proc = proc
        bps = self._bytes_per_second()
        sent, start = 0, time.monotonic()
        aligner = FrameAligner()
        while not self._stop.is_set() and not self._skip.is_set():
            raw = proc.stdout.read(CHUNK)
            if not raw:
                break
            data = aligner.feed(raw)
            if not data:
                continue
            while (sent / bps) - (time.monotonic() - start) > MAX_AHEAD_SECONDS:
                time.sleep(0.1)
                if self._stop.is_set() or self._skip.is_set():
                    break
            self._broadcast(data)
            sent += len(data)
            with self._lock:
                self._played_seconds = sent / bps
        aligner.discard_leftover()
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)  # reap the child instead of leaving a zombie
        except subprocess.TimeoutExpired:
            pass
        self._proc = None

    def _run(self) -> None:
        log.info("broadcaster started")
        while not self._stop.is_set():
            try:
                if self._paused.is_set():
                    # Nobody is listening: stay off the queue entirely, so no
                    # track is consumed and no DJ break is aired to an empty
                    # house, while the silence keeps the mount alive.
                    self._pause_hold()
                    continue
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
                    self._register_success()
                    try:
                        db.record_play(track.get("id"))
                    except Exception:
                        log.debug("could not record play", exc_info=True)
                else:
                    # Back off once files keep failing: a missing library used to
                    # let this loop consume thousands of queue items per second
                    # (refill → new DJ talks → LLM storm) while playing nothing.
                    backoff = self._register_failure(track["path"])
                    if backoff:
                        if self._stop.wait(backoff):
                            break
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
        self._last_audio_at = time.monotonic()
        self._next_silence_at = time.monotonic() + KEEPALIVE_GAP_S
        # The keep-alive starts first: from the very first moment the station is
        # up, silence stands in for whatever is not playing, so the Icecast
        # source never sees an idle socket and the mount is never missing.
        self._keepalive_warned = False
        self._keepalive_thread = threading.Thread(
            target=self._keepalive, name="vdj-keepalive", daemon=True)
        self._keepalive_thread.start()
        log.info("keep-alive started: silence whenever nothing plays, so the "
                 "Icecast source never idles out")
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
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=5)
            self._keepalive_thread = None

    # --- introspection ----------------------------------------------------

    def state(self) -> dict[str, Any]:
        # One lock for the whole snapshot: these fields are written by the
        # broadcast thread, and read separately they could disagree with each
        # other (the previous track paired with this one's elapsed time). The
        # lock is released before anything that calls out — the presence watcher
        # and the scheduler take their own locks.
        with self._lock:
            now = self._now_playing
            elapsed = round(self._played_seconds, 1)
            now_track = now.get("track") or {}
            spoken, spoken_for = self._last_dj_text, self._dj_for_track_id
        # Progress = audio actually broadcast for the current item, so it stands
        # still while the audio does (a pause, or a stalled decoder). It used to
        # be `time.time() - started_at`, which kept the UI's progress bar
        # crawling through a paused, silent song — the bar's clock and the audio
        # were never the same thing.
        # The DJ phrase shows only for the track it introduced. Once the next
        # track starts (the announced song has ended) it clears.
        show_dj = (spoken is not None
                   and spoken_for is not None
                   and now_track.get("id") == spoken_for
                   and now.get("kind") in ("dj", "track"))
        dj_text = spoken if show_dj else None
        return {
            "kind": now.get("kind"),
            "track": now.get("track"),
            "dj_text": dj_text,
            "elapsed": elapsed,
            "duration": now.get("duration"),
            "paused": self.paused,
            "listeners": self.listener_count(),
            # Who is listening and who paused the station (web + Icecast, with
            # "unknown" spelled out), so the UI can explain an auto-pause.
            "listeners_detail": (self._watcher.status() if self._watcher else None),
            # Set when queued files keep failing to play (library not mounted /
            # moved). The UI shows this as a warning instead of looking idle.
            "error": self._playback_error,
            "uptime": (round(time.time() - self._started_stream_at, 1)
                       if self._started_stream_at else None),
            "scheduler": SCHEDULER.status(),
            # Icecast delivery info so the web UI can point external players
            # (Winamp/VLC/Sonos) at the mount. Browser clients keep using the
            # app's own /stream.mp3.
            "icecast": {
                "enabled": bool(config.get("icecast.enabled", False)),
                "mount": config.get("icecast.mount", "virtualdj"),
                "public_port": int(config.get("icecast.public_port", 8008)),
                "public_host": config.get("icecast.public_host", "") or "",
            },
        }


BROADCASTER = Broadcaster()
