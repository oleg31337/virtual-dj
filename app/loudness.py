"""Per-track loudness measurement and static normalization (EBU R128).

Why: a music library spans a wide loudness range (measured here: 17 dB between
the quietest and loudest file). Playing that as a continuous stream makes loud
tracks stand out and quiet ones disappear. The fix is the classic
ReplayGain approach — measure each track ONCE, then play it with a fixed gain:

    gain_db = target_lufs - integrated_lufs         (e.g. -16 - (-20.1) = +4.1)
    gain_db = min(gain_db, true_peak_ceiling - true_peak_dbfs)   # never clip
    gain_db = clamp(gain_db, min_gain_db, max_boost_db)          # never shout

That is transparent (no per-second riding of the level inside a song, no
start-of-track ramp) and deterministic (a track sounds the same every play).

The scan does NOT measure tracks inline — it is already network-bound. Instead
this module owns a small background worker pool that drains "unmeasured tracks"
(a DB query, so it is resumable across restarts) at low CPU priority while the
station keeps streaming. A track that has not been measured yet simply keeps
the legacy dynamic ``loudnorm`` chain, so the change is gradual and invisible.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import config, db

log = logging.getLogger(__name__)

# Bump when the measurement itself changes materially. Every row stores the
# algo id it was measured with; a mismatch re-queues the row automatically, so
# changing the analyzed window (which is part of the id) re-measures the
# library instead of silently mixing two different estimates.
ALGO = "ebur128-v1"

# An analysis is bounded: generous, but a wedged ffmpeg must never occupy a
# worker forever. Scaled per minute of analyzed audio, plus a fixed overhead.
TIMEOUT_BASE_S = 30.0
TIMEOUT_PER_MINUTE_S = 20.0
TIMEOUT_MAX_S = 1800.0

# EBU R128 gates silence at -70 LUFS; below that a "measurement" is really
# "this file is silent" and must not be turned into a maximum boost.
_SILENCE_FLOOR_LUFS = -70.0


def _clamped(key: str, default: float, lo: float, hi: float) -> float:
    """Read a numeric knob, tolerating junk values in a hand-edited config."""
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(lo, min(value, hi))


def settings() -> dict[str, Any]:
    """Resolved + clamped loudness settings (a mis-set config can never raise)."""
    return {
        "enabled": bool(config.get("loudness.enabled", True)),
        "target_lufs": _clamped("loudness.target_lufs", -16.0, -30.0, -6.0),
        "true_peak_ceiling": _clamped("loudness.true_peak_ceiling", -1.5, -6.0, 0.0),
        "max_boost_db": _clamped("loudness.max_boost_db", 6.0, 0.0, 18.0),
        "min_gain_db": _clamped("loudness.min_gain_db", -12.0, -30.0, 0.0),
        "window_seconds": int(_clamped("loudness.window_seconds", 120, 0, 900)),
        "workers": int(_clamped("loudness.workers", 6, 1, 16)),
        "autostart": bool(config.get("loudness.autostart", True)),
    }


def algo_for(window_seconds: int) -> str:
    """Measurement id stored per row (the analyzed window is part of it)."""
    return f"{ALGO}-w{int(window_seconds)}"


def gain_for(
    lufs_i: float | None,
    true_peak_dbfs: float | None = None,
    *,
    target_lufs: float | None = None,
    true_peak_ceiling: float | None = None,
    max_boost_db: float | None = None,
    min_gain_db: float | None = None,
) -> float | None:
    """Static gain in dB for a measured track (None when it cannot be used).

    Pure math, no I/O — the whole normalization policy lives here so it can be
    unit-tested exhaustively.
    """
    if lufs_i is None:
        return None
    try:
        loudness = float(lufs_i)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(loudness) or loudness < _SILENCE_FLOOR_LUFS:
        return None

    cfg = settings()
    target = cfg["target_lufs"] if target_lufs is None else float(target_lufs)
    ceiling = (cfg["true_peak_ceiling"] if true_peak_ceiling is None
               else float(true_peak_ceiling))
    max_boost = cfg["max_boost_db"] if max_boost_db is None else float(max_boost_db)
    min_gain = cfg["min_gain_db"] if min_gain_db is None else float(min_gain_db)

    gain = target - loudness
    # A track that already peaks near/over 0 dBFS cannot take the full cut of
    # headroom: cap the gain so the post-gain true peak stays under the ceiling.
    if true_peak_dbfs is not None:
        try:
            peak = float(true_peak_dbfs)
        except (TypeError, ValueError):
            peak = None
        if peak is not None and math.isfinite(peak):
            gain = min(gain, ceiling - peak)
    return round(max(min_gain, min(gain, max_boost)), 2)


def measure(path: str | Path, window_seconds: int | None = None,
            timeout_s: float | None = None) -> dict[str, Any] | None:
    """Measure the FIRST ``window_seconds`` of ``path`` (0 = whole file).

    Returns ``{"lufs_i", "true_peak_dbfs", "seconds"}`` or ``None`` when the
    file cannot be decoded / is silent. Never raises: an unmeasurable file must
    not break a scan or a worker thread.
    """
    win = (settings()["window_seconds"] if window_seconds is None
           else max(0, int(window_seconds)))
    if timeout_s is None:
        minutes = max(1.0, (win or 300) / 60.0)
        timeout_s = min(TIMEOUT_MAX_S, TIMEOUT_BASE_S + minutes * TIMEOUT_PER_MINUTE_S)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin"]
    if win > 0:
        # As an INPUT option, -t stops ffmpeg after reading that much audio —
        # the head of a track is the cheapest representative sample.
        cmd += ["-t", str(win)]
    cmd += ["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s,
            # Real libraries contain names whose bytes are not valid UTF-8;
            # a strict decode here would kill the whole run.
            text=True, errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("loudness measure failed for %s: %s", path, exc)
        return None

    lufs: float | None = None
    peak: float | None = None
    for line in (proc.stderr or "").splitlines():
        text = line.strip()
        # The final summary block prints the totals; the periodic lines above
        # it are running values, so last-wins is the integrated result.
        if text.startswith("I:") and "LUFS" in text:
            try:
                lufs = float(text.split()[1])
            except (IndexError, ValueError):
                continue
        elif text.startswith("Peak:") and "dBFS" in text:
            try:
                peak = float(text.split()[1])
            except (IndexError, ValueError):
                continue

    if lufs is None or not math.isfinite(lufs) or lufs < _SILENCE_FLOOR_LUFS:
        return None
    if peak is not None and not math.isfinite(peak):
        peak = None
    return {"lufs_i": lufs, "true_peak_dbfs": peak, "seconds": win}


class Analyzer:
    """Background worker pool that drains the "not measured yet" queue.

    The queue is a DB predicate, so the run is resumable: restarting the app
    mid-pass simply continues where it stopped. Results are applied per track
    as they land, so normalization improves gradually while the station plays.
    """

    def __init__(self) -> None:
        # RLock: status() is called from API threads while start()/workers hold it.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._claimed: set[int] = set()

        self.running = False
        self.workers = 0
        self.total = 0
        self.analyzed = 0
        self.failed = 0
        self.missing = 0
        self.current = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None

    # --- queue ------------------------------------------------------------

    def queue_size(self) -> int:
        algo = algo_for(settings()["window_seconds"])
        row = db.connect().execute(
            "SELECT COUNT(*) AS n FROM tracks WHERE excluded = 0 "
            "AND (loudness_analyzed_at IS NULL OR IFNULL(loudness_algo, '') <> ?)",
            (algo,),
        ).fetchone()
        return int(row["n"] or 0)

    def _claim(self) -> dict[str, Any] | None:
        """Atomically reserve the next unmeasured track for this worker."""
        algo = algo_for(settings()["window_seconds"])
        with self._lock:
            rows = db.connect().execute(
                "SELECT id, path FROM tracks WHERE excluded = 0 "
                "AND (loudness_analyzed_at IS NULL "
                "OR IFNULL(loudness_algo, '') <> ?) "
                "ORDER BY id LIMIT 64",
                (algo,),
            ).fetchall()
            for row in rows:
                if row["id"] not in self._claimed:
                    self._claimed.add(row["id"])
                    return {"id": row["id"], "path": row["path"]}
        return None

    def _store(self, track_id: int, result: dict[str, Any] | None) -> None:
        """Persist a measurement (or the fact that the file is unmeasurable).

        A failed analysis is stamped too, with a NULL gain: the track then keeps
        the dynamic loudnorm chain instead of being retried on every pass.
        """
        win = settings()["window_seconds"]
        gain = None if result is None else gain_for(result["lufs_i"],
                                                    result.get("true_peak_dbfs"))
        conn = db.connect()
        conn.execute(
            "UPDATE tracks SET lufs_i = ?, true_peak_dbfs = ?, gain_db = ?, "
            "loudness_analyzed_at = ?, loudness_algo = ? WHERE id = ?",
            (
                None if result is None else result["lufs_i"],
                None if result is None else result.get("true_peak_dbfs"),
                gain,
                time.time(),
                algo_for(win),
                track_id,
            ),
        )
        # Commit per track: an open write transaction across a long ffmpeg call
        # would block the app's other writers ("database is locked").
        conn.commit()

    # --- workers ----------------------------------------------------------

    def _worker(self) -> None:
        try:
            # Linux keeps nice per-thread, and the ffmpeg child inherits it, so
            # the broadcast never competes with the analyzer.
            os.nice(10)
        except (AttributeError, OSError):
            pass

        empty = 0
        while not self._stop.is_set():
            claimed = self._claim()
            if claimed is None:
                # Nothing claimable right now. Either the other workers hold
                # every pending row, or the queue is genuinely empty — one
                # short re-check absorbs a scan that adds rows while we wind
                # down, then the worker exits.
                empty += 1
                if empty >= 2 or self.queue_size() == 0:
                    break
                time.sleep(1.0)
                continue
            empty = 0

            path = Path(claimed["path"])
            result = None
            vanished = False
            try:
                if path.exists():
                    result = measure(path)
                else:
                    # A row whose file is gone (a stale index — the next scan
                    # deletes those rows). Stamped anyway so the queue can
                    # drain and the pass does not re-stat the whole NFS mount on
                    # every run; counted apart from real analysis failures.
                    vanished = True
                    log.debug("loudness: file is gone: %s", path)
            except Exception as exc:  # noqa: BLE001 - one file must not kill the pool
                log.warning("loudness analysis failed for %s: %s", path, exc)

            try:
                self._store(claimed["id"], result)
            except Exception as exc:  # noqa: BLE001 - a DB hiccup must not kill the pool
                log.warning("loudness store failed for %s: %s", path, exc)

            with self._lock:
                self.current = path.name
                if vanished:
                    self.missing += 1
                elif result is None:
                    self.failed += 1
                else:
                    self.analyzed += 1

    def _run_worker(self) -> None:
        try:
            self._worker()
        except Exception as exc:  # noqa: BLE001 - surface it in status, keep others alive
            self.error = str(exc)
            log.exception("loudness worker crashed")
        finally:
            me = threading.current_thread()
            with self._lock:
                self._threads = [t for t in self._threads if t is not me]
                if not self._threads:
                    self.running = False
                    self.finished_at = time.time()

    # --- control ----------------------------------------------------------

    def start(self, workers: int | None = None) -> dict[str, Any]:
        """Start (or resume) the background pass. Idempotent."""
        with self._lock:
            if self.running:
                return self.status()
            cfg = settings()
            if not cfg["enabled"]:
                self.error = "loudness normalization is disabled in the config"
                return self.status()
            count = max(1, min(int(workers or cfg["workers"]), 16))
            self._stop.clear()
            self._claimed.clear()
            self.workers = count
            self.analyzed = self.failed = self.missing = 0
            self.current = ""
            self.error = None
            self.total = self.queue_size()
            self.started_at = time.time()
            self.finished_at = None
            if self.total == 0:
                self.running = False
                self.finished_at = self.started_at
                return self.status()
            self.running = True
            for index in range(count):
                thread = threading.Thread(target=self._run_worker, daemon=True,
                                          name=f"loudness-{index}")
                self._threads.append(thread)
                thread.start()
            log.info("loudness analyzer started: %d workers, %d tracks queued",
                     count, self.total)
            return self.status()

    def stop(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """Ask the workers to finish the current file and stop.

        The budget is shared across workers (not per worker), so shutdown stays
        bounded: an app stop must not wait N x timeout. Workers are daemon
        threads, so a long ffmpeg analysis can never hold up process exit.
        """
        self._stop.set()
        deadline = time.time() + timeout_s
        for thread in list(self._threads):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        with self._lock:
            self.running = False
            self._threads = []
            self.finished_at = self.finished_at or time.time()
        return self.status()

    def reset(self) -> int:
        """Forget every measurement (forces a full re-analysis). Returns rows."""
        conn = db.connect()
        cur = conn.execute(
            "UPDATE tracks SET lufs_i = NULL, true_peak_dbfs = NULL, "
            "gain_db = NULL, loudness_analyzed_at = NULL, loudness_algo = NULL"
        )
        conn.commit()
        self.total = self.analyzed = self.failed = self.missing = 0
        self.finished_at = self.started_at = None
        return cur.rowcount or 0

    def status(self) -> dict[str, Any]:
        cfg = settings()
        with self._lock:
            done = self.analyzed + self.failed + self.missing
            elapsed = 0.0
            if self.started_at:
                elapsed = (self.finished_at or time.time()) - self.started_at
            rate = done / elapsed if elapsed > 0.5 and done else 0.0
            if self.running and rate > 0:
                remaining = max(0, self.total - done)
                eta: float | None = remaining / rate
            else:
                eta = None
            return {
                "running": self.running,
                "enabled": cfg["enabled"],
                "workers": self.workers or cfg["workers"],
                "total": self.total,
                "analyzed": self.analyzed,
                "failed": self.failed,
                "missing": self.missing,
                "done": done,
                "current": self.current,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "eta_seconds": eta,
                "error": self.error,
                "queue": self.queue_size(),
                "settings": cfg,
            }


ANALYZER = Analyzer()


def analyze_in_background(workers: int | None = None) -> dict[str, Any]:
    """Start the background pass (used by the API + scan completion)."""
    return ANALYZER.start(workers)


def autostart() -> None:
    """Kick the analyzer when configured to run automatically (best-effort)."""
    try:
        if settings()["enabled"] and settings()["autostart"]:
            ANALYZER.start()
    except Exception as exc:  # noqa: BLE001 - analysis is never fatal
        log.warning("loudness autostart skipped (%s)", exc)
