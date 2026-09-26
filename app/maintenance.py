"""Housekeeping: keep the app's temporary data bounded, automatically.

Every rendered DJ break, every TTS scratch file and every partially written
download used to live forever: the DJ audio cache grew without limit (the user
noticed it), and an interrupted voice download or a killed process could leave a
`.tmp` file behind with nothing to collect it.

What this module guarantees:

* the **DJ audio cache** (`cache/dj/*.mp3`, one file per spoken line) is pruned to
  the newest ``cache.dj_keep_files`` entries, anything older than
  ``cache.dj_max_age_hours``, and an overall ``cache.dj_max_mb`` budget — while
  never deleting a file that the queue is about to play (`protected_names()`);
* **stale partial writes** (`*.tmp`, `*.part`, `*.partial`, `*.download`, `*.wav`)
  older than ``cache.tmp_grace_minutes`` are removed from the data/cache/voices
  directories — a grace period keeps an in-flight download safe;
* voice models themselves are NEVER touched (they are downloads, not cache).

The sweep runs on startup and then every ``cache.cleanup_interval_minutes`` on a
daemon thread, and on demand from the API (`POST /api/cache/clean`).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import config

log = logging.getLogger("virtual_dj.maintenance")

# Partial-write leftovers. Matching on the suffix keeps this independent of the
# producer: voices.py writes *.part-style temporaries, config.py writes *.tmp.
STALE_SUFFIXES = (".tmp", ".part", ".partial", ".download", ".wav")


def _files(directory: Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    if not directory.exists():
        return []
    try:
        entries = [p for p in directory.iterdir() if p.is_file()]
    except OSError:
        return []
    if suffixes is not None:
        entries = [p for p in entries if p.suffix.lower() in suffixes]
    return entries


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def protected_names() -> set[str]:
    """Audio cache files the queue is about to play — never prune these.

    A prepared break lives in the scheduler's map until its track starts; a break
    whose file disappeared mid-queue would silence the DJ (or worse, make the
    broadcaster skip the talk), so protection is read from the live queue.
    """
    names: set[str] = set()
    try:
        from .scheduler import SCHEDULER  # local import: avoid an import cycle

        for prepared in SCHEDULER.prepared_breaks():
            path = prepared.get("audio_path")
            if path:
                names.add(Path(path).name)
    except Exception as exc:  # noqa: BLE001 - housekeeping must never break playback
        log.debug("could not read prepared breaks (%s)", exc)
    return names


def cache_stats() -> dict[str, Any]:
    """Current footprint of everything the app may prune."""
    dj = _files(config.DJ_CACHE_DIR, (".mp3",))
    stale = _stale_files()
    return {
        "dj_files": len(dj),
        "dj_bytes": sum(_size(p) for p in dj),
        "dj_oldest_hours": round((time.time() - min((_mtime(p) for p in dj), default=time.time())) / 3600, 1),
        "stale_files": len(stale),
        "stale_bytes": sum(_size(p) for p in stale),
        "voices_bytes": sum(_size(p) for p in _files(config.VOICES_DIR)),
        "enabled": bool(config.get("cache.enabled", True)),
        "interval_minutes": int(config.get("cache.cleanup_interval_minutes", 10) or 10),
        "keep_files": int(config.get("cache.dj_keep_files", 100) or 0),
        "max_age_hours": float(config.get("cache.dj_max_age_hours", 48) or 0),
        "max_mb": float(config.get("cache.dj_max_mb", 256) or 0),
        "last_run": _LAST.get("at"),
        "last_removed_files": _LAST.get("removed_files", 0),
        "last_removed_bytes": _LAST.get("removed_bytes", 0),
    }


def _stale_files() -> list[Path]:
    grace = float(config.get("cache.tmp_grace_minutes", 60) or 0) * 60
    cutoff = time.time() - grace
    found: list[Path] = []
    for directory in (config.DATA_DIR, config.CACHE_DIR, config.VOICES_DIR):
        for path in _files(directory, STALE_SUFFIXES):
            if _mtime(path) < cutoff:
                found.append(path)
    return found


_LAST: dict[str, Any] = {}


def clean(force: bool = False) -> dict[str, Any]:
    """Prune the cache and stale partial writes. Returns a report.

    ``force`` sweeps even when ``cache.enabled`` is false (the API's "Clean now"
    button) — the button is an explicit instruction, not a policy default.
    """
    if not force and not config.get("cache.enabled", True):
        return {"skipped": "cache cleanup disabled", "removed_files": 0, "removed_bytes": 0}

    config.ensure_dirs()
    protect = protected_names()
    removed: list[str] = []
    freed = 0

    def drop(path: Path, why: str) -> None:
        nonlocal freed
        try:
            size = _size(path)
            path.unlink()
        except OSError as exc:
            log.debug("could not remove %s (%s): %s", path, why, exc)
            return
        removed.append(f"{path.name} ({why})")
        freed += size

    # 1. stale partial writes (never a live download: they carry a grace period)
    for path in _stale_files():
        drop(path, "stale temporary file")

    # 2. the DJ audio cache, newest-first, respecting the queue
    keep_count = int(config.get("cache.dj_keep_files", 100) or 0)
    max_age_h = float(config.get("cache.dj_max_age_hours", 48) or 0)
    max_bytes = float(config.get("cache.dj_max_mb", 256) or 0) * 1024 * 1024
    now = time.time()

    entries = sorted(
        (p for p in _files(config.DJ_CACHE_DIR, (".mp3",)) if p.name not in protect),
        key=_mtime,
        reverse=True,
    )
    kept_bytes = 0
    for index, path in enumerate(entries):
        age_h = (now - _mtime(path)) / 3600
        over_count = keep_count and index >= keep_count
        too_old = max_age_h and age_h > max_age_h
        over_size = max_bytes and kept_bytes + _size(path) > max_bytes
        if over_count or too_old or over_size:
            # Protected files were filtered out above; anything here is safe.
            drop(path, "dj cache pruned" if not too_old else "dj cache expired")
        else:
            kept_bytes += _size(path)

    report = {
        "removed_files": len(removed),
        "removed_bytes": freed,
        "removed": removed[:50],
        "protected": sorted(protect),
        "kept_files": len([p for p in _files(config.DJ_CACHE_DIR, (".mp3",))]),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _LAST.update(report)
    if removed:
        log.info("cache cleanup: removed %d file(s), %.1f MB freed",
                 len(removed), freed / 1024 / 1024)
    return report


class Maintainer:
    """Daemon thread that sweeps on startup and then on an interval."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="housekeeping", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            clean()
        except Exception as exc:  # noqa: BLE001
            log.debug("startup cache cleanup failed: %s", exc)
        while not self._stop.is_set():
            minutes = int(config.get("cache.cleanup_interval_minutes", 10) or 10)
            # Re-read the interval each tick so a config change (or a disabled
            # cache) takes effect without a restart.
            if self._stop.wait(max(60, minutes * 60)):
                return
            try:
                clean()
            except Exception as exc:  # noqa: BLE001
                log.debug("periodic cache cleanup failed: %s", exc)

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        self._thread = None


MAINTAINER = Maintainer()
