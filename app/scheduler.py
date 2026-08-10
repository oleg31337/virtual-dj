"""Playlist scheduling and look-ahead preparation of DJ breaks.

The scheduler owns the upcoming queue. A background worker prepares (LLM
script + Piper audio) the DJ break for tracks *before* the stream reaches
them, so a break is always ready on time and never stalls playback.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from . import config, dj, library

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []
        self._prepared: dict[int, dict[str, Any]] = {}   # queue item uid -> break
        self._uid = 0
        self._track_counter = 0
        self._previous: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._wake = threading.Event()

    # --- queue management -------------------------------------------------

    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _wrap(self, track: dict[str, Any], with_dj: bool | None = None) -> dict[str, Any]:
        return {
            "uid": self._next_uid(),
            "track": track,
            "dj_requested": with_dj,
        }

    def refill(self, count: int = 20) -> int:
        """Top the queue up from the library using the active filters."""
        playback = config.get("playback", {}) or {}
        tracks = library.query_tracks(
            search=playback.get("search", "") or "",
            genres=playback.get("genres") or None,
            artists=playback.get("artists") or None,
            limit=count,
            random_order=bool(playback.get("shuffle", True)),
        )
        if not tracks:
            # Filters matched nothing — fall back to the whole library so the
            # stream never goes silent.
            tracks = library.query_tracks(limit=count, random_order=True)
        if not tracks:
            return 0
        if not playback.get("shuffle", True):
            existing_paths = {i["track"]["path"] for i in self._queue}
            tracks = [t for t in tracks if t["path"] not in existing_paths]
        else:
            random.shuffle(tracks)
        with self._lock:
            for track in tracks:
                self._queue.append(self._wrap(track))
        self._wake.set()
        return len(tracks)

    def ensure_filled(self, minimum: int = 5) -> None:
        with self._lock:
            need = minimum - len(self._queue)
        if need > 0:
            self.refill(max(need, 10))

    def peek(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = self._queue[:limit]
            return [
                {
                    "uid": item["uid"],
                    "track": item["track"],
                    "dj_ready": item["uid"] in self._prepared,
                    "dj_text": (self._prepared.get(item["uid"]) or {}).get("text"),
                }
                for item in items
            ]

    def enqueue_track_ids(self, track_ids: list[int], position: str = "end") -> int:
        added = 0
        with self._lock:
            for track_id in track_ids:
                track = library.get_track(int(track_id))
                if not track:
                    continue
                item = self._wrap(track)
                if position == "next":
                    self._queue.insert(0, item)
                else:
                    self._queue.append(item)
                added += 1
        self._wake.set()
        return added

    def remove(self, uid: int) -> bool:
        with self._lock:
            for index, item in enumerate(self._queue):
                if item["uid"] == uid:
                    self._queue.pop(index)
                    self._prepared.pop(uid, None)
                    return True
        return False

    def move(self, uid: int, new_index: int) -> bool:
        with self._lock:
            for index, item in enumerate(self._queue):
                if item["uid"] == uid:
                    self._queue.pop(index)
                    self._queue.insert(max(0, min(new_index, len(self._queue))), item)
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._prepared.clear()

    def replace(self, track_ids: list[int]) -> int:
        self.clear()
        return self.enqueue_track_ids(track_ids)

    # --- consumption ------------------------------------------------------

    def _dj_due(self, item: dict[str, Any], index: int) -> bool:
        """Should a DJ break precede this item?"""
        if item.get("dj_requested") is not None:
            return bool(item["dj_requested"])
        if not config.get("dj.enabled", True):
            return False
        every = int(config.get("dj.every_n_tracks", 3) or 0)
        if every <= 0:
            return False
        return (self._track_counter + index) % every == 0

    def pop_next(self) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        """Return (track, dj_break_or_None) for the next thing to play."""
        self.ensure_filled(5)
        with self._lock:
            if not self._queue:
                return None
            item = self._queue.pop(0)
            due = self._dj_due(item, 0)
            prepared = self._prepared.pop(item["uid"], None)
            self._track_counter += 1
            self._previous = item["track"]
        self._wake.set()
        if due and prepared is None:
            # Look-ahead missed this one (fresh start, slow LLM). Prepare it
            # inline only if it is cheap; otherwise skip the break.
            prepared = None
        return item["track"], (prepared if due else None)

    def previous_track(self) -> dict[str, Any] | None:
        with self._lock:
            return self._previous

    # --- look-ahead worker ------------------------------------------------

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._prefetch_loop, name="vdj-prefetch", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _prefetch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._prefetch_once()
            except Exception:
                log.exception("prefetch iteration failed")
            self._wake.wait(timeout=5.0)
            self._wake.clear()

    def _prefetch_once(self) -> None:
        if not config.get("dj.enabled", True):
            return
        self.ensure_filled(5)
        depth = max(1, int(config.get("dj.prefetch_depth", 3)))
        with self._lock:
            upcoming = list(self._queue[:depth])
            counter = self._track_counter
            previous = self._previous

        for index, item in enumerate(upcoming):
            if self._stop.is_set():
                return
            uid = item["uid"]
            with self._lock:
                if uid in self._prepared:
                    continue
            explicit = item.get("dj_requested")
            every = int(config.get("dj.every_n_tracks", 3) or 0)
            due = bool(explicit) if explicit is not None else (
                every > 0 and (counter + index) % every == 0
            )
            if not due:
                continue
            prior = upcoming[index - 1]["track"] if index else previous
            started = time.monotonic()
            prepared = dj.prepare_break(item["track"], prior)
            if prepared:
                with self._lock:
                    self._prepared[uid] = prepared
                log.info(
                    "prepared DJ break for %s - %s in %.1fs (%.1fs audio)",
                    item["track"].get("artist"), item["track"].get("title"),
                    time.monotonic() - started, prepared.get("duration", 0.0),
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "prepared_breaks": len(self._prepared),
                "tracks_played": self._track_counter,
            }


SCHEDULER = Scheduler()
