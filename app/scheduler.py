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
        # Tracks remaining until the next DJ break. When it reaches 0 the next
        # wrapped-track is flagged for a talk, and a fresh random interval is
        # rolled from dj.talk_min..talk_max. Stamping the decision at enqueue
        # time (under the lock) keeps the consumer and the prefetch worker in
        # perfect agreement, so we never double-talk or skip a gap. None means
        # "not yet initialized" — set on first wrap.
        self._tracks_until_talk: int | None = None
        self._previous: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._wake = threading.Event()

    # --- queue management -------------------------------------------------

    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _roll_interval(self) -> int:
        """Pick a random number of tracks before the next talk (0 = never)."""
        return config.randint_range(
            "dj.talk_min", "dj.talk_max",
            config.DEFAULTS["dj"]["talk_min"], config.DEFAULTS["dj"]["talk_max"])

    def _schedule_next(self) -> None:
        """Begin a fresh countdown to the next talk from config."""
        self._tracks_until_talk = self._roll_interval()

    def _wrap(self, track: dict[str, Any], with_dj: bool | None = None,
              program: dict[str, Any] | None = None) -> dict[str, Any]:
        # Decide whether THIS track gets a DJ break. The decision is stamped
        # here, at enqueue time, so the consumer and prefetch worker agree.
        if with_dj is None:
            with_dj = False
            if config.get("dj.enabled", True):
                if self._tracks_until_talk is None:
                    # First track is never a talk; begin the countdown.
                    self._schedule_next()
                elif self._tracks_until_talk:
                    self._tracks_until_talk -= 1
                    if self._tracks_until_talk == 0:
                        with_dj = True
                        self._schedule_next()
        else:
            # Explicit decision (e.g. a forced program-start talk). A real talk
            # here starts a fresh random interval for whatever follows.
            if with_dj and config.get("dj.enabled", True):
                self._schedule_next()
        return {
            "uid": self._next_uid(),
            "track": track,
            "dj_requested": with_dj,
            "program": program,
        }

    def _build_programs(self, count: int) -> list[dict[str, Any]]:
        """Build ``count`` themed program items, ordering the queue into runs.

        Each program is a contiguous block of tracks sharing a theme (genre,
        artist, or decade, per ``playback.program.strategy``). The first track
        of every program after the first carries a ``program`` theme and
        ``dj_requested=True`` so the DJ announces the vibe switch before it.
        """
        playback = config.get("playback", {}) or {}
        prog = playback.get("program", {}) or {}
        size = max(2, int(prog.get("size", 6)))
        strategy = str(prog.get("strategy", "genre"))
        search = playback.get("search", "") or ""
        genres_filter = playback.get("genres") or None
        artists_filter = playback.get("artists") or None

        themes = library.program_themes(strategy)
        themes = [t for t in themes if t.get("n", 0) >= size]
        # NOTE: the global genre/artist filters do NOT pre-filter this theme
        # list. A theme carries only its own dimension (genre themes have no
        # `artist` key, etc.), so filtering the list by the wrong dimension
        # empties it and silently falls back to a flat shuffle. Instead we
        # carry the filters into each theme's track query below, where they
        # are AND-ed correctly.
        if not themes:
            return []

        random.shuffle(themes)
        items: list[dict[str, Any]] = []
        programs_made = 0
        # Round-robin themes so consecutive programs differ, like a real DJ
        # alternating vibes rather than repeating one.
        for theme in themes:
            if programs_made >= count:
                break
            if strategy == "genre":
                kwargs = {"genres": [theme["genre"]], "search": search}
            elif strategy == "artist":
                kwargs = {"artists": [theme["artist"]], "search": search}
            elif strategy == "language":
                kwargs = {"languages": [theme["language"]], "search": search}
            else:  # decade
                kwargs = {"decade": int(theme["decade"]), "search": search}
            # Apply global genre/artist filters to the track pool for this
            # theme. AND-ing here means e.g. "Artist" theme + genre filter
            # yields that artist's tracks in that genre (skipped if none).
            if genres_filter:
                kwargs["genres"] = (kwargs.get("genres") or []) + list(genres_filter)
            if artists_filter:
                kwargs["artists"] = (kwargs.get("artists") or []) + list(artists_filter)
            tracks = library.query_tracks(limit=size, random_order=True, **kwargs)
            if len(tracks) < 2:
                continue
            program = {
                "kind": strategy,
                "label": theme.get("genre") or theme.get("artist")
                or theme.get("language") or f"{theme['decade']}s",
            }
            first = True
            for track in tracks:
                if first and items:
                    # Announce the switch into this new program.
                    items.append(self._wrap(
                        track, with_dj=True, program=program))
                else:
                    items.append(self._wrap(track, with_dj=None, program=program))
                first = False
            programs_made += 1
        return items

    def refill(self, count: int = 20) -> int:
        """Top the queue up from the library using the active filters.

        When ``playback.program.enabled`` the queue is filled in themed runs
        (programs) with a DJ break announcing each vibe switch; otherwise it is
        a flat shuffle.
        """
        playback = config.get("playback", {}) or {}
        program_enabled = bool((playback.get("program") or {}).get("enabled", False))
        if program_enabled:
            size = max(2, int((playback.get("program") or {}).get("size", 6)))
            n_programs = max(1, count // size)
            items = self._build_programs(n_programs)
            if items:
                with self._lock:
                    self._queue.extend(items)
                self._wake.set()
                return len(items)
            # No theme had enough tracks (tiny library) — fall through to flat.

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
                    "program": item.get("program"),
                    "dj_requested": bool(item.get("dj_requested")),
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
        """Should a DJ break precede this item?

        The decision is stamped onto each item at enqueue time (see
        ``_wrap``), so this just reads it. ``index`` is accepted for call
        compatibility but the stored decision is authoritative — both the
        consumer and the prefetch worker see the same flag.
        """
        return bool(item.get("dj_requested"))

    def pop_next(self) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None] | None:
        """Return (track, dj_break_or_None, program_or_None) for the next thing to play."""
        self.ensure_filled(5)
        with self._lock:
            if not self._queue:
                return None
            item = self._queue.pop(0)
            due = self._dj_due(item, 0)
            prepared = self._prepared.pop(item["uid"], None)
            self._track_counter += 1
            self._previous = item["track"]
            program = item.get("program")
        self._wake.set()
        if due and prepared is None:
            # Look-ahead missed this one (fresh start, slow LLM). Prepare it
            # inline only if it is cheap; otherwise skip the break.
            prepared = None
        return item["track"], (prepared if due else None), program

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
            previous = self._previous

        for index, item in enumerate(upcoming):
            if self._stop.is_set():
                return
            uid = item["uid"]
            with self._lock:
                if uid in self._prepared:
                    continue
            # The talk decision is already stamped on the item at enqueue time
            # (see _wrap); just read it. A forced program-start (dj_requested
            # explicitly True) or a rolled-interval hit both count.
            explicit = item.get("dj_requested")
            if not explicit:
                continue
            prior = upcoming[index - 1]["track"] if index else previous
            started = time.monotonic()
            prepared = dj.prepare_break(
                item["track"], prior, program=item.get("program"))
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
