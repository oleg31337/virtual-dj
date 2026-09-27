"""Playlist scheduling and look-ahead preparation of DJ breaks.

The scheduler owns the upcoming queue. A background worker prepares (LLM
script + Piper audio) the DJ break for tracks *before* the stream reaches
them, so a break is always ready on time and never stalls playback.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any

from . import config, dj, library

log = logging.getLogger(__name__)

# How many times the prefetch worker tries to prepare one DJ break before
# giving up on it. Each attempt is an LLM call (+Piper), and the loop re-runs
# every 5 s and on every refill — so an unbounded retry means the model is hit
# every few seconds forever when preparation keeps failing.
MAX_BREAK_ATTEMPTS = 3
# The repeat window never shrinks below this while the library has anything to
# play: repeating a song after 2 others is the minimum acceptable quality bar.
REPEAT_WINDOW_FLOOR = 2


def _genre_matches(theme_genre: Any, term: Any) -> bool:
    """Does a theme's genre satisfy one entry of the Genres-card filter?

    Mirrors ``query_tracks``'s lenient matching (``genre LIKE '%term%'``, plus
    the ``Unknown`` = genre-less convention) so narrowing the theme list and
    querying the tracks can never disagree.
    """
    genre = str(theme_genre if theme_genre not in (None, "") else "Unknown")
    wanted = str(term or "").strip()
    if not wanted:
        return False
    if wanted == "Unknown":
        return genre == "Unknown"
    return wanted.lower() in genre.lower()


def _artist_run_settings() -> int:
    """How many songs in a row one artist may play (>= 1)."""
    return library.artist_run_cap()


class Scheduler:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []
        self._prepared: dict[int, dict[str, Any]] = {}   # queue item uid -> break
        self._attempts: dict[int, int] = {}              # uid -> failed attempts
        # How often the artist-spacing rule had to give way in the queue that is
        # currently waiting (a one-artist theme cannot honour it). Reset whenever
        # the queue is rebuilt, exposed by /api/programs so the card can say so.
        self._gap_breaks = 0
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
              program: dict[str, Any] | None = None,
              program_start: bool = False) -> dict[str, Any]:
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
            # Whether this track OPENS its program. `dj_requested` cannot say:
            # it is also set by the periodic talk cadence, which is why a
            # playlist audit that keyed program boundaries off it saw phantom
            # 2-4 song "programs" (the DJ talks every 2-4 tracks).
            "program_start": program_start,
        }

    def _do_not_repeat(self, window: int | None = None) -> set[int]:
        """Track ids the next songs must avoid.

        The last ``window`` plays (history, default:
        ``playback.repeat_window``) plus everything already queued — the queued
        part also keeps one refill from adding a song the previous refill just
        added, and is applied even when the window is switched off.
        """
        block = library.recent_played_ids(window)
        with self._lock:
            block |= {i["track"]["id"] for i in self._queue
                      if i.get("track", {}).get("id") is not None}
        return block

    @staticmethod
    def _repeat_ladder() -> list[int]:
        """Repeat windows to try, longest first.

        The configured window is used whenever the library can fill the queue.
        Only when it cannot (a small filtered slice, or every track of a tiny
        library recently played) does the window shrink — halving, and never
        below ``REPEAT_WINDOW_FLOOR`` — with 0 as the absolute last resort so a
        two-track library still has something to play.
        """
        window = library.repeat_window()
        if window <= 0:
            return [0]
        ladder: list[int] = []
        current = window
        while current >= REPEAT_WINDOW_FLOOR:
            ladder.append(current)
            current //= 2
        if ladder[-1] != REPEAT_WINDOW_FLOOR:
            ladder.append(REPEAT_WINDOW_FLOOR)
        ladder.append(0)
        return ladder

    def prepared_breaks(self) -> list[dict[str, Any]]:
        """The DJ breaks rendered for upcoming tracks (housekeeping protection).

        Their audio files live in the cache directory and must survive a sweep:
        deleting one would silence a talk the broadcaster is about to play.
        """
        with self._lock:
            return list(self._prepared.values())

    def _queued_music(self) -> bool:
        """True when the queue already holds something to switch away from.

        The first program ever queued has nothing to announce; every later
        program — including the first one of a refill, which the old
        ``items``-only test missed — does.
        """
        with self._lock:
            return bool(self._queue)

    def _note_gap_breaks(self, count: int) -> None:
        if count > 0:
            self._gap_breaks += count

    def gap_breaks(self) -> int:
        """Spacing relaxations in the current queue (see ``_gap_breaks``)."""
        return self._gap_breaks

    def _artist_positions(self) -> tuple[dict[str, int], int]:
        """Where every artist last played, and the position the next song takes.

        Combines the recent plays with the queue that is already waiting, so a
        refill cannot put an artist back on air a few tracks after its own song
        played — the rule spans queue boundaries and refills, exactly like the
        no-long-runs cap. Positions: the newest play is -1, the queue's first
        item is 0, and the NEXT song to be built sits at ``len(queue)`` — which
        is why the origin is returned with the map: a batch build has to record
        its own placements in these queue coordinates, not in batch coordinates.
        """
        gap = library.artist_gap()
        with self._lock:
            queued = [i.get("track") for i in self._queue if i.get("track")]
        if gap <= 0:
            return {}, len(queued)
        history = library.recent_played_artist_keys(gap)
        return library.artist_positions(history, queued), len(queued)

    def _tail_artist(self) -> tuple[str | None, int]:
        """Artist of the queue's tail plus how many of its songs run back.

        Refills append, so the rule has to be measured against what is already
        queued — otherwise every refill could add a third song by whoever is
        playing at the end of the current queue.
        """
        with self._lock:
            artist: str | None = None
            run = 0
            for item in reversed(self._queue):
                key = library.artist_key(item.get("track"))
                if run == 0:
                    artist, run = key, 1
                elif key == artist:
                    run += 1
                else:
                    break
            return artist, run

    def _build_programs(self, count: int, window: int | None = None) -> list[dict[str, Any]]:
        """Build ``count`` themed program items, ordering the queue into runs.

        Each program is a contiguous block of tracks sharing a theme (genre,
        artist, or decade, per ``playback.program.strategy``). The first track
        of every program after the first carries a ``program`` theme and
        ``dj_requested=True`` so the DJ announces the vibe switch before it.

        The rotation is ``library.program_selection`` — the SAME list the
        Programs card shows, so a theme switched off in the browser is switched
        off here too.
        """
        playback = config.get("playback", {}) or {}
        prog = playback.get("program", {}) or {}
        size = max(2, int(prog.get("size", 6)))
        strategy = str(prog.get("strategy", "genre"))
        # The "language" strategy was removed (language detection is gone);
        # coerce any leftover saved value so it degrades to genre grouping.
        if strategy not in ("genre", "artist", "decade"):
            strategy = "genre"
        limit = max(1, int(prog.get("limit", 20) or 20))
        search = playback.get("search", "") or ""
        genres_filter = playback.get("genres") or None
        artists_filter = playback.get("artists") or None
        excludes = library.program_exclusions()

        themes = library.program_selection(strategy, size, limit)["themes"]
        themes = list(themes)
        # NOTE: the global genre/artist filters do NOT pre-filter this theme
        # list when they are on a DIFFERENT dimension. A theme carries only its
        # own dimension (genre themes have no `artist` key, etc.), so filtering
        # the list by the wrong dimension empties it and silently falls back to
        # a flat shuffle. Cross-dimension filters are carried into each theme's
        # track query below, where they are AND-ed correctly.
        #
        # A filter on the SAME dimension as the theme is different: it must
        # NARROW the rotation. Putting it in the same `genres=[...]` list as the
        # theme's own genre OR-ed the two, so a "Punk" program kept playing Punk
        # tracks while only "Electronic" was selected in the Genres card
        # (caught by live validation — see tests for the regression).
        if strategy == "genre" and genres_filter:
            themes = [t for t in themes
                      if any(_genre_matches(t.get("genre"), f)
                             for f in genres_filter)]
        if strategy == "artist" and artists_filter:
            wanted = {str(a).strip().lower() for a in artists_filter}
            themes = [t for t in themes
                      if str(t.get("artist") or "").strip().lower() in wanted]
        if not themes:
            return []

        random.shuffle(themes)
        items: list[dict[str, Any]] = []
        programs_made = 0
        max_consec = _artist_run_settings()
        gap = library.artist_gap()
        # Carry the artist run in from whatever is already queued.
        run_artist, run_len = self._tail_artist()
        # ...and where each artist last played, so spacing holds inside the
        # batch and across programs and refills.
        positions, queue_len = self._artist_positions()
        # Songs that must not come back yet (recently played + already queued).
        block = self._do_not_repeat(window)
        # Round-robin themes so consecutive programs differ, like a real DJ
        # alternating vibes rather than repeating one. Two passes: the first
        # only takes themes the artist gap allows, the second (run only when the
        # first produced nothing at all, so a tiny rotation never goes silent)
        # drops that preference.
        used: set[str] = set()
        # The biggest theme takes the full `size`; everyone else scales against it.
        # NOTE: this is the MAX theme size — passing the SUM made every program
        # scale to the floor (all runs came out 2 tracks while the card promised
        # Rock a 6-track run).
        biggest = max((int(t.get("n", 0) or 0) for t in themes), default=1)
        # Pass 1: honour the artist gap strictly — a theme that cannot fill its
        # program without bringing a band back inside the gap is skipped, exactly
        # like a theme without enough songs. Pass 2 relaxes the INTRA-program
        # spacing (an artist theme is one band, so it can never satisfy it) but
        # still prefers themes that are not themselves inside the gap. Pass 3 is
        # the last resort that also drops that preference, so a one-theme
        # rotation still plays programs instead of going silent.
        for strict, allow_blocked in ((True, False), (False, False), (False, True)):
            for theme in themes:
                if programs_made >= count:
                    break
                theme_id = str(theme.get(strategy))
                if theme_id in used:
                    continue
                offset = queue_len + len(items)
                if (strategy == "artist" and not allow_blocked and gap > 0
                        and self._artist_is_blocked(theme, offset, positions, gap)):
                    # An artist program is ONE band, so the gap is the only rule
                    # that stops the same band's program coming back a couple of
                    # songs later — the run cap cannot, a program IS a run.
                    log.debug("artist program %r held back by the %d-track gap",
                              theme.get("artist"), gap)
                    continue
                # NOT marked used yet: a theme that fails strict spacing in this
                # pass must be retryable in the next one. Marking it here meant
                # pass 1 consumed every theme and the relaxed passes then found
                # nothing left to try, so a small library built NO programs.
                program, ordered = self._build_one_program(
                    theme, strategy, size, biggest,
                    max_consec, gap, positions, run_artist, run_len, block,
                    excludes, genres_filter, artists_filter, search, offset,
                    strict)
                if not ordered:
                    continue
                used.add(theme_id)
                for index, track in enumerate(ordered):
                    # Record the placement so the next program in this same batch
                    # (and the refill after it, via the queue) respects the gap.
                    positions[library.artist_key(track)] = offset + index
                    # The first track of every program after the first carries
                    # the announce. The queue counts too: a refill starts a new
                    # program whose theme may differ from the tail of the
                    # existing queue, and checking the batch alone left that
                    # boundary — the FIRST program of every batch — silently
                    # unannounced (found by the playlist matrix: 4-7 per 120).
                    first = index == 0
                    items.append(self._wrap(
                        track,
                        with_dj=True if (first and (items or self._queued_music()))
                        else None,
                        program=program,
                        program_start=first))
                run_artist = library.artist_key(ordered[-1])
                run_len = self._trailing_run(ordered)
                programs_made += 1
            if programs_made:
                break
        return items

    @staticmethod
    def _artist_is_blocked(theme: dict[str, Any], offset: int,
                           positions: dict[str, int], gap: int) -> bool:
        """Is this artist theme's band still inside the artist gap?"""
        key = library.artist_key({"artist": theme.get("artist")})
        return offset - positions.get(key, library._NEVER_PLAYED) < gap

    def _build_one_program(
        self,
        theme: dict[str, Any],
        strategy: str,
        size: int,
        biggest: int,
        max_consec: int,
        gap: int,
        positions: dict[str, int],
        run_artist: str | None,
        run_len: int,
        block: set[int],
        excludes: dict[str, Any],
        genres_filter: list[str] | None,
        artists_filter: list[str] | None,
        search: str,
        offset: int,
        strict: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One themed run: its label plus the tracks, spaced and rule-checked.

        Returns ``({}, [])`` when the theme cannot fill its own program length
        — "not enough songs for this genre, skip to another program".
        """
        if strategy == "genre":
            kwargs: dict[str, Any] = {"genres": [theme["genre"]], "search": search}
        elif strategy == "artist":
            kwargs = {"artists": [theme["artist"]], "search": search}
        else:  # decade
            kwargs = {"decade": int(theme["decade"]), "search": search}
        # Filters from the OTHER dimensions still apply (AND): e.g. an "Artist"
        # theme + genre filter yields that artist's tracks in that genre (the
        # program is skipped if there are none).
        if genres_filter and strategy != "genre":
            kwargs["genres"] = list(genres_filter)
        if artists_filter and strategy != "artist":
            kwargs["artists"] = list(artists_filter)
        # A program's length comes from its theme's share of the library: the
        # biggest theme runs the full `size`, a 30-track genre runs briefly
        # instead of being skipped for not filling `size`. Artist themes are
        # additionally capped by the no-long-runs rule (it is one band).
        want = int(theme.get("program_size")
                   or library.program_size_for(int(theme.get("n", 0) or 0),
                                               size, biggest))
        if strategy == "artist":
            want = min(want, max_consec)
        want = max(1, want)
        # Over-fetch so the spacer has spare candidates to work with.
        tracks = library.query_tracks(
            limit=max(want * 4, want), random_order=True, **kwargs, **excludes,
            exclude_ids=block)
        placed: dict[str, Any] = {}
        ordered = library.interleave_artists(
            tracks, max_consec, run_artist, run_len, want,
            min_gap=gap, last_positions=positions, report=placed,
            origin=offset, strict=strict)
        self._note_gap_breaks(int(placed.get("gap_breaks", 0)))
        if len(ordered) < want:
            log.debug("program theme %r skipped: %d/%d tracks under the "
                      "%d-in-a-row rule", theme, len(ordered), want, max_consec)
            return {}, []
        program = {
            "kind": strategy,
            "label": theme.get("genre") or theme.get("artist")
            or f"{theme['decade']}s",
        }
        return program, ordered

    @staticmethod
    def _trailing_run(ordered: list[dict[str, Any]]) -> int:
        """How many songs at the end of ``ordered`` share the last artist."""
        key = library.artist_key(ordered[-1])
        run = 0
        for track in reversed(ordered):
            if library.artist_key(track) == key:
                run += 1
            else:
                break
        return run

    def refill(self, count: int = 20) -> int:
        """Top the queue up from the library using the active filters.

        When ``playback.program.enabled`` the queue is filled in themed runs
        (programs) with a DJ break announcing each vibe switch; otherwise it is
        a flat shuffle.
        """
        playback = config.get("playback", {}) or {}
        program_enabled = bool((playback.get("program") or {}).get("enabled", False))
        # Try the configured repeat window first and only relax it if the
        # library cannot fill the queue at all — a small slice must degrade to a
        # shorter window, never to "anything goes" (that let a song repeat
        # immediately, worse than having no rule).
        ladder = self._repeat_ladder()
        for window in ladder:
            added = self._refill_once(count, window, playback, program_enabled)
            if added:
                if window != ladder[0]:
                    log.info("repeat window relaxed from %d to %d songs "
                             "(library too small to fill the queue otherwise)",
                             ladder[0], window)
                return added
        return 0

    def _refill_once(self, count: int, window: int, playback: dict[str, Any],
                     program_enabled: bool) -> int:
        """One refill attempt with a given repeat window (0 = no window)."""
        if program_enabled:
            prog = playback.get("program") or {}
            size = max(2, int(prog.get("size", 6)))
            strategy = str(prog.get("strategy", "genre"))
            # Programs are not all the same length any more (a program scales
            # with its theme's track count), so ask for enough programs to cover
            # `count` tracks at the average length rather than at the ceiling —
            # otherwise a batch of short programs would under-fill the queue.
            lengths = [int(t.get("program_size", size) or size)
                       for t in library.program_selection(
                           strategy, size,
                           max(1, int(prog.get("limit", 20) or 20)))["themes"]]
            if strategy == "artist":
                lengths = [min(length, _artist_run_settings()) for length in lengths]
            per = round(sum(lengths) / len(lengths)) if lengths else size
            n_programs = max(1, math.ceil(count / max(1, per)))
            items = self._build_programs(n_programs, window=window)
            if items:
                with self._lock:
                    self._queue.extend(items)
                self._wake.set()
                return len(items)
            # No theme had enough tracks (tiny library) — fall through to flat.

        excludes: dict[str, Any] = library.program_exclusions()
        # Over-fetch: the spacer needs spare candidates to honour the artist gap.
        # Fetching exactly `count` left the interleave no choice but to reuse an
        # artist it had just placed (measured on the real library: gap 1 where 10
        # was configured), because a random 20-track slice of a genre usually
        # contains several songs by the same band.
        fetch = max(count * 4, count + 20) if library.artist_gap() > 0 else count
        queries: dict[str, Any] = {
            "search": playback.get("search", "") or "",
            "genres": playback.get("genres") or None,
            "artists": playback.get("artists") or None,
            "limit": fetch,
            "random_order": bool(playback.get("shuffle", True)),
            **excludes,
        }
        tracks = library.query_tracks(**queries, exclude_ids=self._do_not_repeat(window))
        if not tracks and not any(excludes.values()):
            # Distinguish "this filter matches nothing at all" from "the repeat
            # window ate the filtered pool". Only the former falls back to the
            # whole library (never silence the station); the latter must not
            # hijack the filter — the window ladder relaxes it instead, so an
            # artist/genre selection is never silently replaced by the whole
            # library.
            filtered_alone = library.query_tracks(**{**queries, "limit": 1})
            if not filtered_alone:
                # Filters matched nothing — fall back to the whole library so the
                # stream never goes silent. Deliberately NOT applied when themes
                # were switched off: resurrecting the whole library would play
                # exactly what the user asked not to hear. The repeat window still
                # applies here; it is the ladder that relaxes it, not this branch.
                tracks = library.query_tracks(
                    limit=fetch, random_order=True,
                    exclude_ids=self._do_not_repeat(window))
        if not tracks:
            return 0
        if not playback.get("shuffle", True):
            existing_paths = {i["track"]["path"] for i in self._queue}
            tracks = [t for t in tracks if t["path"] not in existing_paths]
        else:
            random.shuffle(tracks)
            # Flat shuffle obeys the no-long-runs rule too, so switching
            # programs off cannot reintroduce artist runs. Every track is kept
            # (a single-artist pool simply cannot satisfy the rule).
            run_artist, run_len = self._tail_artist()
            placed: dict[str, Any] = {}
            positions, origin = self._artist_positions()
            ordered = library.interleave_artists(
                tracks, _artist_run_settings(), run_artist, run_len, count,
                min_gap=library.artist_gap(),
                last_positions=positions, report=placed, origin=origin)
            # Keep the requested size: restore covers a stall (a single-artist
            # pool the cap cannot space), while the over-fetched spares are only
            # candidates and must not inflate the queue unspaced. A short result
            # is normal — the next refill sees the updated positions.
            tracks = library.restore_after_interleave(ordered, tracks)[:count]
            self._note_gap_breaks(int(placed.get("gap_breaks", 0)))
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
                    "program_start": bool(item.get("program_start")),
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
            self._attempts.clear()
            self._gap_breaks = 0

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
            self._attempts.pop(item["uid"], None)
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
                attempts = self._attempts.get(uid, 0)
            # The talk decision is already stamped on the item at enqueue time
            # (see _wrap); just read it. A forced program-start (dj_requested
            # explicitly True) or a rolled-interval hit both count.
            explicit = item.get("dj_requested")
            if not explicit:
                continue
            if attempts >= MAX_BREAK_ATTEMPTS:
                # Preparation keeps failing (LLM down, TTS broken): stop asking.
                # Retrying this item every wake (the loop runs every 5 s and on
                # every refill) meant an LLM call every few seconds forever —
                # that is what saturated the user's Ollama. The station plays on
                # without the spoken break rather than hammering the model.
                continue
            prior = upcoming[index - 1]["track"] if index else previous
            started = time.monotonic()
            prepared = dj.prepare_break(
                item["track"], prior, program=item.get("program"))
            if prepared:
                with self._lock:
                    self._prepared[uid] = prepared
                    self._attempts.pop(uid, None)
                log.info(
                    "prepared DJ break for %s - %s in %.1fs (%.1fs audio)",
                    item["track"].get("artist"), item["track"].get("title"),
                    time.monotonic() - started, prepared.get("duration", 0.0),
                )
            else:
                with self._lock:
                    self._attempts[uid] = attempts + 1
                log.warning(
                    "DJ break preparation failed for %s - %s (attempt %d/%d)",
                    item["track"].get("artist"), item["track"].get("title"),
                    attempts + 1, MAX_BREAK_ATTEMPTS,
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "prepared_breaks": len(self._prepared),
                "tracks_played": self._track_counter,
            }


SCHEDULER = Scheduler()
