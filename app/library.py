"""Filesystem scanner: walks the music directory and indexes tags into SQLite.

Metadata is resolved in three escalating stages, stopping as soon as the track
is announceable:

1. **Tags** — whatever mutagen can read from the file.
2. **Path** — "Artist - Title.mp3" and folder conventions, when tags are
   missing or useless.
3. **Web** — a search that confirms the guess really is a real recording and
   brings back a genre (see :mod:`app.websearch`).

Anything still unidentifiable, or written in a script the DJ cannot read, is
marked ``excluded``: it stays in the database so the scan report can account
for it, but no playlist will ever select it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from mutagen import File as MutagenFile

from . import config, db, textq, websearch

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".oga", ".m4a", ".mp4", ".wav", ".opus", ".wma"}

_TAG_KEYS = {
    "title": ("title", "TIT2", "\xa9nam"),
    "artist": ("artist", "TPE1", "\xa9ART", "albumartist", "TPE2"),
    "album": ("album", "TALB", "\xa9alb"),
    "genre": ("genre", "TCON", "\xa9gen"),
    "year": ("date", "TDRC", "TYER", "\xa9day", "year"),
}

# Human-readable labels for the "unknown:" stats block.
REJECTION_LABELS = {
    "no_title": "no usable title",
    "no_artist": "no usable artist",
    "non_latin": "non-Latin script",
    "mojibake": "garbled text encoding",
    "unconfirmed": "could not confirm on the web",
}


@dataclass
class ScanStatus:
    running: bool = False
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    total_seen: int = 0
    # Identification outcomes for the scan report.
    from_tags: int = 0
    from_path: int = 0
    from_web: int = 0
    excluded: int = 0
    web_lookups: int = 0
    genres_found: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    current_dir: str = ""
    phase: str = "idle"
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "scanned": self.scanned,
                "added": self.added,
                "updated": self.updated,
                "removed": self.removed,
                "total_seen": self.total_seen,
                "from_tags": self.from_tags,
                "from_path": self.from_path,
                "from_web": self.from_web,
                "excluded": self.excluded,
                "web_lookups": self.web_lookups,
                "genres_found": self.genres_found,
                "reasons": dict(self.reasons),
                "reason_labels": {
                    key: REJECTION_LABELS.get(key, key)
                    for key in self.reasons
                },
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "current_dir": self.current_dir,
                "phase": self.phase,
            }


STATUS = ScanStatus()


def _first_tag(tags: Any, keys: tuple[str, ...]) -> str | None:
    if tags is None:
        return None
    for key in keys:
        try:
            value = tags.get(key)
        except (AttributeError, TypeError, ValueError):
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text[:500]
    return None


def guess_from_filename(path: Path) -> tuple[str | None, str | None]:
    """Best-effort (artist, title) from a filename like 'Artist - Title.mp3'."""
    guess = textq.guess_from_path(path)
    return guess.get("artist"), guess.get("title")


def read_metadata(path: Path, root: Path | None = None,
                  allow_web: bool = False) -> dict[str, Any]:
    """Resolve metadata for one audio file: tags, then path, then the web.

    Never raises for a malformed file. The returned dict always carries
    ``excluded``/``exclude_reason`` so the caller can record why a track will
    not be played, and ``meta_source`` describing where the names came from.
    """
    meta: dict[str, Any] = {
        "title": None, "artist": None, "album": None,
        "genre": None, "year": None, "duration": None,
        "meta_source": None, "excluded": 0, "exclude_reason": None,
        "web_checked": False,
    }
    try:
        audio = MutagenFile(path, easy=True)
    except Exception as exc:  # mutagen raises a wide variety on bad files
        log.debug("mutagen failed on %s: %s", path, exc)
        audio = None

    if audio is not None:
        for field_name, keys in _TAG_KEYS.items():
            meta[field_name] = _first_tag(getattr(audio, "tags", None), keys)
        info = getattr(audio, "info", None)
        length = getattr(info, "length", None)
        if isinstance(length, (int, float)) and length > 0:
            meta["duration"] = float(length)

    # Clean up tag text before judging it: a value that is only decoration
    # ("- - - - Metallica - - - -") should be treated as absent.
    for field_name in ("title", "artist", "album"):
        if meta[field_name]:
            meta[field_name] = textq.clean_name(meta[field_name]) or None

    # Truncated tags sometimes carry an unmatched opener, e.g.
    # "Success (Thievery Corporation" — drop the parenthetical tail.
    for field_name in ("title", "artist", "album"):
        value = meta.get(field_name)
        if value and value.count("(") > value.count(")"):
            meta[field_name] = value.split("(", 1)[0].strip() or None

    tags_usable = textq.is_usable(meta["title"]) and textq.is_usable(meta["artist"])
    if tags_usable:
        meta["meta_source"] = "tags"

    # --- stage 2: guess from the path ------------------------------------
    # Record whether the *guess* was unreadable, so the exclusion reason can
    # say "non_latin" rather than the misleading "no_title" when the only
    # guess available was in a script we cannot announce.
    guess_non_latin = False
    if not tags_usable:
        guess = textq.guess_from_path(path, root)
        used_guess = False
        for field_name in ("artist", "title", "album"):
            if not textq.is_usable(meta.get(field_name)) and guess.get(field_name):
                if textq.is_usable(guess[field_name]):
                    meta[field_name] = guess[field_name]
                    used_guess = True
                elif textq.has_non_latin_script(guess[field_name]):
                    guess_non_latin = True
        if used_guess:
            meta["meta_source"] = "path"

    if meta["year"]:
        match = re.search(r"(\d{4})", str(meta["year"]))
        meta["year"] = match.group(1) if match else None

    meta["genre"] = websearch.canonical_genre(meta.get("genre"))

    # --- stage 3: confirm on the web -------------------------------------
    # Only for path guesses: confirmed tags are already trustworthy, and a
    # 13k-file library must not make thousands of network calls per scan.
    # Genre for tag-identified tracks that lack one is filled lazily during
    # DJ preparation (see enrich.py), not here.
    reason = textq.rejection_reason(meta.get("artist"), meta.get("title"))
    # If the only metadata we could recover was in an unreadable script, label
    # it non_latin even though the field ended up empty after filtering.
    if guess_non_latin and reason in (None, "no_title", "no_artist"):
        reason = "non_latin"
    if allow_web and meta.get("meta_source") == "path" and reason is None:
        meta["web_checked"] = True
        result = websearch.confirm_track(meta.get("artist"), meta.get("title"))
        if result.get("confirmed"):
            # Prefer the database's spelling over our filename guess.
            if result.get("artist"):
                meta["artist"] = result["artist"]
            if result.get("title"):
                meta["title"] = result["title"]
            if result.get("album") and not meta.get("album"):
                meta["album"] = result["album"]
            if result.get("year") and not meta.get("year"):
                meta["year"] = result["year"]
            meta["meta_source"] = "web"
        else:
            reason = "unconfirmed"
        if result.get("genre") and not meta.get("genre"):
            meta["genre"] = result["genre"]

    if reason is not None:
        meta["excluded"] = 1
        meta["exclude_reason"] = reason
        if not meta.get("meta_source"):
            meta["meta_source"] = "none"
    return meta


def iter_audio_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        STATUS.current_dir = dirpath
        for name in filenames:
            if Path(name).suffix.lower() in AUDIO_EXTS:
                yield Path(dirpath) / name


def scan_library(music_dir: str | None = None, full: bool = False,
                 use_web: bool | None = None) -> dict[str, Any]:
    """Index every audio file under ``music_dir``.

    Files whose size and mtime are unchanged are skipped unless ``full``.
    Tracks that vanished from disk are flagged ``missing`` rather than deleted,
    so history and presets keep resolving. Tracks that cannot be identified
    well enough to announce are flagged ``excluded`` with a reason.
    """
    root = Path(music_dir or config.get("music_dir", "")).expanduser()
    if use_web is None:
        use_web = bool(config.get("websearch.enabled", True))
    with STATUS.lock:
        if STATUS.running:
            return STATUS.snapshot()
        STATUS.running = True
        STATUS.scanned = STATUS.added = STATUS.updated = STATUS.removed = 0
        STATUS.total_seen = 0
        STATUS.from_tags = STATUS.from_path = STATUS.from_web = 0
        STATUS.excluded = STATUS.web_lookups = STATUS.genres_found = 0
        STATUS.reasons = {}
        STATUS.started_at = time.time()
        STATUS.finished_at = None
        STATUS.error = None
        STATUS.phase = "scanning"

    conn = db.connect()
    try:
        if not root.is_dir():
            raise NotADirectoryError(f"music_dir does not exist: {root}")

        existing = {
            row["path"]: (row["id"], row["mtime"], row["size"])
            for row in conn.execute("SELECT id, path, mtime, size FROM tracks")
        }
        seen: set[str] = set()
        batch = 0

        for path in iter_audio_files(root):
            spath = str(path)
            seen.add(spath)
            try:
                stat = path.stat()
            except OSError:
                continue
            STATUS.total_seen += 1

            prior = existing.get(spath)
            unchanged = (
                prior is not None
                and not full
                and prior[1] == stat.st_mtime
                and prior[2] == stat.st_size
            )
            if unchanged:
                if prior:  # clear a stale missing flag
                    conn.execute("UPDATE tracks SET missing=0 WHERE id=?", (prior[0],))
                continue

            meta = read_metadata(path, root=root, allow_web=use_web)
            STATUS.scanned += 1

            with STATUS.lock:
                source = meta.get("meta_source")
                if source == "tags":
                    STATUS.from_tags += 1
                elif source == "path":
                    STATUS.from_path += 1
                elif source == "web":
                    STATUS.from_web += 1
                if meta.get("web_checked"):
                    STATUS.web_lookups += 1
                if meta.get("genre"):
                    STATUS.genres_found += 1
                if meta.get("excluded"):
                    STATUS.excluded += 1
                    reason = meta.get("exclude_reason") or "unknown"
                    STATUS.reasons[reason] = STATUS.reasons.get(reason, 0) + 1

            if prior is None:
                conn.execute(
                    "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
                    "mtime,size,missing,excluded,exclude_reason,meta_source) "
                    "VALUES(?,?,?,?,?,?,?,?,?,0,?,?,?)",
                    (spath, meta["title"], meta["artist"], meta["album"],
                     meta["genre"], meta["year"], meta["duration"],
                     stat.st_mtime, stat.st_size,
                     meta["excluded"], meta["exclude_reason"], meta["meta_source"]),
                )
                STATUS.added += 1
            else:
                conn.execute(
                    "UPDATE tracks SET title=?,artist=?,album=?,genre=?,year=?,"
                    "duration=?,mtime=?,size=?,missing=0,excluded=?,"
                    "exclude_reason=?,meta_source=? WHERE id=?",
                    (meta["title"], meta["artist"], meta["album"], meta["genre"],
                     meta["year"], meta["duration"], stat.st_mtime, stat.st_size,
                     meta["excluded"], meta["exclude_reason"], meta["meta_source"],
                     prior[0]),
                )
                STATUS.updated += 1

            batch += 1
            if batch % 500 == 0:
                conn.commit()

        conn.commit()

        gone = [p for p in existing if p not in seen and p.startswith(str(root))]
        for chunk_start in range(0, len(gone), 500):
            chunk = gone[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            conn.execute(
                f"UPDATE tracks SET missing=1 WHERE path IN ({placeholders})", chunk
            )
        STATUS.removed = len(gone)
        conn.commit()
        log.info(
            "scan complete: %d seen, %d indexed (%d tags, %d path, %d web), "
            "%d excluded %s",
            STATUS.total_seen, STATUS.scanned, STATUS.from_tags,
            STATUS.from_path, STATUS.from_web, STATUS.excluded, STATUS.reasons,
        )
    except Exception as exc:
        log.exception("library scan failed")
        with STATUS.lock:
            STATUS.error = str(exc)
    finally:
        with STATUS.lock:
            STATUS.running = False
            STATUS.finished_at = time.time()
            STATUS.current_dir = ""
            STATUS.phase = "idle"
    return STATUS.snapshot()


def scan_in_background(music_dir: str | None = None, full: bool = False,
                       use_web: bool | None = None) -> dict[str, Any]:
    if STATUS.running:
        return STATUS.snapshot()
    thread = threading.Thread(
        target=scan_library,
        kwargs={"music_dir": music_dir, "full": full, "use_web": use_web},
        name="vdj-scanner", daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    return STATUS.snapshot()


# --- queries ---------------------------------------------------------------

def library_stats() -> dict[str, Any]:
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN missing=1 THEN 1 ELSE 0 END) AS missing, "
        "SUM(CASE WHEN excluded=1 THEN 1 ELSE 0 END) AS excluded, "
        "SUM(CASE WHEN missing=0 AND excluded=0 THEN 1 ELSE 0 END) AS playable, "
        "SUM(CASE WHEN missing=0 AND excluded=0 AND "
        "    (genre IS NULL OR TRIM(genre)='') THEN 1 ELSE 0 END) AS no_genre, "
        "SUM(CASE WHEN missing=0 AND excluded=0 THEN COALESCE(duration,0) "
        "    ELSE 0 END) AS seconds FROM tracks"
    ).fetchone()
    reasons = {
        r["exclude_reason"] or "unknown": r["n"]
        for r in conn.execute(
            "SELECT exclude_reason, COUNT(*) AS n FROM tracks "
            "WHERE excluded=1 AND missing=0 GROUP BY exclude_reason ORDER BY n DESC"
        )
    }
    sources = {
        r["meta_source"] or "unknown": r["n"]
        for r in conn.execute(
            "SELECT meta_source, COUNT(*) AS n FROM tracks "
            "WHERE missing=0 AND excluded=0 GROUP BY meta_source ORDER BY n DESC"
        )
    }
    return {
        "total": row["total"] or 0,
        "missing": row["missing"] or 0,
        "excluded": row["excluded"] or 0,
        "playable": row["playable"] or 0,
        "no_genre": row["no_genre"] or 0,
        "seconds": float(row["seconds"] or 0.0),
        "unknown_reasons": reasons,
        "reason_labels": {
            key: REJECTION_LABELS.get(key, key) for key in reasons
        },
        "meta_sources": sources,
    }


def excluded_tracks(limit: int = 200, offset: int = 0,
                    reason: str | None = None) -> list[dict[str, Any]]:
    """The tracks that were skipped, for the 'unknown' panel in the UI."""
    where = ["excluded = 1", "missing = 0"]
    params: list[Any] = []
    if reason:
        where.append("exclude_reason = ?")
        params.append(reason)
    params += [max(1, min(limit, 2000)), max(0, offset)]
    rows = db.connect().execute(
        "SELECT id, path, title, artist, album, exclude_reason, meta_source "
        f"FROM tracks WHERE {' AND '.join(where)} "
        "ORDER BY path LIMIT ? OFFSET ?",
        params,
    ).fetchall()
    return db.rows_to_dicts(rows)


def list_genres() -> list[dict[str, Any]]:
    rows = db.connect().execute(
        "SELECT COALESCE(NULLIF(TRIM(genre),''),'Unknown') AS genre, COUNT(*) AS n "
        "FROM tracks WHERE missing=0 AND excluded=0 GROUP BY 1 "
        "ORDER BY n DESC, genre"
    ).fetchall()
    return db.rows_to_dicts(rows)


def list_artists(limit: int = 500) -> list[dict[str, Any]]:
    rows = db.connect().execute(
        "SELECT COALESCE(NULLIF(TRIM(artist),''),'Unknown') AS artist, COUNT(*) AS n "
        "FROM tracks WHERE missing=0 AND excluded=0 GROUP BY 1 "
        "ORDER BY n DESC, artist LIMIT ?",
        (limit,),
    ).fetchall()
    return db.rows_to_dicts(rows)


def query_tracks(
    search: str = "",
    genres: list[str] | None = None,
    artists: list[str] | None = None,
    limit: int = 200,
    offset: int = 0,
    random_order: bool = False,
    include_excluded: bool = False,
) -> list[dict[str, Any]]:
    # Excluded tracks are never playlist material — the DJ cannot announce
    # them. They are only reachable via ``excluded_tracks()`` for reporting.
    where = ["missing = 0"]
    if not include_excluded:
        where.append("excluded = 0")
    params: list[Any] = []

    if search:
        where.append(
            "(title LIKE ? OR artist LIKE ? OR album LIKE ? OR path LIKE ?)"
        )
        like = f"%{search}%"
        params += [like, like, like, like]

    if genres:
        clauses = []
        for genre in genres:
            if genre == "Unknown":
                clauses.append("(genre IS NULL OR TRIM(genre) = '')")
            else:
                clauses.append("genre LIKE ?")
                params.append(f"%{genre}%")
        where.append("(" + " OR ".join(clauses) + ")")

    if artists:
        placeholders = ",".join("?" * len(artists))
        where.append(f"artist IN ({placeholders})")
        params += artists

    order = "RANDOM()" if random_order else "artist COLLATE NOCASE, album, title"
    sql = (
        "SELECT id, path, title, artist, album, genre, year, duration FROM tracks "
        f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ? OFFSET ?"
    )
    params += [max(1, min(limit, 2000)), max(0, offset)]
    return db.rows_to_dicts(db.connect().execute(sql, params).fetchall())


def get_track(track_id: int) -> dict[str, Any] | None:
    row = db.connect().execute(
        "SELECT * FROM tracks WHERE id = ?", (track_id,)
    ).fetchone()
    return dict(row) if row else None
