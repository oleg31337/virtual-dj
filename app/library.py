"""Filesystem scanner: walks the music directory and indexes tags into SQLite."""

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

from . import config, db

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".oga", ".m4a", ".mp4", ".wav", ".opus", ".wma"}

_TAG_KEYS = {
    "title": ("title", "TIT2", "\xa9nam"),
    "artist": ("artist", "TPE1", "\xa9ART", "albumartist", "TPE2"),
    "album": ("album", "TALB", "\xa9alb"),
    "genre": ("genre", "TCON", "\xa9gen"),
    "year": ("date", "TDRC", "TYER", "\xa9day", "year"),
}

# "Artist - Title.mp3" / "01 - Artist - Title.mp3" / "01 Title.mp3"
_FN_TRACKNUM = re.compile(r"^\s*\d{1,3}\s*[-._)]?\s+")


@dataclass
class ScanStatus:
    running: bool = False
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    total_seen: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    current_dir: str = ""
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
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "current_dir": self.current_dir,
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
    stem = path.stem.strip()
    stem = _FN_TRACKNUM.sub("", stem)
    for sep in (" - ", " – ", " — ", "-"):
        if sep in stem:
            left, right = stem.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return None, (stem or None)


def read_metadata(path: Path) -> dict[str, Any]:
    """Extract tags for one audio file. Never raises for a malformed file."""
    meta: dict[str, Any] = {
        "title": None, "artist": None, "album": None,
        "genre": None, "year": None, "duration": None,
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

    if not meta["title"] or not meta["artist"]:
        artist, title = guess_from_filename(path)
        meta["artist"] = meta["artist"] or artist
        meta["title"] = meta["title"] or title

    if meta["year"]:
        match = re.search(r"(\d{4})", str(meta["year"]))
        meta["year"] = match.group(1) if match else None

    for field_name in ("title", "artist", "album"):
        value = meta.get(field_name)
        if not value:
            continue
        # Truncated tags sometimes carry an unmatched opener, e.g.
        # "Success (Thievery Corporation" (the closing ")" was cut off).
        # Drop the parenthetical tail in that case.
        if value.count("(") > value.count(")"):
            meta[field_name] = value.split("(", 1)[0].strip()
    return meta


def iter_audio_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        STATUS.current_dir = dirpath
        for name in filenames:
            if Path(name).suffix.lower() in AUDIO_EXTS:
                yield Path(dirpath) / name


def scan_library(music_dir: str | None = None, full: bool = False) -> dict[str, Any]:
    """Index every audio file under ``music_dir``.

    Files whose size and mtime are unchanged are skipped unless ``full``.
    Tracks that vanished from disk are flagged ``missing`` rather than deleted,
    so history and presets keep resolving.
    """
    root = Path(music_dir or config.get("music_dir", "/mnt/mp3")).expanduser()
    with STATUS.lock:
        if STATUS.running:
            return STATUS.snapshot()
        STATUS.running = True
        STATUS.scanned = STATUS.added = STATUS.updated = STATUS.removed = 0
        STATUS.total_seen = 0
        STATUS.started_at = time.time()
        STATUS.finished_at = None
        STATUS.error = None

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

            meta = read_metadata(path)
            STATUS.scanned += 1
            if prior is None:
                conn.execute(
                    "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
                    "mtime,size,missing) VALUES(?,?,?,?,?,?,?,?,?,0)",
                    (spath, meta["title"], meta["artist"], meta["album"],
                     meta["genre"], meta["year"], meta["duration"],
                     stat.st_mtime, stat.st_size),
                )
                STATUS.added += 1
            else:
                conn.execute(
                    "UPDATE tracks SET title=?,artist=?,album=?,genre=?,year=?,"
                    "duration=?,mtime=?,size=?,missing=0 WHERE id=?",
                    (meta["title"], meta["artist"], meta["album"], meta["genre"],
                     meta["year"], meta["duration"], stat.st_mtime, stat.st_size,
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
    except Exception as exc:
        log.exception("library scan failed")
        with STATUS.lock:
            STATUS.error = str(exc)
    finally:
        with STATUS.lock:
            STATUS.running = False
            STATUS.finished_at = time.time()
            STATUS.current_dir = ""
    return STATUS.snapshot()


def scan_in_background(music_dir: str | None = None, full: bool = False) -> dict[str, Any]:
    if STATUS.running:
        return STATUS.snapshot()
    thread = threading.Thread(
        target=scan_library, kwargs={"music_dir": music_dir, "full": full},
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
        "SUM(COALESCE(duration,0)) AS seconds FROM tracks"
    ).fetchone()
    return {
        "total": row["total"] or 0,
        "missing": row["missing"] or 0,
        "seconds": float(row["seconds"] or 0.0),
    }


def list_genres() -> list[dict[str, Any]]:
    rows = db.connect().execute(
        "SELECT COALESCE(NULLIF(TRIM(genre),''),'Unknown') AS genre, COUNT(*) AS n "
        "FROM tracks WHERE missing=0 GROUP BY 1 ORDER BY n DESC, genre"
    ).fetchall()
    return db.rows_to_dicts(rows)


def list_artists(limit: int = 500) -> list[dict[str, Any]]:
    rows = db.connect().execute(
        "SELECT COALESCE(NULLIF(TRIM(artist),''),'Unknown') AS artist, COUNT(*) AS n "
        "FROM tracks WHERE missing=0 GROUP BY 1 ORDER BY n DESC, artist LIMIT ?",
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
) -> list[dict[str, Any]]:
    where = ["missing = 0"]
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
