"""SQLite storage for the music library, DJ scripts, presets and history."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from . import config

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    genre       TEXT,
    year        TEXT,
    duration    REAL,
    mtime       REAL,
    size        INTEGER,
    missing     INTEGER NOT NULL DEFAULT 0,
    -- Tracks we cannot announce (unreadable script, no usable name) are kept
    -- in the table for the stats panel but excluded from every playlist.
    excluded    INTEGER NOT NULL DEFAULT 0,
    exclude_reason TEXT,
    -- Where title/artist ultimately came from: tags, path, or web.
    meta_source TEXT,
    added_at    REAL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
CREATE INDEX IF NOT EXISTS idx_tracks_genre  ON tracks(genre);
CREATE INDEX IF NOT EXISTS idx_tracks_missing ON tracks(missing);

-- Cache of web confirmation lookups, keyed on "artist|title".
CREATE TABLE IF NOT EXISTS web_lookups (
    query       TEXT PRIMARY KEY,
    result      TEXT NOT NULL,
    fetched_at  REAL DEFAULT (strftime('%s','now'))
);

-- Cache of local-LLM metadata resolutions (genre + corrupt-tag name recovery),
-- keyed on the inputs. Same idea as web_lookups: a rescan costs nothing.
CREATE TABLE IF NOT EXISTS ai_lookups (
    query       TEXT PRIMARY KEY,
    result      TEXT NOT NULL,
    fetched_at  REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS enrichment (
    track_id    INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    facts       TEXT,
    source      TEXT,
    fetched_at  REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS dj_scripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    audio_path  TEXT,
    duration    REAL,
    created_at  REAL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_dj_track ON dj_scripts(track_id);

CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    payload     TEXT NOT NULL,
    created_at  REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER,
    played_at   REAL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_history_played ON history(played_at);
"""


def connect() -> sqlite3.Connection:
    """Return a thread-local connection (SQLite objects are not thread-safe)."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        config.ensure_dirs()
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _LOCAL.conn = conn
    return conn


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


# Columns added after the first release. ``CREATE TABLE IF NOT EXISTS`` is a
# no-op on an existing database, so new columns must be added explicitly.
_ADDED_COLUMNS = (
    ("tracks", "excluded", "INTEGER NOT NULL DEFAULT 0"),
    ("tracks", "exclude_reason", "TEXT"),
    ("tracks", "meta_source", "TEXT"),
    ("tracks", "ai_resolved", "INTEGER NOT NULL DEFAULT 0"),
    ("tracks", "ai_genre", "INTEGER NOT NULL DEFAULT 0"),
    ("tracks", "language", "TEXT NOT NULL DEFAULT 'english'"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to the current schema, in place."""
    for table, column, decl in _ADDED_COLUMNS:
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    # The playable index spans a column added after release; create it here so
    # it exists for both brand-new and pre-existing databases.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_playable "
        "ON tracks(missing, excluded)"
    )
    # ai_lookups may not exist on databases created before this release.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ai_lookups ("
        "query TEXT PRIMARY KEY, result TEXT NOT NULL, "
        "fetched_at REAL DEFAULT (strftime('%s','now')))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_language ON tracks(language)"
    )
    # Cache of artist country-of-origin (MusicBrainz), keyed by artist name.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS artist_origin ("
        "artist TEXT PRIMARY KEY, country TEXT NOT NULL DEFAULT '')"
    )


def close() -> None:
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None:
        conn.close()
        _LOCAL.conn = None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


# --- presets ---------------------------------------------------------------

def save_preset(name: str, payload: dict[str, Any]) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO presets(name, payload) VALUES(?, ?) "
        "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
        (name, json.dumps(payload)),
    )
    conn.commit()


def list_presets() -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT id, name, payload, created_at FROM presets ORDER BY name"
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except (TypeError, json.JSONDecodeError):
            item["payload"] = {}
        out.append(item)
    return out


def get_preset(name: str) -> dict[str, Any] | None:
    row = connect().execute(
        "SELECT payload FROM presets WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        return None


def delete_preset(name: str) -> bool:
    conn = connect()
    cur = conn.execute("DELETE FROM presets WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount > 0


# --- history ---------------------------------------------------------------

def record_play(track_id: int | None) -> None:
    if track_id is None:
        return
    conn = connect()
    conn.execute("INSERT INTO history(track_id) VALUES(?)", (track_id,))
    conn.commit()


def recent_history(limit: int = 50) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT h.played_at, t.id, t.title, t.artist, t.album "
        "FROM history h LEFT JOIN tracks t ON t.id = h.track_id "
        "ORDER BY h.played_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return rows_to_dicts(rows)
