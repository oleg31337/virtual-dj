"""Fetch extra facts about a track from MusicBrainz and Wikipedia.

Everything here is best-effort: the network may be absent (the app is designed
to run on a LAN appliance) and the DJ must keep talking regardless. Results are
cached in SQLite so we hit the public APIs at most once per track.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from . import config, db

log = logging.getLogger(__name__)

MB_ROOT = "https://musicbrainz.org/ws/2"
WIKI_ROOT = "https://en.wikipedia.org/api/rest_v1/page/summary"

# MusicBrainz asks for max 1 request/second from anonymous clients.
_MIN_INTERVAL = 1.1
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    delta = time.monotonic() - _last_call
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_call = time.monotonic()


def cached_facts(track_id: int) -> dict[str, Any] | None:
    row = db.connect().execute(
        "SELECT facts, source FROM enrichment WHERE track_id = ?", (track_id,)
    ).fetchone()
    if not row:
        return None
    try:
        return {"facts": json.loads(row["facts"]), "source": row["source"]}
    except (TypeError, json.JSONDecodeError):
        return None


def store_facts(track_id: int, facts: dict[str, Any], source: str) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO enrichment(track_id, facts, source) VALUES(?,?,?) "
        "ON CONFLICT(track_id) DO UPDATE SET facts=excluded.facts, "
        "source=excluded.source, fetched_at=strftime('%s','now')",
        (track_id, json.dumps(facts), source),
    )
    conn.commit()


def _musicbrainz(client: httpx.Client, artist: str, title: str) -> dict[str, Any]:
    query = f'recording:"{title}" AND artist:"{artist}"'
    _throttle()
    resp = client.get(
        f"{MB_ROOT}/recording",
        params={"query": query, "fmt": "json", "limit": 1},
    )
    resp.raise_for_status()
    recordings = resp.json().get("recordings") or []
    if not recordings:
        return {}
    rec = recordings[0]
    out: dict[str, Any] = {}
    if rec.get("first-release-date"):
        out["first_release_date"] = rec["first-release-date"]
    releases = rec.get("releases") or []
    if releases:
        first = releases[0]
        if first.get("title"):
            out["release_title"] = first["title"]
        group = first.get("release-group") or {}
        if group.get("primary-type"):
            out["release_type"] = group["primary-type"]
    tags = [t.get("name") for t in (rec.get("tags") or []) if t.get("name")]
    if tags:
        out["tags"] = tags[:8]
    credits = rec.get("artist-credit") or []
    names = [c.get("name") for c in credits if isinstance(c, dict) and c.get("name")]
    if len(names) > 1:
        out["credited_artists"] = names
    return out


def _musicbrainz_artist_country(name: str) -> str | None:
    """Best-effort artist country-of-origin from MusicBrainz (for language hint).

    Returns the ISO country code (or area name) of the top artist match, or
    ``None`` if unresolved. The result is mapped to a language bucket elsewhere;
    here we just return the raw origin so caching is language-agnostic.
    """
    query = f'artist:"{name}"'
    _throttle()
    resp = httpx.get(
        f"{MB_ROOT}/artist",
        params={"query": query, "fmt": "json", "limit": 1},
    )
    resp.raise_for_status()
    artists = resp.json().get("artists") or []
    if not artists:
        return None
    top = artists[0]
    # Guard against a wildly wrong match (MusicBrainz fuzzy search): the top
    # hit should at least share the queried name (accent-insensitive).
    norm = lambda s: "".join(c for c in (s or "").lower() if c.isalnum())
    if norm(name) and norm(name) not in norm(top.get("name", "")):
        return None
    return top.get("country") or (top.get("area") or {}).get("name")


def _norm_artist(name: str) -> str:
    """Accent- and case-insensitive artist key (e.g. Téléphone -> telephone).

    Lets one MusicBrainz origin resolve cover both "Téléphone" and "Telephone"
    spellings of the same band without a second network call.
    """
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(c)
    ).lower().strip()


def artist_country_cached(name: str) -> str | None:
    """Cache-only artist country lookup (no network call).

    Returns the cached MusicBrainz country for ``name`` (or its accent-stripped
    variant), or ``None`` if not cached. Used at scan time for artists without a
    language diacritic, so an already-resolved origin (e.g. an unaccented
    "Telephone" reusing "Téléphone" -> France) is applied without hitting the
    API for every obvious English artist.
    """
    name = (name or "").strip()
    if not name:
        return None
    conn = db.connect()
    row = conn.execute(
        "SELECT country FROM artist_origin WHERE artist = ?", (name,)
    ).fetchone()
    if row is not None:
        return row["country"] or None
    norm = _norm_artist(name)
    if norm and norm != name.lower():
        nrow = conn.execute(
            "SELECT country FROM artist_origin WHERE artist = ?", (norm,)
        ).fetchone()
        if nrow is not None:
            return nrow["country"] or None
    return None


def artist_country(name: str) -> str | None:
    """Resolve and cache an artist's country-of-origin.

    Caches both hits and misses (empty string) in the ``artist_origin`` table
    so each distinct artist is looked up at most once, and the result survives
    re-scans. Also shares a resolved origin across accent/spelling variants of
    the same name (Téléphone / Telephone) via an accent-stripped key. Returns the
    ISO country code / area name, or ``None``.
    """
    name = (name or "").strip()
    if not name:
        return None
    conn = db.connect()
    row = conn.execute(
        "SELECT country FROM artist_origin WHERE artist = ?", (name,)
    ).fetchone()
    if row is not None:
        return row["country"] or None
    # Fall back to the accent-stripped key so an unaccented spelling reuses a
    # previously resolved accented one (no second MusicBrainz call).
    norm = _norm_artist(name)
    if norm and norm != name.lower():
        nrow = conn.execute(
            "SELECT country FROM artist_origin WHERE artist = ?", (norm,)
        ).fetchone()
        if nrow is not None:
            return nrow["country"] or None
    country = None
    try:
        country = _musicbrainz_artist_country(name)
    except Exception as exc:
        log.debug("musicbrainz artist lookup failed for %s: %s", name, exc)
    # Store under both the exact name and the accent-stripped key so either
    # spelling finds it next time.
    conn.execute(
        "INSERT INTO artist_origin(artist, country) VALUES(?, ?) "
        "ON CONFLICT(artist) DO UPDATE SET country=excluded.country",
        (name, country or ""),
    )
    if norm and norm != name.lower():
        conn.execute(
            "INSERT INTO artist_origin(artist, country) VALUES(?, ?) "
            "ON CONFLICT(artist) DO UPDATE SET country=excluded.country",
            (norm, country or ""),
        )
    conn.commit()
    return country or None


def _wikipedia(client: httpx.Client, term: str) -> dict[str, Any]:
    """Fetch a short Wikipedia summary for an artist.

    Artist names often resolve to disambiguation pages (e.g. "Queen"), so we
    try the bare term first, then common artist disambiguators ("(band)",
    "(musician)", "(singer)", ...). Returns ``{"summary": str,
    "wikipedia_title": str}`` or ``{}`` if none resolve to a real article.
    """
    from urllib.parse import quote

    candidates = [
        term,
        f"{term}_(band)",
        f"{term}_(musician)",
        f"{term}_(singer)",
        f"{term}_(American_band)",
        f"{term}_(English_band)",
    ]
    for cand in candidates:
        _throttle()
        resp = client.get(f"{WIKI_ROOT}/{quote(cand, safe='')}")
        if resp.status_code != 200:
            continue
        data = resp.json()
        if data.get("type", "").endswith("disambiguation"):
            continue
        extract = (data.get("extract") or "").strip()
        if not extract:
            continue
        return {"summary": extract[:1200], "wikipedia_title": data.get("title")}
    return {}


def _musicbrainz_artist_tags(name: str) -> list[str]:
    """Best-effort artist tags from MusicBrainz (genre/style facts for the LLM).

    Recording searches rarely carry tags, so the reliable tag source is the
    artist. Returns up to a handful of tag names, or ``[]`` if unresolved.
    """
    query = f'artist:"{name}"'
    _throttle()
    resp = httpx.get(
        f"{MB_ROOT}/artist",
        params={"query": query, "fmt": "json", "limit": 1},
    )
    resp.raise_for_status()
    artists = resp.json().get("artists") or []
    if not artists:
        return []
    top = artists[0]
    norm = lambda s: "".join(c for c in (s or "").lower() if c.isalnum())
    if norm(name) and norm(name) not in norm(top.get("name", "")):
        return []
    return [t.get("name") for t in (top.get("tags") or []) if t.get("name")][:8]


def enrich_track(track: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Return a dict of facts about ``track``, using the cache when possible."""
    track_id = track.get("id")
    if track_id and not force:
        cached = cached_facts(track_id)
        if cached is not None:
            return cached["facts"]

    if not config.get("enrich.enabled", True):
        return {}

    artist = (track.get("artist") or "").strip()
    title = (track.get("title") or "").strip()
    if not artist or not title:
        return {}

    facts: dict[str, Any] = {}
    sources: list[str] = []
    headers = {"User-Agent": config.get("enrich.user_agent", "VirtualDJ/1.0")}
    timeout = float(config.get("enrich.timeout_s", 15))

    try:
        with httpx.Client(timeout=timeout, headers=headers,
                          follow_redirects=True) as client:
            try:
                mb = _musicbrainz(client, artist, title)
                if mb:
                    facts.update(mb)
                    sources.append("musicbrainz")
            except Exception as exc:
                log.debug("musicbrainz lookup failed for %s - %s: %s",
                          artist, title, exc)
            try:
                wiki = _wikipedia(client, artist.replace(" ", "_"))
                if wiki:
                    facts["artist_summary"] = wiki["summary"]
                    sources.append("wikipedia")
            except Exception as exc:
                log.debug("wikipedia lookup failed for %s: %s", artist, exc)
            try:
                artist_tags = _musicbrainz_artist_tags(artist)
                if artist_tags:
                    # Merge with any tags from the recording lookup, de-duped.
                    merged = list(facts.get("tags") or [])
                    for t in artist_tags:
                        if t not in merged:
                            merged.append(t)
                    facts["tags"] = merged[:8]
                    if "musicbrainz" not in sources:
                        sources.append("musicbrainz")
            except Exception as exc:
                log.debug("musicbrainz artist tags failed for %s: %s", artist, exc)
    except Exception as exc:
        log.debug("enrichment client error: %s", exc)

    if track_id:
        store_facts(track_id, facts, ",".join(sources) or "none")
    return facts
