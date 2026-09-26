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
# A Cyrillic artist name almost never has an article on the English Wikipedia, so
# the Russian edition is tried FIRST for Cyrillic terms (it is also the version
# that actually has the facts) and the English disambiguator fallbacks are skipped
# for them: on the real library those 4-5 en.wikipedia probes all 404'd, wasting a
# request per name per track.
WIKI_ROOT_RU = "https://ru.wikipedia.org/api/rest_v1/page/summary"
# Unresolvable term -> the moment we may try again. Repeated tracks by the same
# untagged/unfindable artist used to re-probe on every single play.
_WIKI_MISSES: dict[str, float] = {}
_WIKI_MISS_TTL_S = 6 * 60 * 60


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text or "")

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


def _wikipedia(client: httpx.Client, term: str) -> dict[str, Any]:
    """Fetch a short Wikipedia summary for an artist.

    Artist names often resolve to disambiguation pages (e.g. "Queen"), so we
    try the bare term first, then common artist disambiguators ("(band)",
    "(musician)", "(singer)", ...). Returns ``{"summary": str,
    "wikipedia_title": str}`` or ``{}`` if none resolve to a real article.
    """
    import time
    from urllib.parse import quote

    # A known miss costs nothing the second time (per process, with a TTL).
    missed_at = _WIKI_MISSES.get(term)
    if missed_at and time.time() - missed_at < _WIKI_MISS_TTL_S:
        return {}

    if _has_cyrillic(term):
        hosts = [WIKI_ROOT_RU, WIKI_ROOT]
        suffixes = ["", "_(группа)", "_(музыкант)", "_(певица)", "_(группа)"]
    else:
        hosts = [WIKI_ROOT]
        suffixes = ["", "_(band)", "_(musician)", "_(singer)",
                    "_(American_band)", "_(English_band)"]

    for suffix in suffixes:
        cand = f"{term}{suffix}"
        for host in hosts:
            _throttle()
            try:
                resp = client.get(f"{host}/{quote(cand, safe='')}")
            except Exception:                                   # noqa: BLE001
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            if data.get("type", "").endswith("disambiguation"):
                continue
            extract = (data.get("extract") or "").strip()
            if not extract:
                continue
            _WIKI_MISSES.pop(term, None)
            return {"summary": extract[:1200], "wikipedia_title": data.get("title")}

    _WIKI_MISSES[term] = time.time()
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
