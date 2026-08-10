"""Built-in web search used to confirm guessed metadata and infer genre.

When a track's tags are missing we guess artist/title from the file path. A
guess is not trustworthy on its own — "Fade To Black" in a folder called
"Metallica" could be the right pairing or a mislabelled bootleg. This module
asks the open web whether that artist/title pair actually exists, and brings
back a genre while it is there.

Sources, in order, all keyless and rate-limit-friendly:

1. **MusicBrainz** — authoritative recording/artist match, plus community
   genre tags. A hit here is treated as confirmation.
2. **iTunes Search** — excellent for mainstream music and returns a
   ``primaryGenreName`` directly.
3. **DuckDuckGo Instant Answer** — last-resort corroboration for anything the
   music databases miss.

Everything is cached in SQLite keyed on the query, so a rescan of a 13k-file
library re-uses previous answers instead of hammering public APIs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Any

import httpx

from . import config, db
from .textq import clean_name

log = logging.getLogger(__name__)

MB_ROOT = "https://musicbrainz.org/ws/2"
ITUNES_ROOT = "https://itunes.apple.com/search"
DDG_ROOT = "https://api.duckduckgo.com/"

# MusicBrainz allows ~1 req/s for anonymous clients; iTunes ~20/min.
_MB_INTERVAL = 1.1
_ITUNES_INTERVAL = 2.0
_lock = threading.Lock()
_last_call: dict[str, float] = {}

# Free-text genre strings collapse onto these canonical buckets so the genre
# filter in the UI stays usable instead of sprouting 400 near-duplicates.
_GENRE_CANON = {
    "rock": "Rock", "hard rock": "Rock", "classic rock": "Rock",
    "album rock": "Rock", "rock & roll": "Rock", "rock and roll": "Rock",
    "indie rock": "Indie", "alternative rock": "Alternative",
    "alternative": "Alternative", "alt rock": "Alternative",
    "grunge": "Grunge", "punk": "Punk", "punk rock": "Punk",
    "post-punk": "Punk", "hardcore": "Punk",
    "metal": "Metal", "heavy metal": "Metal", "thrash metal": "Metal",
    "death metal": "Metal", "black metal": "Metal", "doom metal": "Metal",
    "nu metal": "Metal", "metalcore": "Metal", "power metal": "Metal",
    "pop": "Pop", "pop rock": "Pop", "synth-pop": "Pop", "synthpop": "Pop",
    "dance pop": "Pop", "electropop": "Pop", "teen pop": "Pop",
    "electronic": "Electronic", "electronica": "Electronic",
    "techno": "Electronic", "house": "Electronic", "trance": "Electronic",
    "drum and bass": "Electronic", "dubstep": "Electronic",
    "ambient": "Ambient", "downtempo": "Ambient", "chillout": "Ambient",
    "trip hop": "Trip Hop", "trip-hop": "Trip Hop",
    "hip hop": "Hip Hop", "hip-hop": "Hip Hop", "rap": "Hip Hop",
    "r&b": "R&B", "rnb": "R&B", "soul": "Soul", "funk": "Funk",
    "disco": "Disco", "jazz": "Jazz", "smooth jazz": "Jazz",
    "bebop": "Jazz", "swing": "Jazz", "blues": "Blues",
    "classical": "Classical", "baroque": "Classical", "opera": "Classical",
    "orchestral": "Classical", "piano": "Classical",
    "country": "Country", "folk": "Folk", "folk rock": "Folk",
    "americana": "Folk", "bluegrass": "Folk", "singer-songwriter": "Folk",
    "reggae": "Reggae", "ska": "Reggae", "dub": "Reggae",
    "latin": "Latin", "salsa": "Latin", "bossa nova": "Latin",
    "world": "World", "new age": "New Age", "soundtrack": "Soundtrack",
    "score": "Soundtrack", "film score": "Soundtrack",
    "instrumental": "Instrumental", "experimental": "Experimental",
    "industrial": "Industrial", "gothic": "Gothic", "progressive rock": "Progressive",
    "prog rock": "Progressive", "psychedelic rock": "Psychedelic",
}

_NOISE_TAGS = {
    "seen live", "favourites", "favorites", "awesome", "beautiful", "cool",
    "male vocalists", "female vocalists", "british", "american", "german",
    "russian", "japanese", "french", "swedish", "00s", "90s", "80s", "70s",
    "60s", "albums i own", "under 2000 listeners", "spotify",
}


def canonical_genre(raw: str | None) -> str | None:
    """Map a free-text genre onto a canonical bucket."""
    if not raw:
        return None
    text = clean_name(str(raw)).strip().lower()
    text = re.sub(r"\s*/\s*", " ", text).strip()
    if not text or text in _NOISE_TAGS:
        return None
    if text in _GENRE_CANON:
        return _GENRE_CANON[text]
    # Longest-substring match: "melodic death metal" -> Metal.
    best: tuple[int, str] | None = None
    for key, value in _GENRE_CANON.items():
        if key in text and (best is None or len(key) > best[0]):
            best = (len(key), value)
    if best:
        return best[1]
    # Unrecognised but plausible: title-case it rather than discarding.
    if 2 < len(text) <= 24 and re.fullmatch(r"[a-z0-9 &'\-]+", text):
        return text.title()
    return None


def _similar(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _throttle(key: str, interval: float) -> None:
    with _lock:
        delta = time.monotonic() - _last_call.get(key, 0.0)
        if delta < interval:
            time.sleep(interval - delta)
        _last_call[key] = time.monotonic()


# --- cache -----------------------------------------------------------------

def cached_lookup(key: str) -> dict[str, Any] | None:
    try:
        row = db.connect().execute(
            "SELECT result FROM web_lookups WHERE query = ?", (key,)
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        return json.loads(row["result"])
    except (TypeError, json.JSONDecodeError):
        return None


def store_lookup(key: str, result: dict[str, Any]) -> None:
    try:
        conn = db.connect()
        conn.execute(
            "INSERT INTO web_lookups(query, result) VALUES(?,?) "
            "ON CONFLICT(query) DO UPDATE SET result=excluded.result, "
            "fetched_at=strftime('%s','now')",
            (key, json.dumps(result)),
        )
        conn.commit()
    except Exception as exc:
        log.debug("could not cache web lookup: %s", exc)


# --- providers -------------------------------------------------------------

def _mb_search(client: httpx.Client, artist: str, title: str) -> dict[str, Any]:
    """Ask MusicBrainz whether this artist/title pair exists."""
    _throttle("mb", _MB_INTERVAL)
    query = f'recording:"{title}"'
    if artist:
        query += f' AND artist:"{artist}"'
    resp = client.get(
        f"{MB_ROOT}/recording",
        params={"query": query, "fmt": "json", "limit": 3},
    )
    resp.raise_for_status()
    recordings = resp.json().get("recordings") or []
    if not recordings:
        return {}

    best: dict[str, Any] = {}
    best_score = 0.0
    for rec in recordings:
        rec_title = rec.get("title") or ""
        credits = rec.get("artist-credit") or []
        rec_artist = ""
        for credit in credits:
            if isinstance(credit, dict) and credit.get("name"):
                rec_artist = credit["name"]
                break
        title_score = _similar(title, rec_title) if title else 0.0
        artist_score = _similar(artist, rec_artist) if artist else 0.5
        # MusicBrainz's own relevance score (0-100) as a weak tiebreaker.
        mb_score = float(rec.get("score") or 0) / 100.0
        score = (title_score * 0.5) + (artist_score * 0.35) + (mb_score * 0.15)
        if score > best_score:
            best_score = score
            tags = [
                t.get("name") for t in (rec.get("tags") or [])
                if t.get("name") and t["name"].lower() not in _NOISE_TAGS
            ]
            best = {
                "artist": rec_artist or None,
                "title": rec_title or None,
                "tags": tags[:8],
                "score": round(score, 3),
                "provider": "musicbrainz",
            }
    return best if best_score >= 0.55 else {}


def _itunes_search(client: httpx.Client, artist: str, title: str) -> dict[str, Any]:
    """iTunes: strong mainstream coverage and a genre for free."""
    _throttle("itunes", _ITUNES_INTERVAL)
    term = " ".join(part for part in (artist, title) if part).strip()
    if not term:
        return {}
    resp = client.get(
        ITUNES_ROOT,
        params={"term": term, "media": "music", "entity": "song", "limit": 3},
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    best: dict[str, Any] = {}
    best_score = 0.0
    for item in results:
        item_title = item.get("trackName") or ""
        item_artist = item.get("artistName") or ""
        title_score = _similar(title, item_title) if title else 0.0
        artist_score = _similar(artist, item_artist) if artist else 0.5
        score = (title_score * 0.6) + (artist_score * 0.4)
        if score > best_score:
            best_score = score
            year = None
            release = item.get("releaseDate") or ""
            match = re.match(r"(\d{4})", release)
            if match:
                year = match.group(1)
            best = {
                "artist": item_artist or None,
                "title": item_title or None,
                "album": item.get("collectionName") or None,
                "genre": item.get("primaryGenreName") or None,
                "year": year,
                "score": round(score, 3),
                "provider": "itunes",
            }
    return best if best_score >= 0.55 else {}


def _ddg_search(client: httpx.Client, artist: str, title: str) -> dict[str, Any]:
    """DuckDuckGo Instant Answer — corroboration only, no strong claims."""
    term = " ".join(part for part in (artist, title) if part) + " song"
    resp = client.get(
        DDG_ROOT,
        params={"q": term, "format": "json", "no_html": 1, "skip_disambig": 1},
    )
    if resp.status_code != 200:
        return {}
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return {}
    abstract = (data.get("AbstractText") or "").strip()
    heading = (data.get("Heading") or "").strip()
    if not abstract:
        return {}
    genre = None
    lowered = abstract.lower()
    for key in sorted(_GENRE_CANON, key=len, reverse=True):
        if key in lowered:
            genre = _GENRE_CANON[key]
            break
    return {
        "summary": abstract[:600],
        "heading": heading or None,
        "genre": genre,
        "score": 0.6,
        "provider": "duckduckgo",
    }


# --- public API ------------------------------------------------------------

def confirm_track(artist: str | None, title: str | None,
                  use_cache: bool = True) -> dict[str, Any]:
    """Confirm an artist/title guess on the web and infer a genre.

    Returns a dict with ``confirmed`` (bool), the corrected ``artist`` /
    ``title`` / ``album`` / ``year`` / ``genre`` when a provider supplied
    better values, ``confidence`` (0-1) and the ``sources`` consulted. Always
    returns a dict — network problems degrade to ``confirmed: False``.
    """
    artist = (artist or "").strip()
    title = (title or "").strip()
    out: dict[str, Any] = {
        "confirmed": False, "artist": None, "title": None, "album": None,
        "year": None, "genre": None, "confidence": 0.0, "sources": [],
    }
    if not title:
        return out

    key = f"{artist.casefold()}|{title.casefold()}"
    if use_cache:
        cached = cached_lookup(key)
        if cached is not None:
            return cached

    if not config.get("websearch.enabled", True):
        return out

    timeout = float(config.get("websearch.timeout_s", 12))
    headers = {"User-Agent": config.get("enrich.user_agent", "VirtualDJ/1.0")}
    best_score = 0.0

    try:
        with httpx.Client(timeout=timeout, headers=headers,
                          follow_redirects=True) as client:
            # Run the music databases concurrently so a lookup costs the *max* of
            # their latencies, not the sum. Each provider still respects its own
            # rate limit inside its throttle helper.
            from concurrent.futures import ThreadPoolExecutor

            def _call(provider):
                try:
                    return provider(client, artist, title)
                except Exception as exc:
                    log.debug("%s lookup failed for %s - %s: %s",
                              provider.__name__, artist, title, exc)
                    return {}

            with ThreadPoolExecutor(max_workers=2) as pool:
                hits = list(pool.map(_call, (_mb_search, _itunes_search)))
            for hit in hits:
                if not hit:
                    continue
                out["sources"].append(hit["provider"])
                score = float(hit.get("score") or 0.0)
                if score > best_score:
                    best_score = score
                    out["artist"] = hit.get("artist") or out["artist"]
                    out["title"] = hit.get("title") or out["title"]
                out["album"] = out["album"] or hit.get("album")
                out["year"] = out["year"] or hit.get("year")
                genre = canonical_genre(hit.get("genre"))
                if not genre:
                    for tag in hit.get("tags") or []:
                        genre = canonical_genre(tag)
                        if genre:
                            break
                out["genre"] = out["genre"] or genre

            # Only fall back to DDG when the music databases came up empty.
            if best_score < 0.55:
                try:
                    hit = _ddg_search(client, artist, title)
                except Exception as exc:
                    log.debug("ddg lookup failed: %s", exc)
                    hit = {}
                if hit:
                    out["sources"].append(hit["provider"])
                    out["genre"] = out["genre"] or canonical_genre(hit.get("genre"))
                    # A matching heading is weak evidence the pair is real.
                    heading = hit.get("heading") or ""
                    if heading and title and _similar(title, heading) > 0.6:
                        best_score = max(best_score, 0.6)
    except Exception as exc:
        log.debug("web confirmation client error: %s", exc)

    out["confidence"] = round(best_score, 3)
    out["confirmed"] = best_score >= 0.55
    store_lookup(key, out)
    return out


def health() -> dict[str, Any]:
    """Is web confirmation usable right now?"""
    if not config.get("websearch.enabled", True):
        return {"ok": False, "enabled": False, "error": "disabled in config"}
    try:
        with httpx.Client(timeout=6, follow_redirects=True) as client:
            resp = client.get(ITUNES_ROOT,
                              params={"term": "nirvana", "limit": 1, "media": "music"})
            resp.raise_for_status()
        return {"ok": True, "enabled": True}
    except Exception as exc:
        return {"ok": False, "enabled": True, "error": str(exc)}


__all__ = ["confirm_track", "canonical_genre", "health",
           "cached_lookup", "store_lookup"]
