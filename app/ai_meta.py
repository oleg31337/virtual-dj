"""Local-LLM metadata helpers: genre inference and corrupt-tag name recovery.

Both jobs lean on the same local model (Ollama, configured under ``llm.*``):

* **Genre** — when web search can't place a track, the LLM is the second
  opinion. It sees the artist/title/album/path and is asked for *one* canonical
  genre from a controlled list, so the result stays usable as a playlist filter.
* **Name recovery** — when ID3 tags are corrupt (mojibake, placeholders) we
  recover artist/title from the filename/folder, then let the LLM normalise and
  confirm the guess (e.g. split "01artist-title" into the right pair, or
  transliterate a Cyrillic name for the English-speaking DJ).

Every result is cached in SQLite keyed on the inputs, so a re-scan of a 13k
library re-uses previous answers instead of hammering the model.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

import httpx

from . import config, db

log = logging.getLogger(__name__)

# Canonical genres the LLM is allowed to choose from. Must stay in sync with
# websearch._GENRE_CANON's value set so a web-derived and an AI-derived genre
# land in the same bucket.
GENRE_CHOICES = [
    "Rock", "Metal", "Pop", "Electronic", "Hip Hop", "R&B", "Soul", "Funk",
    "Jazz", "Blues", "Classical", "Country", "Folk", "Reggae", "Latin",
    "World", "Ambient", "Soundtrack", "Punk", "Indie", "Alternative",
    "Grunge", "Disco", "New Age", "Instrumental", "Experimental",
    "Progressive", "Psychedelic", "Gothic", "Trip Hop", "House",
]

# Plausible free-form genres that map onto a canonical bucket, as a fallback
# when the model ignores the list.
_GENRE_WORDS = {g.lower(): g for g in GENRE_CHOICES}

_SYSTEM_GENRE = (
    "You are a music-tagging assistant. Given a track's available metadata, "
    "reply with EXACTLY ONE word: the single best-fitting genre chosen from "
    "this list only — " + ", ".join(GENRE_CHOICES) + ". "
    "No explanation, no punctuation. If unsure, pick the closest broad match."
)

_SYSTEM_RECOVER = (
    "You recover metadata for an MP3 whose ID3 tags are corrupt. You are given "
    "a guessed artist and title (often derived from the filename). Return ONLY "
    "valid JSON of the form {\"artist\": str, \"title\": str, \"album\": str|null, "
    "\"language\": \"latin\"|\"cyrillic\"|\"other\", \"confident\": bool}. "
    "Rules: keep the artist and title clean and split correctly; if the name is "
    "in Cyrillic, provide a romanised/latin-spelled artist and title for an "
    "English-speaking DJ but keep the original meaning; set confident=false if "
    "the input is too damaged to trust. No prose outside the JSON."
)

_lock = threading.Lock()


def _ollama_chat(system: str, user: str, temperature: float = 0.2,
                 max_tokens: int = 64) -> str | None:
    """One chat completion against the local model. Returns text or None."""
    if not config.get("llm.enabled", True):
        return None
    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    if not base_url:
        return None
    model = config.get("llm.model", "qwen3.5:9b")
    timeout = float(config.get("llm.timeout_s", 120))
    try:
        resp = httpx.post(
            f"{base_url}/api/chat",
            timeout=timeout,
            json={
                "model": model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "") or None
    except Exception as exc:
        log.debug("local LLM call failed: %s", exc)
        return None


def _cache_get(table: str, key: str) -> dict[str, Any] | None:
    try:
        row = db.connect().execute(
            f"SELECT result FROM {table} WHERE query = ?", (key,)
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        return json.loads(row["result"])
    except (TypeError, json.JSONDecodeError):
        return None


def _cache_put(table: str, key: str, result: dict[str, Any]) -> None:
    try:
        conn = db.connect()
        conn.execute(
            f"INSERT INTO {table}(query, result) VALUES(?,?) "
            f"ON CONFLICT(query) DO UPDATE SET result=excluded.result, "
            f"fetched_at=strftime('%s','now')",
            (key, json.dumps(result)),
        )
        conn.commit()
    except Exception as exc:
        log.debug("could not cache AI result: %s", exc)


def _canon(text: str) -> str | None:
    if not text:
        return None
    t = text.strip().strip("*\"'`").strip().title()
    low = t.lower()
    if low in _GENRE_WORDS:
        return _GENRE_WORDS[low]
    for choice in GENRE_CHOICES:
        if choice.lower() in low:
            return choice
    # Single plausible capitalized word.
    if re.fullmatch(r"[A-Za-z][A-Za-z ]{1,20}", t):
        return t
    return None


def infer_genre(artist: str | None, title: str | None, album: str | None = None,
                path: str | None = None, use_cache: bool = True) -> str | None:
    """Best-effort genre from the local LLM, constrained to GENRE_CHOICES.

    Returns a canonical genre string or None. Cheap and cached.
    """
    title = (title or "").strip()
    artist = (artist or "").strip()
    if not (title or artist):
        return None
    key = f"g|{artist.casefold()}|{title.casefold()}|{(album or '').casefold()}"
    if use_cache:
        cached = _cache_get("ai_lookups", key)
        if cached is not None:
            return cached.get("genre")
    if not config.get("ai.free_text_genre", True):
        return None
    bits = []
    if artist:
        bits.append(f"Artist: {artist}")
    if title:
        bits.append(f"Title: {title}")
    if album:
        bits.append(f"Album: {album}")
    if path:
        bits.append(f"File: {path}")
    text = _ollama_chat(_SYSTEM_GENRE, "\n".join(bits), temperature=0.1,
                        max_tokens=24)
    genre = _canon(text or "") if text else None
    if use_cache:
        _cache_put("ai_lookups", key, {"genre": genre})
    return genre


def recover_names(artist_guess: str | None, title_guess: str | None,
                 album_guess: str | None = None,
                 path: str | None = None,
                 use_cache: bool = True) -> dict[str, Any]:
    """Recover clean artist/title/album (+ language) from a messy guess.

    Used when ID3 tags are corrupt: the path guess is fed to the LLM which
    splits/normalises it and, for Cyrillic names, supplies a romanised form so
    the English-speaking DJ can still announce it. Returns a dict with keys
    ``artist``, ``title``, ``album`` (each str|None), ``language`` and
    ``confident`` (bool). Always returns a dict.
    """
    artist_guess = (artist_guess or "").strip()
    title_guess = (title_guess or "").strip()
    if not (artist_guess or title_guess):
        return {"artist": None, "title": None, "album": None,
                "language": "other", "confident": False}
    key = f"n|{artist_guess.casefold()}|{title_guess.casefold()}|{(album_guess or '').casefold()}"
    if use_cache:
        cached = _cache_get("ai_lookups", key)
        if cached is not None:
            return cached
    bits = []
    if artist_guess:
        bits.append(f"Artist guess: {artist_guess}")
    if title_guess:
        bits.append(f"Title guess: {title_guess}")
    if album_guess:
        bits.append(f"Album guess: {album_guess}")
    if path:
        bits.append(f"Path: {path}")
    text = _ollama_chat(_SYSTEM_RECOVER, "\n".join(bits), temperature=0.2,
                        max_tokens=160)
    result: dict[str, Any] = {
        "artist": None, "title": None, "album": None,
        "language": "other", "confident": False,
    }
    if text:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                result["artist"] = (data.get("artist") or None) if data.get("artist") else None
                result["title"] = (data.get("title") or None) if data.get("title") else None
                result["album"] = (data.get("album") or None) if data.get("album") else None
                result["language"] = str(data.get("language") or "other")
                result["confident"] = bool(data.get("confident"))
            except (ValueError, TypeError):
                log.debug("AI name recovery returned unparseable JSON: %s", text[:120])
    if use_cache:
        _cache_put("ai_lookups", key, result)
    return result


def health() -> dict[str, Any]:
    """Is the local LLM reachable for metadata work?"""
    if not config.get("llm.enabled", True):
        return {"ok": False, "enabled": False, "error": "llm disabled in config"}
    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    if not base_url:
        return {"ok": False, "enabled": True, "error": "no base_url configured"}
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return {"ok": True, "enabled": True}
    except Exception as exc:
        return {"ok": False, "enabled": True, "error": str(exc)}


__all__ = ["infer_genre", "recover_names", "health", "GENRE_CHOICES"]
