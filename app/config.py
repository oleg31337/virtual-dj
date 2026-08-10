"""Configuration handling for Virtual DJ.

Settings live in a JSON file under the data directory (never in code, never in
git). Secrets, if any are ever needed, are read from the environment only.
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VDJ_DATA_DIR", BASE_DIR / "data"))
WEB_DIR = BASE_DIR / "web"
CACHE_DIR = DATA_DIR / "cache"
DJ_CACHE_DIR = CACHE_DIR / "dj"
VOICES_DIR = DATA_DIR / "voices"
CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = Path(os.environ.get("VDJ_DB_PATH", DATA_DIR / "vdj.sqlite3"))

DEFAULTS: dict[str, Any] = {
    # Where to look for music. Set it in the web UI (Library panel), or
    # override at first boot with VDJ_MUSIC_DIR.
    "music_dir": os.environ.get("VDJ_MUSIC_DIR", str(Path.home() / "Music")),
    "stream": {
        "bitrate_kbps": 128,
        "sample_rate": 44100,
        "channels": 2,
        "station_name": "Virtual DJ",
    },
    "dj": {
        "enabled": True,
        # Speak before 1 out of every N tracks. 1 = every track.
        "every_n_tracks": 3,
        "max_sentences": 3,
        "style": "warm, witty late-night radio host",
        "voice": "en_US-amy-medium",
        "speed": 1.0,
        # Gain applied to the DJ voice segment, in dB.
        "gain_db": 0.0,
        "intro_music_duck_db": -8.0,
        "prefetch_depth": 3,
    },
    "llm": {
        "enabled": True,
        # Ollama endpoint. Override per-install without touching source via
        # the VDJ_OLLAMA_URL env var or "llm.base_url" in data/config.json.
        "base_url": os.environ.get("VDJ_OLLAMA_URL", "http://127.0.0.1:11434"),
        "model": os.environ.get("VDJ_OLLAMA_MODEL", "qwen3.5:9b"),
        "timeout_s": 120,
        "temperature": 0.7,
    },
    "ai": {
        # Use the local LLM to fill genres web search couldn't, and to recover
        # artist/title from corrupt tags. "unknown genre" is never acceptable
        # for a playable track, so the AI is the guaranteed last resort.
        "free_text_genre": True,
        "name_recovery": True,
    },
    "enrich": {
        # Look up extra facts on MusicBrainz / Wikipedia.
        "enabled": True,
        "user_agent": (
            "VirtualDJ/1.0 (https://github.com/virtual-dj/virtual-dj; "
            "self-hosted personal radio)"
        ),
        "timeout_s": 15,
    },
    "websearch": {
        # Confirm filename-guessed tracks on the web and infer their genre.
        # Uses keyless public APIs (MusicBrainz, iTunes, DuckDuckGo) and
        # caches every answer, so a rescan costs nothing for known tracks.
        "enabled": True,
        "timeout_s": 12,
    },
    "playback": {
        "shuffle": True,
        # Active filters applied when auto-filling the queue.
        "genres": [],
        "artists": [],
        "search": "",
        "program": {
            # Group tracks into themed "programs" the way a real DJ would: a run
            # of tracks that share a genre/era/mood, announced together, then a
            # DJ break before the next program switches the vibe. Adjustable.
            "enabled": True,
            # Tracks per program before a DJ break announces the switch.
            "size": 6,
            # How programs are themed: "genre" (same genre), "artist" (same
            # artist run), or "decade" (same era).
            "strategy": "genre",
        },
    },
}

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dirs() -> None:
    for path in (DATA_DIR, CACHE_DIR, DJ_CACHE_DIR, VOICES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_config(force: bool = False) -> dict[str, Any]:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not force:
            return deepcopy(_CACHE)
        ensure_dirs()
        stored: dict[str, Any] = {}
        if CONFIG_PATH.exists():
            try:
                stored = json.loads(CONFIG_PATH.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                stored = {}
        _CACHE = _deep_merge(DEFAULTS, stored)
        return deepcopy(_CACHE)


def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``patch`` into the stored config and persist it."""
    global _CACHE
    with _LOCK:
        current = load_config()
        merged = _deep_merge(current, patch)
        ensure_dirs()
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(merged, indent=2), "utf-8")
        tmp.replace(CONFIG_PATH)
        _CACHE = merged
        return deepcopy(merged)


def get(dotpath: str, default: Any = None) -> Any:
    node: Any = load_config()
    for part in dotpath.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
