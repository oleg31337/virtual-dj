"""The DJ brain: turns track metadata into a spoken intro.

Pipeline per track:
    metadata (+ enrichment facts) -> LLM script -> Piper TTS -> mp3 segment

Both stages degrade gracefully: with no LLM we fall back to a templated line,
and if TTS fails the stream simply skips the DJ break rather than stalling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from . import config, db, enrich

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = (
    "You are the voice of a radio station. You introduce songs on air. "
    "Write ONLY the words to be spoken aloud: no stage directions, no track "
    "listings, no markdown, no emoji, no quotation marks around the whole "
    "reply, and never mention that you are an AI. Keep it natural, spoken "
    "English. "
    "STRICT FACT RULE: state ONLY facts that appear verbatim in the "
    "information block you are given. Never guess or embellish a song's "
    "history, chart position, recording story, meaning, or origin. If the "
    "information block is thin, say something short and atmospheric about the "
    "title, artist and genre instead of inventing history. A short accurate "
    "intro is always better than a longer invented one."
)


def piper_binary() -> str | None:
    """Locate the piper executable (venv bin first, then PATH)."""
    candidate = Path(sys.executable).parent / "piper"
    if candidate.exists():
        return str(candidate)
    return shutil.which("piper")


def voice_model_path(voice: str | None = None) -> Path | None:
    name = voice or config.get("dj.voice", "en_US-amy-medium")
    # Allow tests/installs to point the voice library elsewhere (e.g. system
    # share dir) without duplicating multi-hundred-MB models.
    voices_dir = Path(config.get("voices_dir", config.VOICES_DIR))
    path = voices_dir / f"{name}.onnx"
    return path if path.exists() else None


def available_voices() -> list[str]:
    voices_dir = Path(config.get("voices_dir", config.VOICES_DIR))
    if not voices_dir.is_dir():
        return []
    return sorted(p.stem for p in voices_dir.glob("*.onnx"))


def _clean_script(text: str, max_sentences: int) -> str:
    text = _THINK_RE.sub("", text or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_#`>\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    if not text:
        return ""
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    return " ".join(sentences[: max(1, max_sentences)]).strip()


def fallback_script(track: dict[str, Any]) -> str:
    artist = (track.get("artist") or "").strip()
    title = (track.get("title") or "").strip() or "this next one"
    year = (track.get("year") or "").strip()
    if artist and year:
        return f"Coming up next: {title}, by {artist}, from {year}."
    if artist:
        return f"Here's {title}, by {artist}."
    return f"Up next, {title}."


def _facts_block(track: dict[str, Any], facts: dict[str, Any]) -> str:
    lines = []
    unknown = []
    for label, key in (("Title", "title"), ("Artist", "artist"),
                       ("Album", "album"), ("Genre", "genre"), ("Year", "year")):
        value = (track.get(key) or "").strip() if track.get(key) else ""
        if value:
            lines.append(f"{label}: {value}")
        elif label != "Title":
            unknown.append(label.lower())
    # NOTE: neither MusicBrainz "first-release-date" nor "release_title" are
    # surfaced to the model. Both describe one arbitrary *recording/release*
    # row (often a later compilation or live cut), not the song's origin, and
    # small models confidently restate them as "originally released in ...".
    # The ID3 year above is far more trustworthy for a personal collection.
    if facts.get("tags"):
        lines.append("Style tags: " + ", ".join(facts["tags"]))
    if facts.get("credited_artists"):
        lines.append("Credited artists: " + ", ".join(facts["credited_artists"]))
    if facts.get("artist_summary"):
        lines.append("About the artist: " + facts["artist_summary"][:800])
    if unknown:
        lines.append(
            "UNKNOWN (do not mention, do not guess): " + ", ".join(unknown)
        )
    return "\n".join(lines)


def generate_script(track: dict[str, Any], previous: dict[str, Any] | None = None) -> str:
    """Ask the local LLM for an on-air intro; fall back to a template."""
    max_sentences = int(config.get("dj.max_sentences", 3))
    facts: dict[str, Any] = {}
    try:
        facts = enrich.enrich_track(track)
    except Exception as exc:
        log.debug("enrichment failed: %s", exc)

    if not config.get("llm.enabled", True):
        return fallback_script(track)

    style = config.get("dj.style", "warm, witty late-night radio host")
    prev_line = ""
    if previous and (previous.get("title") or previous.get("artist")):
        prev_line = (
            f"\nThe song that just finished was "
            f"{previous.get('title') or 'unknown'} by "
            f"{previous.get('artist') or 'unknown'}. You may briefly reference it."
        )

    user_prompt = (
        f"Persona: {style}.\n"
        f"Write at most {max_sentences} sentences introducing the next song.\n"
        f"Mention one genuinely interesting fact ONLY if the information block "
        f"below directly supports it. Do NOT state any release date or year "
        f"that is not listed below.{prev_line}\n\n"
        f"Information about the next song:\n"
        f"{_facts_block(track, facts)}\n"
    )

    base_url = str(config.get("llm.base_url", "")).rstrip("/")
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": float(config.get("llm.temperature", 0.7)),
                    "num_predict": 220,
                },
            },
        )
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "")
        script = _clean_script(content, max_sentences)
        if script:
            return script
        log.warning("LLM returned an empty script; using fallback")
    except Exception as exc:
        log.warning("LLM script generation failed (%s); using fallback", exc)
    return fallback_script(track)


def _cache_path(text: str, voice: str, speed: float) -> Path:
    digest = hashlib.sha256(
        f"{voice}|{speed}|{text}".encode("utf-8")
    ).hexdigest()[:32]
    return config.DJ_CACHE_DIR / f"{digest}.mp3"


def synthesize(text: str, voice: str | None = None,
               speed: float | None = None) -> Path | None:
    """Render ``text`` to an mp3 with Piper. Returns None on failure."""
    text = (text or "").strip()
    if not text:
        return None

    voice = voice or config.get("dj.voice", "en_US-amy-medium")
    speed = float(speed if speed is not None else config.get("dj.speed", 1.0))
    speed = max(0.5, min(speed, 2.0))
    out_path = _cache_path(text, voice, speed)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    binary = piper_binary()
    model = voice_model_path(voice)
    if not binary or not model:
        log.warning("piper unavailable (binary=%s, model=%s)", binary, model)
        return None

    config.DJ_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "dj.wav"
        # Piper's length_scale is inverse to speed.
        length_scale = round(1.0 / speed, 4)
        cmd = [binary, "-m", str(model), "-f", str(wav_path),
               "--length_scale", str(length_scale)]
        try:
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"),
                capture_output=True, timeout=180, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.error("piper invocation failed: %s", exc)
            return None
        if proc.returncode != 0 or not wav_path.exists():
            log.error("piper failed rc=%s: %s", proc.returncode,
                      proc.stderr.decode("utf-8", "replace")[:400])
            return None

        gain_db = float(config.get("dj.gain_db", 0.0))
        filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"]
        if abs(gain_db) > 0.01:
            filters.append(f"volume={gain_db}dB")
        tmp_mp3 = Path(tmpdir) / "dj.mp3"
        ff = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(wav_path), "-af", ",".join(filters),
             "-ar", str(config.get("stream.sample_rate", 44100)),
             "-ac", str(config.get("stream.channels", 2)),
             "-b:a", f"{int(config.get('stream.bitrate_kbps', 128))}k",
             "-write_xing", "0", "-id3v2_version", "0",
             str(tmp_mp3)],
            capture_output=True, timeout=180, check=False,
        )
        if ff.returncode != 0 or not tmp_mp3.exists():
            log.error("ffmpeg dj encode failed: %s",
                      ff.stderr.decode("utf-8", "replace")[:400])
            return None
        shutil.move(str(tmp_mp3), str(out_path))
    return out_path


def audio_duration(path: Path) -> float:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, timeout=30, check=False,
        )
        return float(proc.stdout.decode().strip() or 0.0)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return 0.0


def prepare_break(track: dict[str, Any],
                  previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Generate script + audio for one DJ break. Returns None if unavailable."""
    script = generate_script(track, previous)
    if not script:
        return None
    audio = synthesize(script)
    if audio is None:
        return None
    duration = audio_duration(audio)
    try:
        conn = db.connect()
        conn.execute(
            "INSERT INTO dj_scripts(track_id, text, audio_path, duration) "
            "VALUES(?,?,?,?)",
            (track.get("id"), script, str(audio), duration),
        )
        conn.commit()
    except Exception as exc:
        log.debug("could not persist dj script: %s", exc)
    return {"text": script, "audio_path": str(audio), "duration": duration}


def recent_scripts(limit: int = 20) -> list[dict[str, Any]]:
    rows = db.connect().execute(
        "SELECT d.id, d.text, d.duration, d.created_at, t.title, t.artist "
        "FROM dj_scripts d LEFT JOIN tracks t ON t.id = d.track_id "
        "ORDER BY d.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return db.rows_to_dicts(rows)


def llm_health() -> dict[str, Any]:
    base_url = str(config.get("llm.base_url", "")).rstrip("/")
    if not base_url:
        return {"ok": False, "error": "no base_url configured"}
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", [])]
        wanted = config.get("llm.model")
        return {"ok": True, "models": models, "model_present": wanted in models}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tts_health() -> dict[str, Any]:
    return {
        "ok": bool(piper_binary() and voice_model_path()),
        "binary": piper_binary(),
        "voice": config.get("dj.voice"),
        "voices_available": available_voices(),
    }


__all__ = [
    "generate_script", "synthesize", "prepare_break", "fallback_script",
    "available_voices", "llm_health", "tts_health", "recent_scripts",
    "audio_duration", "json",
]
