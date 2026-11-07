"""HTTP client for the standalone TTS service.

The app knows nothing about speech engines: it asks the service what it can do
(``capabilities``) and posts text to ``/v1/speech``. Everything here is about
that contract — caching the capability document, mapping failures to a sentence
a human can act on, and never letting a TTS outage take the station down.

Service URL: ``tts.base_url`` in config (env ``VDJ_TTS_URL``), default
``http://127.0.0.1:9000``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)

CAPS_TTL_S = 60.0
HEALTH_TIMEOUT = 6.0
SPEECH_TIMEOUT = 300.0
INSTALL_TIMEOUT = 900.0

_lock = threading.Lock()
_caps: dict[str, Any] | None = None
_caps_at = 0.0
_caps_error: str | None = None


def base_url() -> str:
    """The service endpoint (config first, env default from config.DEFAULTS)."""
    return str(config.get("tts.base_url", "") or "").rstrip("/")


def _url(path: str) -> str:
    return f"{base_url()}{path}"


def _detail(resp: httpx.Response) -> str:
    """Best-effort human message from an error response."""
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:200] or f"HTTP {resp.status_code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)[:300]


def capabilities(force: bool = False) -> dict[str, Any] | None:
    """The engine capability document, cached for a minute.

    Returns ``None`` when the service cannot be reached *and* there is no cached
    copy — callers treat that as "TTS unavailable" rather than raising.
    """
    global _caps, _caps_at, _caps_error
    now = time.monotonic()
    with _lock:
        fresh = _caps is not None and (now - _caps_at) < CAPS_TTL_S
        if fresh and not force:
            return _caps
    if not base_url():
        with _lock:
            _caps_error = "no tts.base_url configured"
        return _caps
    try:
        resp = httpx.get(_url("/v1/capabilities"), timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - a dead service is a normal state
        with _lock:
            _caps_error = str(exc)
        log.debug("capabilities fetch failed: %s", exc)
        return _caps  # stale beats nothing
    with _lock:
        _caps = data
        _caps_at = now
        _caps_error = None
    return data


def invalidate() -> None:
    """Drop the cached capabilities (called after an install)."""
    global _caps_at
    with _lock:
        _caps_at = 0.0


def health() -> dict[str, Any]:
    """The service's own health, or a synthesised 'unreachable' answer."""
    url = base_url()
    if not url:
        return {"ok": False, "detail": "no tts.base_url configured"}
    try:
        resp = httpx.get(_url("/v1/health"), timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"TTS service unreachable at {url}: {exc}",
                "unreachable": True}


def voices() -> list[dict[str, Any]]:
    caps = capabilities() or {}
    return list(caps.get("voices") or [])


def params() -> list[dict[str, Any]]:
    caps = capabilities() or {}
    return list(caps.get("params") or [])


def languages() -> list[dict[str, str]]:
    caps = capabilities() or {}
    return list(caps.get("languages") or [])


def engine() -> dict[str, Any]:
    caps = capabilities() or {}
    return dict(caps.get("engine") or {})


def defaults() -> dict[str, Any]:
    caps = capabilities() or {}
    return dict(caps.get("defaults") or {})


def voice(voice_id: str | None) -> dict[str, Any] | None:
    for entry in voices():
        if entry.get("id") == voice_id:
            return entry
    return None


def installed_voice_ids() -> list[str]:
    return [v["id"] for v in voices() if v.get("installed")]


def default_voice() -> str:
    """A voice that can speak right now, if any.

    Order: the configured voice *when it is installed*, then the station's own
    documented pick (``DEFAULTS["dj"]["voice"]``), then the service's declared
    default, then any installed voice. That middle step is what a config carried
    over from a previous engine lands on: its voice id is dead, and the answer
    must be the station's own choice rather than whichever style the engine
    happens to list first (M1, alphabetically — a different voice from the one
    picked for its Russian).

    Last resorts: the configured voice if the service merely knows it (it can be
    installed), then the service's default/first entry. Returns ``""`` when the
    service offers nothing.
    """
    configured = str(config.get("dj.voice", "") or "")
    installed = installed_voice_ids()
    if configured and configured in installed:
        return configured
    station_pick = str((config.DEFAULTS.get("dj") or {}).get("voice") or "")
    for candidate in (station_pick, str(defaults().get("voice") or "")):
        if candidate and candidate in installed:
            return candidate
    if installed:
        return installed[0]
    if configured and voice(configured):
        return configured
    fallback = defaults().get("voice") or ""
    if fallback:
        return str(fallback)
    entries = voices()
    return entries[0]["id"] if entries else ""


def resolve_language(voice_entry: dict[str, Any] | None,
                     wanted: str | None = None) -> str:
    """The language to speak: the request, else config, else the voice default."""
    wanted = (wanted or config.get("dj.language") or "").strip().lower()
    langs = list((voice_entry or {}).get("languages") or [])
    if wanted and wanted in langs:
        return wanted
    if langs:
        return (voice_entry or {}).get("default_language") or langs[0]
    return wanted or "en"


def resolve_params(given: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge saved values over the service's declared defaults.

    Unknown keys are kept (the service ignores and reports them, so switching
    engines back and forth does not lose a setting), but only declared keys are
    sent as real values in the UI.
    """
    declared = params()
    out = {p["key"]: p.get("default") for p in declared if p.get("key")}
    saved = given if given is not None else dict(config.get("dj.tts_params") or {})
    for key, value in (saved or {}).items():
        out[key] = value
    return out


class TTSError(RuntimeError):
    """The service rejected a request or could not be reached."""


def install_voice(voice_id: str) -> dict[str, Any]:
    """Ask the service to make a voice usable (downloads on demand)."""
    resp = httpx.post(_url(f"/v1/voices/{voice_id}/install"),
                      timeout=INSTALL_TIMEOUT)
    if resp.status_code >= 400:
        raise TTSError(_detail(resp))
    invalidate()
    return resp.json()


def speech(text: str, voice: str | None = None, language: str | None = None,
           params: dict[str, Any] | None = None) -> tuple[bytes, dict[str, Any]]:
    """Render ``text``; return ``(wav_bytes, meta)`` from the response headers.

    Raises :class:`TTSError` with a readable message — the caller decides
    whether that means "skip this break" or "tell the user".
    """
    body = {
        "text": text,
        "voice": voice or None,
        "language": language or None,
        "params": params or None,
    }
    last_error: str | None = None
    for attempt in (1, 2):
        try:
            resp = httpx.post(_url("/v1/speech"), json=body, timeout=SPEECH_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - retried once below
            last_error = f"TTS service unreachable: {exc}"
            if attempt == 1:
                time.sleep(0.5)
                continue
            raise TTSError(last_error) from exc
        if resp.status_code >= 400:
            raise TTSError(_detail(resp))
        meta = {
            "engine": resp.headers.get("X-TTS-Engine"),
            "voice": resp.headers.get("X-TTS-Voice"),
            "language": resp.headers.get("X-TTS-Language"),
            "duration": float(resp.headers.get("X-TTS-Duration") or 0.0),
            "sample_rate": int(resp.headers.get("X-TTS-Sample-Rate") or 0),
            "render_ms": int(resp.headers.get("X-TTS-Render-Ms") or 0),
            "params_used": resp.headers.get("X-TTS-Params-Used"),
            "params_ignored": resp.headers.get("X-TTS-Params-Ignored"),
            "params_clamped": resp.headers.get("X-TTS-Params-Clamped"),
            "steps": resp.headers.get("X-TTS-Steps"),
        }
        return resp.content, meta
    raise TTSError(last_error or "TTS request failed")


def problem() -> str | None:
    """Why the DJ cannot speak right now, or ``None`` when it can.

    One sentence, actionable, and the same everywhere it is shown: the boot log,
    ``/api/health``, the DJ card and the prefetch worker.
    """
    url = base_url()
    if not url:
        return ("no TTS service configured — set tts.base_url "
                "(env VDJ_TTS_URL to the tts container)")
    service = health()
    if not service.get("ok"):
        detail = service.get("detail") or "the TTS service reports not ready"
        return f"TTS service at {url}: {detail}"
    caps = capabilities()
    if caps is None:
        return f"TTS service at {url} did not answer /v1/capabilities"
    available = installed_voice_ids()
    if not available:
        return ("the TTS service has no voice installed — run "
                "`python -m app.voices` or pick a voice in the DJ card")
    # Judge the voice that will actually speak, not the raw configured id: a
    # config carried over from a previous engine names a voice this one does not
    # offer, and refusing to speak over that would leave the station silent while
    # a perfectly good voice is resolved for it.
    usable = default_voice()
    if usable and usable in available:
        return None
    wanted = str(config.get("dj.voice", "") or "")
    return (f"the configured voice {wanted!r} is not installed on the TTS "
            f"service (have: {', '.join(available)})")


__all__ = [
    "base_url", "capabilities", "invalidate", "health", "voices", "params",
    "languages", "engine", "defaults", "voice", "installed_voice_ids",
    "default_voice", "resolve_language", "resolve_params", "install_voice",
    "speech", "problem", "TTSError",
]
