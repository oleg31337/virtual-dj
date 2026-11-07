"""The universal TTS HTTP API.

One container, one contract: the app asks ``/v1/capabilities`` what exists
(voices, languages, parameters) and posts text + values to ``/v1/speech``. No
engine detail leaks through these routes — swapping ``TTS_ENGINE`` changes the
answers, not the shapes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config, schema
from .engines import EngineError, build_engine

logging.basicConfig(
    level=getattr(logging, config.log_level().upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tts")

STATE: dict[str, Any] = {
    "engine": None,
    "boot_note": "starting",
    "boot_error": None,
    "started": time.time(),
    "renders": 0,
    "chars_rendered": 0,
    "last_render_ms": None,
}
# Synthesis is CPU-bound and each render allocates a few hundred MB; two at a
# time keeps a burst of requests from spiking the box without serialising the
# station behind one slow line.
_SYNTH = threading.Semaphore(2)
_LOCK = threading.Lock()


def _boot() -> None:
    engine = STATE["engine"]
    try:
        STATE["boot_note"] = "loading model"
        engine.ensure_model(progress=lambda msg: STATE.__setitem__("boot_note", msg))
        STATE["boot_note"] = "ready"
        log.info("engine %s ready: %d voices", engine.id, len(engine.voices()))
    except Exception as exc:  # noqa: BLE001 - reported through /v1/health
        STATE["boot_error"] = str(exc)
        STATE["boot_note"] = "model not ready"
        log.error("engine boot failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001, ARG001 - FastAPI signature
    try:
        STATE["engine"] = build_engine(config)
    except Exception as exc:  # noqa: BLE001 - a bad TTS_ENGINE must not hang boot
        STATE["boot_error"] = str(exc)
        STATE["boot_note"] = "misconfigured"
        log.error("cannot build engine: %s", exc)
    if STATE["engine"] is not None and config.autoload():
        # Downloading + loading takes a minute on a cold volume; do it off the
        # request path so /v1/health answers immediately and can say why.
        threading.Thread(target=_boot, name="tts-boot", daemon=True).start()
    yield


app = FastAPI(title="Virtual DJ TTS service", version=config.API_VERSION,
              lifespan=lifespan)

ENGINE_NOT_READY = ("model_not_ready",
                    "the TTS model is not loaded yet; see /v1/health")


def _err(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    payload = {"code": code, "message": message}
    payload.update(extra)
    return JSONResponse(status_code=status, content={"detail": payload})


def _engine():
    engine = STATE["engine"]
    if engine is None:
        raise EngineError(STATE["boot_error"] or "no engine configured")
    return engine


def _voice_map(engine) -> dict[str, dict[str, Any]]:  # noqa: ANN001
    return {v["id"]: v for v in engine.voices()}


def _resolve_voice(engine, voice: str | None) -> dict[str, Any]:  # noqa: ANN001
    voices = _voice_map(engine)
    if voice:
        if voice not in voices:
            raise LookupError(voice)
        return voices[voice]
    wanted = config.default_voice()
    if wanted and wanted in voices:
        return voices[wanted]
    for entry in voices.values():
        if entry.get("installed"):
            return entry
    raise LookupError("<none installed>")


def _default_language(engine, voice_entry: dict[str, Any]) -> str:  # noqa: ANN001
    wanted = config.default_language()
    if wanted and wanted in voice_entry.get("languages", []):
        return wanted
    return voice_entry.get("default_language") or engine.advertised_languages[0]


# --- routes ---------------------------------------------------------------

@app.get("/")
def index() -> dict[str, Any]:
    engine = STATE["engine"]
    return {
        "service": config.SERVICE_NAME,
        "api_version": config.API_VERSION,
        "engine": getattr(engine, "id", None),
        "docs": "/docs",
        "health": "/v1/health",
        "capabilities": "/v1/capabilities",
    }


@app.get("/v1/health")
def health() -> dict[str, Any]:
    engine = STATE["engine"]
    if engine is None:
        return {
            "ok": False,
            "engine": {"id": config.engine_id(), "name": config.engine_id()},
            "model_ready": False,
            "detail": STATE["boot_error"] or "engine not built",
            "voices_ready": 0,
            "note": STATE["boot_note"],
            "renders": STATE["renders"],
            "chars_rendered": STATE["chars_rendered"],
            "last_render_ms": STATE["last_render_ms"],
            "uptime_s": round(time.time() - STATE["started"], 1),
        }
    ready, why = engine.model_ready()
    voices = engine.voices()
    installed = [v["id"] for v in voices if v.get("installed")]
    detail = None
    if not ready:
        detail = why or STATE["boot_error"] or "model not ready"
    elif not installed:
        detail = "no voice is installed"
    return {
        "ok": ready and bool(installed),
        "engine": {
            "id": engine.id, "name": engine.name, "version": engine.version,
            "license": engine.license, "runtime": engine.runtime,
        },
        "model_ready": ready,
        "voices_ready": len(installed),
        "detail": detail,
        "note": STATE["boot_note"],
        "steps": config.steps(),
        "renders": STATE["renders"],
        "chars_rendered": STATE["chars_rendered"],
        "last_render_ms": STATE["last_render_ms"],
        "uptime_s": round(time.time() - STATE["started"], 1),
    }


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    engine = _engine()
    voices = engine.voices()
    try:
        entry = _resolve_voice(engine, None)
    except LookupError:
        entry = voices[0] if voices else {"id": None, "languages": []}
    return {
        "api": {"version": config.API_VERSION, "service": config.SERVICE_NAME},
        "engine": {
            "id": engine.id, "name": engine.name, "version": engine.version,
            "license": engine.license, "runtime": engine.runtime,
        },
        "defaults": {
            "voice": entry.get("id"),
            "language": _default_language(engine, entry) if voices else None,
            "params": schema.defaults(engine.params()),
        },
        "languages": engine.languages(),
        "voices": voices,
        "params": engine.params(),
        "output": {"formats": ["wav"], "sample_rates": [engine.sample_rate()],
                   "channels": [1]},
        "text": engine.text_support(),
        "quality": {"steps": config.steps(), "fixed": True},
    }


@app.post("/v1/voices/{voice_id}/install")
def install_voice(voice_id: str) -> Any:
    engine = _engine()
    if voice_id not in _voice_map(engine):
        return _err(404, "unknown_voice",
                    f"voice {voice_id!r} is not offered by {engine.id}",
                    available=sorted(_voice_map(engine)))
    try:
        return engine.install_voice(voice_id)
    except KeyError:
        return _err(404, "unknown_voice", f"voice {voice_id!r} does not exist")
    except EngineError as exc:
        return _err(503, "install_failed", str(exc))
    except Exception as exc:  # noqa: BLE001 - network/download failures
        log.exception("install failed")
        return _err(503, "install_failed", str(exc))


class SpeechRequest(BaseModel):
    text: str
    voice: str | None = None
    language: str | None = None
    params: dict[str, Any] | None = None


@app.post("/v1/speech")
def speech(req: SpeechRequest) -> Any:
    engine = _engine()
    text = (req.text or "").strip()
    if not text:
        return _err(400, "bad_request", "text is empty")
    limit = config.max_text_chars()
    if len(text) > limit:
        return _err(400, "bad_request",
                    f"text is {len(text)} characters; the limit is {limit}")

    try:
        entry = _resolve_voice(engine, req.voice)
    except LookupError:
        return _err(404, "unknown_voice",
                    f"voice {req.voice!r} is not offered by {engine.id}",
                    available=sorted(_voice_map(engine)))

    language = (req.language or _default_language(engine, entry)).lower()
    if language not in entry.get("languages", []):
        return _err(400, "bad_request",
                    f"voice {entry['id']!r} does not speak {language!r}",
                    languages=entry.get("languages", []))
    if not entry.get("installed"):
        return _err(409, "voice_not_installed",
                    f"voice {entry['id']!r} is not installed",
                    install=f"/v1/voices/{entry['id']}/install")

    try:
        resolved = schema.resolve(engine.params(), req.params)
    except schema.ParamError as exc:
        return _err(422, "invalid_param", str(exc), param=exc.key)

    ready, why = engine.model_ready()
    if not ready:
        return _err(503, *ENGINE_NOT_READY, reason=why)

    started = time.monotonic()
    with _SYNTH:
        try:
            samples, sample_rate = engine.synthesize(
                text, entry["id"], language, resolved.values)
        except EngineError as exc:
            return _err(503, "model_not_ready", str(exc))
        except ValueError as exc:
            return _err(400, "bad_request", str(exc))
        except Exception as exc:  # noqa: BLE001 - a render failure is a 500
            log.exception("synthesis failed")
            return _err(500, "render_failed", str(exc))
    elapsed_ms = int((time.monotonic() - started) * 1000)
    duration = round(len(samples) / float(sample_rate), 3) if sample_rate else 0.0

    with _LOCK:
        STATE["renders"] += 1
        STATE["chars_rendered"] += len(text)
        STATE["last_render_ms"] = elapsed_ms

    headers = {
        "X-TTS-Engine": engine.id,
        "X-TTS-Voice": str(entry["id"]),
        "X-TTS-Language": language,
        "X-TTS-Duration": f"{duration:.3f}",
        "X-TTS-Sample-Rate": str(sample_rate),
        "X-TTS-Render-Ms": str(elapsed_ms),
        "X-TTS-Params-Used": json.dumps(resolved.values, sort_keys=True),
        "X-TTS-Params-Ignored": json.dumps(sorted(resolved.ignored)),
        "X-TTS-Params-Clamped": json.dumps(sorted(resolved.clamped)),
        "X-TTS-Steps": str(config.steps()),
    }
    from .engines.base import wav_bytes
    return Response(content=wav_bytes(samples, sample_rate),
                    media_type="audio/wav", headers=headers)


@app.exception_handler(EngineError)
def _engine_error(_request: Request, exc: EngineError) -> JSONResponse:
    return _err(503, "model_not_ready", str(exc))
