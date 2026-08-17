"""FastAPI application: stream endpoint, REST control API, live WebSocket."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, dj, library, websearch, ai_meta
from . import icecast as icecast_mod
from .scheduler import SCHEDULER
from .stream import BROADCASTER

log = logging.getLogger(__name__)

_ws_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _push_state(state: dict[str, Any]) -> None:
    """Called from broadcaster threads; hop onto the event loop to fan out."""
    if _loop is None or _loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast_ws(state), _loop)
    except RuntimeError:
        pass


async def _broadcast_ws(state: dict[str, Any]) -> None:
    payload = {"type": "state", "data": state}
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    config.ensure_dirs()
    db.init_db()

    stats = library.library_stats()
    if stats["total"] == 0:
        log.info("library empty, starting initial scan")
        library.scan_in_background()

    SCHEDULER.ensure_filled(10)
    SCHEDULER.start()
    BROADCASTER.on_change(_push_state)
    BROADCASTER.start()
    icecast_mod.PUSHER.start()
    try:
        yield
    finally:
        icecast_mod.PUSHER.stop()
        BROADCASTER.stop()
        SCHEDULER.stop()


app = FastAPI(title="Virtual DJ", version="1.0.0", lifespan=lifespan)


# --- stream ----------------------------------------------------------------
#
# IMPORTANT (worked out under test): this endpoint must NOT return a Starlette
# StreamingResponse. Starlette races the response body against a background
# disconnect-watcher that treats a client that is connected-but-idle (the
# normal state for a live radio listener who is not sending requests) as
# "disconnected" and cancels the generator after the first chunk. The raw
# ASGI path below yields the same MP3 bytes with none of that behaviour; a
# healthy listener keeps the connection until it truly goes away.

class _RawStreamResponse(Response):
    """ASGI response that sends chunks directly, with no disconnect-watcher.

    Starlette's StreamingResponse races the body against a background watcher
    that cancels healthy-but-idle listeners; this class avoids that entirely.

    The broadcaster already paces the MP3 at real time (with a multi-second
    cushion in each listener's queue), so the job here is simply to relay
    whatever audio is buffered, promptly and continuously, in modest chunks.
    We deliberately do NOT wait to coalesce a large blob: a 1-second clump
    followed by a 1-second silence is exactly what makes small-buffer players
    (Winamp, VLC) underrun and "chop". Relaying steadily keeps the client's
    jitter buffer full.
    """

    def __init__(self, listener, station: str, bitrate: int):
        self.listener = listener
        self.station = station
        self.bitrate = bitrate
        super().__init__(media_type="audio/mpeg")

    async def __call__(self, scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"audio/mpeg"),
                (b"cache-control", b"no-cache, no-store"),
                (b"pragma", b"no-cache"),
                (b"icy-name", self.station.encode("latin-1", "replace")),
                (b"icy-genre", b"Various"),
                (b"icy-br", str(self.bitrate).encode()),
                (b"icy-pub", b"0"),
                (b"accept-ranges", b"none"),
            ],
        })

        # Watch the receive channel for a *real* http.disconnect message (the
        # client is gone or went away). This is the correct way to notice
        # disconnects: Starlette's alternative — polling is_disconnected() from
        # the body loop — fires on any idle receive channel, which is the normal
        # state for a live listener, and kills healthy connections.
        async def disconnect_watcher():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return

        loop = asyncio.get_running_loop()
        watcher = asyncio.create_task(disconnect_watcher())
        try:
            # Relay buffered audio continuously in modest chunks. The
            # broadcaster's real-time pacing + cushion means data is almost
            # always available, so this loop drains it as fast as it arrives
            # with no artificial waits -- a smooth real-time trickle.
            while not watcher.done():
                chunk = await loop.run_in_executor(None, _get_chunk, self.listener)
                if chunk is None:
                    continue
                if chunk == b"":
                    break
                await send({"type": "http.response.body", "body": chunk,
                            "more_body": True})
        finally:
            watcher.cancel()
            BROADCASTER.remove_listener(self.listener)


# Size of one relayed body chunk. Small enough to keep the client's buffer
# topped up continuously (no multi-second silence between clumps), large enough
# to avoid per-byte syscall/throttle overhead. At 128 kbps this is ~0.25-0.5 s
# of audio -- well within any player's jitter buffer.
SEND_CHUNK = 4096


def _get_chunk(listener) -> bytes | None:
    """Return buffered audio promptly, in modest chunks.

    Returns bytes to send, ``None`` if nothing is buffered yet (caller retries
    without emitting a body frame), or ``b""`` when the listener is closing.
    Does NOT wait to accumulate a large blob -- steady, continuous delivery is
    what keeps weak-buffer players (Winamp/VLC) from stuttering.
    """
    import queue as _q

    try:
        # Return one modest chunk as soon as it's available. The broadcaster
        # keeps a multi-second cushion in the queue, so in steady state there
        # is almost always an item ready -- we relay it immediately and let
        # the next call fetch the next one, producing a smooth real-time
        # trickle rather than periodic 1-second clumps.
        item = listener.queue.get(timeout=0.05)
    except _q.Empty:
        return None
    if item is None:
        return b""
    return item


@app.get("/stream.mp3", response_class=_RawStreamResponse)
async def stream_mp3(request: Request):
    """Infinite MP3 stream. Compatible with VLC, Winamp, Sonos, browsers."""
    listener = BROADCASTER.add_listener()
    return _RawStreamResponse(
        listener,
        config.get("stream.station_name", "Virtual DJ"),
        int(config.get("stream.bitrate_kbps", 128)),
    )


# --- status ----------------------------------------------------------------

@app.get("/api/library")
def api_library():
    """Library stats consumed by the web UI's Library panel."""
    return {"library": library.library_stats()}


@app.get("/api/status")
def api_status():
    return {
        "now_playing": BROADCASTER.state(),
        "library": library.library_stats(),
        "scan": library.STATUS.snapshot(),
        "config": config.load_config(),
    }


@app.get("/api/health")
def api_health():
    return {
        "llm": dj.llm_health(),
        "tts": dj.tts_health(),
        "websearch": websearch.health(),
        "ai": ai_meta.health(),
        "library": library.library_stats(),
        "broadcaster": BROADCASTER.state(),
    }


@app.get("/api/programs")
def api_programs():
    """Candidate themes for program grouping (genres/artists/decades)."""
    strategy = str(config.get("playback.program.strategy", "genre"))
    return {"strategy": strategy, "themes": library.program_themes(strategy)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_json({"type": "state", "data": BROADCASTER.state()})
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "state", "data": BROADCASTER.state()})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)


# --- library ---------------------------------------------------------------

class ScanRequest(BaseModel):
    music_dir: str | None = None
    full: bool = False
    use_web: bool | None = None


@app.post("/api/library/scan")
def api_scan(req: ScanRequest):
    if req.music_dir:
        config.save_config({"music_dir": req.music_dir})
    return library.scan_in_background(req.music_dir, req.full, req.use_web)


@app.get("/api/library/scan")
def api_scan_status():
    return library.STATUS.snapshot()


@app.get("/api/library/excluded")
def api_excluded(limit: int = 200, offset: int = 0, reason: str = ""):
    """Tracks skipped as unidentifiable, with the reason for each."""
    return {
        "stats": library.library_stats(),
        "tracks": library.excluded_tracks(limit, offset, reason or None),
    }


@app.get("/api/library/genres")
def api_genres():
    return library.list_genres()


@app.get("/api/library/languages")
def api_languages():
    return library.list_languages()


@app.get("/api/library/artists")
def api_artists(limit: int = 300):
    return library.list_artists(limit)


@app.get("/api/library/tracks")
def api_tracks(search: str = "", genre: str = "", artist: str = "",
               language: str = "", limit: int = 200, offset: int = 0):
    return library.query_tracks(
        search=search,
        genres=[g for g in genre.split(",") if g] or None,
        artists=[a for a in artist.split(",") if a] or None,
        languages=[l for l in language.split(",") if l] or None,
        limit=limit, offset=offset,
    )


# --- queue / transport -----------------------------------------------------

class EnqueueRequest(BaseModel):
    track_ids: list[int] = Field(default_factory=list)
    position: str = "end"
    replace: bool = False


@app.get("/api/queue")
def api_queue(limit: int = 50):
    return SCHEDULER.peek(limit)


@app.post("/api/queue")
def api_enqueue(req: EnqueueRequest):
    if req.replace:
        added = SCHEDULER.replace(req.track_ids)
    else:
        added = SCHEDULER.enqueue_track_ids(req.track_ids, req.position)
    return {"added": added, "queue": SCHEDULER.peek(50)}


@app.delete("/api/queue/{uid}")
def api_dequeue(uid: int):
    if not SCHEDULER.remove(uid):
        raise HTTPException(404, "queue item not found")
    return {"ok": True, "queue": SCHEDULER.peek(50)}


class MoveRequest(BaseModel):
    index: int


@app.post("/api/queue/{uid}/move")
def api_move(uid: int, req: MoveRequest):
    if not SCHEDULER.move(uid, req.index):
        raise HTTPException(404, "queue item not found")
    return {"ok": True, "queue": SCHEDULER.peek(50)}


@app.post("/api/queue/clear")
def api_queue_clear():
    SCHEDULER.clear()
    SCHEDULER.ensure_filled(10)
    return {"ok": True, "queue": SCHEDULER.peek(50)}


@app.post("/api/transport/skip")
def api_skip():
    BROADCASTER.skip()
    return {"ok": True}


@app.post("/api/transport/pause")
def api_pause():
    BROADCASTER.pause()
    return {"ok": True, "paused": True}


@app.post("/api/transport/resume")
def api_resume():
    BROADCASTER.resume()
    return {"ok": True, "paused": False}


# --- config ----------------------------------------------------------------

@app.get("/api/config")
def api_get_config():
    return config.load_config()


@app.put("/api/config")
async def api_put_config(request: Request):
    patch = await request.json()
    if not isinstance(patch, dict):
        raise HTTPException(400, "config patch must be an object")
    updated = config.save_config(patch)
    # Filter changes should take effect on the next refill. When the program
    # grouping settings change, rebuild the upcoming queue right away so the
    # new theme strategy (genre/artist/decade) is reflected immediately
    # rather than only after the old queue drains.
    if "playback" in patch:
        prog_patch = (patch["playback"] or {}).get("program")
        if prog_patch is not None:
            SCHEDULER.clear()
        SCHEDULER.ensure_filled(10)
    return updated


class LLMModelsRequest(BaseModel):
    base_url: str | None = None


@app.post("/api/llm/models")
def api_llm_models(req: LLMModelsRequest):
    """List models an Ollama instance exposes (for the web UI dropdown).

    Probes ``req.base_url`` if supplied (so the UI can test a URL before
    saving), otherwise the currently configured ``llm.base_url``.
    """
    return dj.list_ollama_models(req.base_url)


class LLMTestRequest(BaseModel):
    base_url: str | None = None
    model: str | None = None
    prompt: str | None = None


@app.post("/api/llm/test")
def api_llm_test(req: LLMTestRequest):
    """Verify the LLM can actually generate a DJ line (real chat completion).

    Uses the supplied base_url/model (so the UI can test before saving) or the
    configured ones. Returns {"ok", "text", "model", "error"}.
    """
    return dj.test_llm(req.base_url, req.model, req.prompt)



# --- DJ --------------------------------------------------------------------

@app.get("/api/dj/scripts")
def api_dj_scripts(limit: int = 20):
    return dj.recent_scripts(limit)


@app.get("/api/dj/voices")
def api_dj_voices():
    return {
        "voices": dj.available_voices(),
        "profiles": dj.voice_profiles(),
        "russian_profiles": dj.voice_profiles("russian"),
        "current": config.get("dj.voice"),
        "current_russian": config.get("dj.russian_voice"),
    }


class VoiceDownloadRequest(BaseModel):
    voice: str | None = None
    # "default" (config voices), "all" (every curated voice), or a list of ids.
    voices: list[str] | None = None


@app.post("/api/dj/voices/download")
def api_dj_voices_download(req: VoiceDownloadRequest):
    """Download Piper voice model(s) on demand.

    Without arguments, fetches the two default voices the app needs. Returns
    the ids that succeeded and any that failed (e.g. no network).
    """
    from . import voices as voice_mgr

    if req.voice:
        targets = [req.voice]
    elif req.voices:
        targets = list(req.voices)
    else:
        targets = [config.get("dj.voice"), config.get("dj.russian_voice")]
    targets = [t for t in targets if t]
    done = voice_mgr.download_voices(targets)
    return {
        "downloaded": done,
        "failed": [t for t in targets if t not in done],
        "available": dj.available_voices(),
    }


@app.get("/api/dj/voices/ensure")
def api_dj_voices_ensure():
    """Best-effort fetch of the default voices if missing (used at startup)."""
    from . import voices as voice_mgr

    done = voice_mgr.ensure_default_voices()
    return {"downloaded": done, "available": dj.available_voices()}


class PreviewRequest(BaseModel):
    text: str | None = None
    track_id: int | None = None
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    noise_scale: float | None = None


@app.post("/api/dj/preview")
def api_dj_preview(req: PreviewRequest):
    """Generate (and optionally speak) a DJ line without touching the stream."""
    text = (req.text or "").strip()
    track = None
    if not text:
        if req.track_id:
            track = library.get_track(req.track_id)
        if track is None:
            tracks = library.query_tracks(limit=1, random_order=True)
            track = tracks[0] if tracks else None
        if track is None:
            raise HTTPException(400, "library is empty")
        text = dj.generate_script(track)
    audio = dj.synthesize(
        text,
        voice=req.voice,
        language=req.language,
        speed=req.speed,
        noise_scale=req.noise_scale,
    )
    return {
        "text": text,
        "track": track,
        "audio_url": f"/api/dj/audio/{audio.name}" if audio else None,
        "duration": dj.audio_duration(audio) if audio else None,
    }


@app.get("/api/dj/audio/{name}")
def api_dj_audio(name: str):
    safe = (config.DJ_CACHE_DIR / name).resolve()
    if safe.parent != config.DJ_CACHE_DIR.resolve() or not safe.exists():
        raise HTTPException(404, "not found")
    return FileResponse(safe, media_type="audio/mpeg")


# --- presets ---------------------------------------------------------------

class PresetRequest(BaseModel):
    name: str
    payload: dict[str, Any] | None = None


@app.get("/api/presets")
def api_presets():
    return db.list_presets()


@app.post("/api/presets")
def api_save_preset(req: PresetRequest):
    if not req.name.strip():
        raise HTTPException(400, "preset name required")
    payload = req.payload
    if payload is None:
        cfg = config.load_config()
        payload = {"playback": cfg["playback"], "dj": cfg["dj"]}
    db.save_preset(req.name.strip(), payload)
    return {"ok": True, "presets": db.list_presets()}


@app.post("/api/presets/{name}/apply")
def api_apply_preset(name: str):
    payload = db.get_preset(name)
    if payload is None:
        raise HTTPException(404, "preset not found")
    updated = config.save_config(payload)
    SCHEDULER.clear()
    SCHEDULER.ensure_filled(10)
    return {"ok": True, "config": updated}


@app.delete("/api/presets/{name}")
def api_delete_preset(name: str):
    if not db.delete_preset(name):
        raise HTTPException(404, "preset not found")
    return {"ok": True}


@app.get("/api/history")
def api_history(limit: int = 50):
    return db.recent_history(limit)


# --- static frontend -------------------------------------------------------

@app.get("/")
def index():
    index_file = config.WEB_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "frontend not built"}, status_code=404)
    return FileResponse(index_file)


if config.WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")
