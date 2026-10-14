#!/usr/bin/env python3
"""Entry point: python -m app.main  (or ./run.sh)"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from . import config, normalize_level

log = logging.getLogger("virtual_dj")


def _ensure_voices() -> None:
    """Best-effort first-run voice download (never fatal)."""
    try:
        from . import voices
        got = voices.ensure_default_voices()
        if got:
            log.info("first-run voice setup fetched: %s", ", ".join(got))
    except Exception as exc:  # noqa: BLE001 - a missing network must not crash boot
        log.warning("voice auto-download skipped (%s)", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual DJ radio station")
    # env_str/env_int treat an empty value (what compose forwards for an
    # unset .env entry) as "unset", and never raise on junk.
    parser.add_argument("--host", default=config.env_str("VDJ_HOST", "0.0.0.0"),
                        help="bind address (default 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int,
                        default=config.env_int("VDJ_PORT", 8420))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=config.env_str("VDJ_LOG_LEVEL", "info"))
    parser.add_argument("--no-voice-download", action="store_true",
                        help="skip the first-run voice auto-download")
    args = parser.parse_args()

    # Allow skipping via env (used by the Docker image / compose).
    skip_voice_download = (
        args.no_voice_download
        or config.env_flag("VDJ_NO_VOICE_DOWNLOAD", False)
    )
    if not skip_voice_download:
        _ensure_voices()

    # Say plainly whether the DJ can speak at all. A missing voice model means
    # Piper has nothing to say with: the breaks are dropped and no more LLM calls
    # are made for them, so the station looks healthy while you hear no DJ.
    try:
        from . import dj as _dj
        problem = _dj.tts_problem()
        if problem:
            log.warning(
                "DJ voice not ready: %s%s. Music still plays; the DJ stays "
                "silent. Fix it by picking an installed voice in the web UI's DJ "
                "card, running `python -m app.voices` to fetch %s, or unsetting "
                "VDJ_NO_VOICE_DOWNLOAD so it downloads on boot.",
                problem,
                " (downloads are DISABLED by VDJ_NO_VOICE_DOWNLOAD)"
                if skip_voice_download else "",
                config.get("dj.voice", "en_GB-alan-medium"))
        else:
            log.info("DJ voice ready: %s", config.get("dj.voice"))
    except Exception as exc:  # noqa: BLE001 - never block boot on a check
        log.debug("could not check the DJ voice: %s", exc)

    # SIGTERM (`docker stop`) is uvicorn's to handle: Server.run() installs its
    # own handlers and would overwrite anything set here, so the services are
    # stopped by the lifespan's shutdown path instead (app/server.py).
    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # uvicorn does LOG_LEVELS[level.lower()] and raises KeyError on an
        # unknown name, so an odd VDJ_LOG_LEVEL must be downgraded here too
        # (uvicorn's own set includes "trace", which `logging` does not know).
        log_level=normalize_level(args.log_level, default="info",
                                  extra=("TRACE",)).lower(),
        access_log=False,
        # The Icecast pusher keeps an open /stream.mp3 connection for the whole
        # lifetime of the app. On `docker stop` we cancel that task promptly
        # instead of waiting out the default 5s window (which uvicorn logs as a
        # confusing "timeout graceful shutdown exceeded" ERROR). The streaming
        # generator swallows CancelledError, so shutdown stays quiet.
        timeout_graceful_shutdown=0,
    )


if __name__ == "__main__":
    main()
