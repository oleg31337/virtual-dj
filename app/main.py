#!/usr/bin/env python3
"""Entry point: python -m app.main  (or ./run.sh)"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

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
    parser.add_argument("--host", default=os.environ.get("VDJ_HOST", "0.0.0.0"),
                        help="bind address (default 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("VDJ_PORT", "8420")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("VDJ_LOG_LEVEL", "info"))
    parser.add_argument("--no-voice-download", action="store_true",
                        help="skip the first-run voice auto-download")
    args = parser.parse_args()

    # Allow skipping via env (used by the Docker image / compose).
    skip_voice_download = (
        args.no_voice_download
        or os.environ.get("VDJ_NO_VOICE_DOWNLOAD", "").strip() == "1"
    )
    if not skip_voice_download:
        _ensure_voices()

    # Stop the Icecast daemon + pusher promptly on SIGTERM (e.g. `docker stop`)
    # so the open /stream.mp3 connection the pusher holds closes *before*
    # uvicorn drains, keeping shutdown quiet. The lifespan `finally` also
    # stops these on the shutdown event; this just front-runs it.
    def _stop_streaming_services(*_sig) -> None:
        try:
            from . import icecast as _icecast, icecast_server as _icecast_srv
            _icecast.PUSHER.stop()
            _icecast_srv.SERVER.stop()
        except Exception:  # noqa: BLE001 - never let a signal handler crash
            pass

    import signal as _signal
    try:
        _signal.signal(_signal.SIGTERM, _stop_streaming_services)
        _signal.signal(_signal.SIGINT, _stop_streaming_services)
    except ValueError:
        pass  # not in main thread (e.g. reload worker); safe to skip

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
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
