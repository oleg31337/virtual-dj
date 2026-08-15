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

    if not args.no_voice_download:
        _ensure_voices()

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        access_log=False,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
