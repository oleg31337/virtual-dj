#!/usr/bin/env python3
"""Entry point: python -m app.main  (or ./run.sh)"""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual DJ radio station")
    parser.add_argument("--host", default=os.environ.get("VDJ_HOST", "0.0.0.0"),
                        help="bind address (default 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("VDJ_PORT", "8420")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("VDJ_LOG_LEVEL", "info"))
    args = parser.parse_args()

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
