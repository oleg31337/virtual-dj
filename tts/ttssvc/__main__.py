"""``python -m ttssvc`` — run the API with the configured host/port."""

from __future__ import annotations

import logging

import uvicorn

from . import config

log = logging.getLogger(__name__)


def main() -> None:
    uvicorn.run(
        "ttssvc.main:app",
        host=config.host(),
        port=config.port(),
        log_level=config.log_level() if config.log_level() in (
            "critical", "error", "warning", "info", "debug", "trace") else "info",
        access_log=False,
        # One long render (or a model download) must not be cut off, and the
        # service is internal: no reason to be precious about the timer.
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
