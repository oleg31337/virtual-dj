"""Package init and logging setup for Virtual DJ.

Logs go to stderr (what the container's `docker logs` shows) AND, by default, to
a rotating file under the data directory — diagnosing a live incident used to mean
redirecting stdout to a temp file and hoping, because `basicConfig` was the only
sink. Rotation is bounded (``logging.max_mb`` × ``logging.backups``), so the log
cannot grow without limit either.

``logging.to_file`` / ``logging.max_mb`` / ``logging.backups`` live in the normal
config (see app/config.py); ``VDJ_LOG_LEVEL`` overrides the console level.
"""

import logging
import os

__version__ = "1.0.0"

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _console_level() -> str:
    # Empty-safe: compose forwards an unset VDJ_LOG_LEVEL as "" and
    # `logging.basicConfig(level="")` raises "Unknown level".
    return (os.environ.get("VDJ_LOG_LEVEL") or "").strip().upper() or "INFO"


def _add_file_handler() -> None:
    """Best-effort rotating file log; never fatal (a read-only volume is fine)."""
    try:
        from logging.handlers import RotatingFileHandler

        from . import config

        if not config.get("logging.to_file", True):
            return
        directory = config.DATA_DIR / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        max_bytes = max(1, int(config.get("logging.max_mb", 5) or 5)) * 1024 * 1024
        backups = max(0, int(config.get("logging.backups", 3) or 3))
        handler = RotatingFileHandler(
            directory / "virtual-dj.log", maxBytes=max_bytes,
            backupCount=backups, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
    except Exception:  # noqa: BLE001 - logging must never stop the app from booting
        logging.getLogger(__name__).warning("file logging unavailable", exc_info=True)


logging.basicConfig(level=_console_level(), format=LOG_FORMAT)
_add_file_handler()
