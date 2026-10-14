"""Package init and logging setup for Virtual DJ.

Logs go to stderr (what the container's `docker logs` shows) AND, by default, to
a rotating file under the data directory — diagnosing a live incident used to mean
redirecting stdout to a temp file and hoping, because `basicConfig` was the only
sink. Rotation is bounded (``logging.max_mb`` × ``logging.backups``), so the log
cannot grow without limit either.

``logging.to_file`` / ``logging.max_mb`` / ``logging.backups`` live in the normal
config (see app/config.py); ``VDJ_LOG_LEVEL`` overrides the console level.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__version__ = "1.0.0"

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def normalize_level(value: str | None, *, default: str = "INFO",
                    extra: tuple[str, ...] = ()) -> str:
    """A level NAME that ``logging`` accepts, or ``default``.

    ``logging.basicConfig(level=...)`` raises ``ValueError: Unknown level`` for
    anything else — and ``VDJ_LOG_LEVEL=trace`` (a level uvicorn itself accepts)
    used to kill the app at import, before uvicorn was even configured, with a
    traceback that named neither the variable nor the fix. Unknown values are
    downgraded instead. ``extra`` carries names only the caller accepts (uvicorn
    knows ``trace``; ``logging`` does not).
    """
    name = (value or "").strip().upper()
    if name in extra or isinstance(logging.getLevelName(name), int):
        return name
    return default


def _add_file_handler() -> None:
    """Best-effort rotating file log; never fatal (a read-only volume is fine)."""
    try:
        from logging.handlers import RotatingFileHandler

        from . import config

        if not config.get("logging.to_file", True):
            return
        directory = config.DATA_DIR / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        max_bytes = int(config.number("logging.max_mb", 5, 1)) * 1024 * 1024
        # At least one backup: `backupCount=0` does not mean "keep only the live
        # file", it means RotatingFileHandler never rolls at all and the log
        # grows without limit — the opposite of what this module promises.
        backups = int(config.number("logging.backups", 3, 1))
        target = directory / "virtual-dj.log"
        root = logging.getLogger()
        # basicConfig is idempotent (it no-ops once root has handlers);
        # addHandler is not. Without this, every reload of the package (tests, a
        # plugin host) stacked another RotatingFileHandler on the SAME file:
        # each line written N times, N handlers racing on the rotation.
        for existing in list(root.handlers):
            if isinstance(existing, RotatingFileHandler) and \
                    Path(getattr(existing, "baseFilename", "")) == target:
                root.removeHandler(existing)
                existing.close()
        handler = RotatingFileHandler(
            target, maxBytes=max_bytes, backupCount=backups, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    except Exception:  # noqa: BLE001 - logging must never stop the app from booting
        logging.getLogger(__name__).warning("file logging unavailable", exc_info=True)


_wanted = (os.environ.get("VDJ_LOG_LEVEL") or "").strip()
_level = normalize_level(_wanted, default="INFO")
logging.basicConfig(level=_level, format=LOG_FORMAT)
if _wanted and _wanted.upper() != _level:
    logging.getLogger(__name__).warning(
        "VDJ_LOG_LEVEL=%r is not a logging level; using %s", _wanted, _level)
_add_file_handler()
