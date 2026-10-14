"""Boot-time logging: a level nobody recognises must never kill the app.

`logging.basicConfig(level="TRACE")` raises ValueError, and that ran at import —
before uvicorn was configured — so `VDJ_LOG_LEVEL=trace` (which uvicorn itself
accepts) used to take the container down with a traceback that named neither the
variable nor the fix.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app import _add_file_handler, config, normalize_level


def test_known_levels_pass_through_and_unknown_ones_downgrade():
    assert normalize_level("debug") == "DEBUG"
    assert normalize_level("WARNING") == "WARNING"
    assert normalize_level("") == "INFO"          # compose forwards unset as ""
    assert normalize_level(None) == "INFO"
    assert normalize_level("trace") == "INFO"     # logging does not know TRACE
    assert normalize_level("verbose") == "INFO"
    assert normalize_level("5") == "INFO"
    # uvicorn does accept trace, so the caller can whitelist it.
    assert normalize_level("trace", default="info", extra=("TRACE",)) == "TRACE"


def test_the_root_file_handler_is_not_stacked_on_reload(tmp_path, monkeypatch):
    """basicConfig no-ops once root has handlers; addHandler does not — a
    reloaded module used to add a second RotatingFileHandler on the SAME file,
    so every line was written twice and the handlers raced on rotation."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_CACHE",
                        {"logging": {"to_file": True, "max_mb": 1, "backups": 1}})
    target = tmp_path / "logs" / "virtual-dj.log"
    root = logging.getLogger()
    try:
        _add_file_handler()
        _add_file_handler()   # e.g. importlib.reload(app)
        mine = [h for h in root.handlers if isinstance(h, RotatingFileHandler)
                and Path(h.baseFilename) == target]
        assert len(mine) == 1
        assert mine[0].backupCount >= 1     # 0 would disable rotation entirely
        assert mine[0].maxBytes == 1024 * 1024
    finally:
        for handler in [h for h in root.handlers
                        if isinstance(h, RotatingFileHandler)
                        and Path(h.baseFilename) == target]:
            root.removeHandler(handler)
            handler.close()


def test_a_junk_logging_config_falls_back_to_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_CACHE",
                        {"logging": {"to_file": True, "max_mb": "5MB",
                                     "backups": "three"}})
    target = tmp_path / "logs" / "virtual-dj.log"
    root = logging.getLogger()
    try:
        _add_file_handler()                       # must not raise
        mine = [h for h in root.handlers if isinstance(h, RotatingFileHandler)
                and Path(h.baseFilename) == target]
        assert mine and mine[0].maxBytes == 5 * 1024 * 1024
        assert mine[0].backupCount == 3
    finally:
        for handler in [h for h in root.handlers
                        if isinstance(h, RotatingFileHandler)
                        and Path(h.baseFilename) == target]:
            root.removeHandler(handler)
            handler.close()
