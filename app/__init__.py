"""Package init and logging setup for Virtual DJ."""

import logging
import os

__version__ = "1.0.0"

# Empty-safe: compose forwards an unset VDJ_LOG_LEVEL as "" and
# `logging.basicConfig(level="")` raises "Unknown level".
logging.basicConfig(
    level=(os.environ.get("VDJ_LOG_LEVEL") or "").strip().upper() or "INFO",
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
