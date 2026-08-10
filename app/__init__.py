"""Package init and logging setup for Virtual DJ."""

import logging
import os

__version__ = "1.0.0"

logging.basicConfig(
    level=os.environ.get("VDJ_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
