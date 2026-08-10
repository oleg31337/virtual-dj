"""Compatibility entry point.

Lets the generic FastAPI detector (and `uvicorn main:app`) boot the app;
the canonical launcher is `./run.sh` (app.main, port 8420).
"""

from app.server import app  # noqa: F401
