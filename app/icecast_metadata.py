"""Push \"now playing\" song titles to Icecast.

Icecast clients (Winamp, VLC, Sonos, ...) display the current track via the
mount's ``song`` metadata, which is updated over Icecast's admin API:

    POST /admin/metadata?mount=/<mount>&mode=updinfo&song=<title>
    Authorization: Basic <base64(admin:password)>

The audio relay (``app/icecast.py``) only carries the MP3 bytes -- ffmpeg's
``icecast://`` muxer cannot inject per-song ICY metadata -- so the app pushes
the title out-of-band here, exactly when the broadcaster starts a new track
(see ``Broadcaster._play_file``). Updates are best-effort and never block or
raise into the broadcast loop.
"""

from __future__ import annotations

import base64
import logging
import threading
import urllib.parse
import urllib.request
from typing import Any

from . import config

log = logging.getLogger(__name__)

# Icecast admin creds are rendered into icecast.xml by app/icecast_server.py.
# The admin user is fixed ("admin"); the password comes from the env the
# server reads, defaulting to "admin" when unset (matches the rendered
# icecast.xml default).
_ADMIN_USER = "admin"


def _admin_password() -> str:
    import os

    return os.environ.get("VDJ_ICECAST_ADMIN_PASSWORD", "admin")


def build_title(meta: dict[str, Any]) -> str | None:
    """Turn a now-playing meta dict into the Icecast ``song`` string.

    Returns ``None`` when there is no meaningful title (e.g. idle silence),
    so callers can clear the metadata instead of showing a junk label.
    """
    track = meta.get("track") or {}
    title = (track.get("title") or "").strip()
    artist = (track.get("artist") or "").strip()
    if title and artist:
        return f"{artist} - {title}"
    return title or artist or None


def update_metadata(title: str | None) -> bool:
    """Update the Icecast mount's ``song`` metadata.

    ``title`` of ``None``/empty clears it (sets it to a single space, which is
    what Icecast's admin API expects to blank the field). Returns ``True`` on a
    successful POST, ``False`` otherwise (no Icecast, disabled, auth error, ...).
    """
    if not config.get("icecast.enabled", False):
        return False
    mount = str(config.get("icecast.mount", "virtualdj")).lstrip("/") or "virtualdj"
    host = config.get("icecast.host", "127.0.0.1")
    port = int(config.get("icecast.port", 8008))
    password = _admin_password()

    song = (title or "").strip() or " "
    url = (
        f"http://{host}:{port}/admin/metadata"
        f"?mount=/{urllib.parse.quote(mount, safe='')}"
        f"&mode=updinfo&song={urllib.parse.quote(song, safe='')}"
    )
    auth = base64.b64encode(
        f"{_ADMIN_USER}:{password}".encode("utf-8")
    ).decode("ascii")
    # Icecast 2.4.x expects the metadata update as a GET with all parameters
    # in the query string (the POST-body form returns "unknown request").
    # Authorization uses the admin user/password (the source password is
    # rejected with "Mountpoint will not accept URL updates").
    req = urllib.request.Request(url, method="GET", headers={
        "Authorization": f"Basic {auth}",
    })
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                log.warning("icecast metadata update returned HTTP %s", resp.status)
            return ok
    except Exception as exc:  # noqa: BLE001 - best-effort; never break the stream
        log.debug("icecast metadata update failed: %s", exc)
        return False


def push_now_playing(kind: str, meta: dict[str, Any]) -> None:
    """Push the current item's title to Icecast (called from the broadcaster).

    ``kind`` is "track" or "dj"; for both we surface the *track's* title (the
    DJ break is an intro to that track). Idle / silence clears the metadata.

    This function performs a network call to Icecast's admin API and is meant
    to be run off the broadcast thread (see ``push_async``) so a slow/unreachable
    Icecast can never stall the audio stream.
    """
    if kind in ("track", "dj"):
        update_metadata(build_title(meta))
    else:
        # idle / unknown: clear so players don't keep showing the last song
        update_metadata(None)


def push_async(kind: str, meta: dict[str, Any]) -> None:
    """Fire-and-forget variant of ``push_now_playing``.

    Runs the Icecast admin call on a short-lived daemon thread so it can never
    block the broadcaster's audio-critical path. Failures are logged at debug
    level and otherwise ignored.
    """
    threading.Thread(
        target=_push_now_playing_safe, args=(kind, meta),
        name="vdj-icecast-meta", daemon=True,
    ).start()


def _push_now_playing_safe(kind: str, meta: dict[str, Any]) -> None:
    try:
        push_now_playing(kind, meta)
    except Exception:  # noqa: BLE001 - metadata is non-critical
        log.debug("icecast metadata push failed", exc_info=True)
