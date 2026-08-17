# Virtual DJ - container image (bundles Icecast2)
#
# Design goals:
#   * Small base (python:3.13-slim) + ffmpeg (decode/transcode) + icecast2
#     (the streaming server Winamp/VLC/Sonos consume natively). libgomp1 is the
#     onnxruntime (Piper) OpenMP backend.
#   * The app OWNS the Icecast lifecycle: it renders icecast.xml from its own
#     data/config.json at startup and spawns `icecast2 -b`. Icecast must start
#     as root to perform its <changeowner> privilege-drop (MP3 source mounts
#     only serve when that drop happens), so this image runs as root.
#   * The application code is read-only; all mutable state lives under /data
#     (config, SQLite DB, downloaded voice models, cache) and /tmp (icecast
#     logs / runtime). Mount a volume on /data. Music is mounted read-only
#     from the host.
#   * First-run voice models auto-download into /data/voices unless
#     VDJ_NO_VOICE_DOWNLOAD=1 is set.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All mutable state goes here; mount a volume on /data.
    VDJ_DATA_DIR=/data \
    # Docker mounts the host music library at /music (read-only); point the
    # default scan target there so a fresh container indexes it automatically.
    VDJ_MUSIC_DIR=/music \
    # Listen on all interfaces by default inside the container.
    VDJ_HOST=0.0.0.0 \
    VDJ_PORT=8420

# ffmpeg (decode/transcode + the icecast pusher) + libgomp1 (onnxruntime/Piper
# OpenMP backend) + icecast2 (streaming server). Installed WITH recommends:
# the MPEG/MP3 stream handler in this build relies on a recommended dependency,
# and we ship a minimal /etc/mime.types so MP3 sources register correctly.
RUN apt-get update \
    && apt-get install -y \
        ffmpeg \
        libgomp1 \
        icecast2 \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Icecast warns "Cannot open mime types file /etc/mime.types" and mishandles
# MP3 sources without it. A minimal mime.types (with the audio/mpeg mapping)
# matches a normal Debian host install and lets sources register correctly.
RUN printf 'audio/mpeg\t\t\tmp3\naudio/mpeg\t\t\tmp2\naudio/ogg\t\t\togg\n' > /etc/mime.types

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (read-only at runtime).
COPY app ./app
COPY web ./web
COPY config.example.json ./
COPY run.sh ./
COPY docker/icecast/icecast.xml.tmpl /app/icecast.xml.tmpl

# Mutable runtime dirs. We run as root (so icecast2 -b can changeowner to
# nobody), but icecast writes its logs to /tmp/icecast which must be world-
# writable for the dropped 'nobody' user.
RUN mkdir -p /data /tmp/icecast \
    && chmod 1777 /tmp/icecast

# NOTE: intentionally no USER directive — runs as root so the app can launch
# icecast2 -b (which itself drops to the unprivileged 'nobody' user).

VOLUME ["/data"]

# Healthcheck hits the built-in readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8420/api/health').status==200 else 1)"

EXPOSE 8420 8000

# Default: auto-download the default voice models on first run.
CMD ["python", "-m", "app.main"]
