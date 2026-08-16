# Virtual DJ - container image
#
# Design goals:
#   * Small base (python:3.13-slim), ffmpeg for decoding, libgomp1 for the
#     onnxruntime (Piper) OpenMP backend.
#   * Runs as a non-root user.
#   * The application code is read-only; all mutable state lives under /data
#     (config, SQLite DB, downloaded voice models, cache). Mount a volume
#     there. Music is mounted read-only from the host.
#   * First-run voice models auto-download into /data/voices unless
#     VDJ_NO_VOICE_DOWNLOAD=1 is set.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All mutable state goes here; mount a volume on /data.
    VDJ_DATA_DIR=/data \
    # Listen on all interfaces by default inside the container.
    VDJ_HOST=0.0.0.0 \
    VDJ_PORT=8420

# ffmpeg (decode/transcode) + libgomp1 (onnxruntime/Piper OpenMP backend).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (read-only at runtime).
COPY app ./app
COPY web ./web
COPY config.example.json ./
COPY run.sh ./

# Non-root user. Needs write access to /data only.
RUN useradd --create-home --uid 1000 --gid 0 vdj \
    && mkdir -p /data && chown -R vdj:0 /data /app \
    && chmod -R g+w /data /app

USER vdj

VOLUME ["/data"]

# Healthcheck hits the built-in readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8420/api/health').status==200 else 1)"

EXPOSE 8420

# Default: auto-download the default voice models on first run.
CMD ["python", "-m", "app.main"]
