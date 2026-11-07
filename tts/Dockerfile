# Virtual DJ TTS service — standalone speech-synthesis container.
#
# One job: turn text into WAV behind a universal HTTP API (/v1/capabilities,
# /v1/speech). The engine is chosen with TTS_ENGINE (supertonic3 | piper); the
# app never changes when that value does.
#
# Model weights live on the /models volume (they are large and re-downloadable),
# so rebuilding the image does not re-fetch 400 MB.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TTS_ENGINE=supertonic3 \
    TTS_STEPS=5 \
    TTS_LANGUAGES=en,ru \
    TTS_MODEL_DIR=/models \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=9000 \
    TTS_AUTOLOAD=1 \
    TTS_AUTODOWNLOAD=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# libgomp1 = onnxruntime's OpenMP backend. espeak-ng-data is what the optional
# Piper engine phonemises with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        espeak-ng-data \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ttssvc ./ttssvc

# Non-root, and the model volume is the only writable place it needs.
RUN useradd --create-home --uid 10001 tts \
    && mkdir -p /models \
    && chown -R tts:tts /models /srv
USER tts

VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import json,urllib.request,sys; d=json.load(urllib.request.urlopen('http://127.0.0.1:9000/v1/health', timeout=8)); sys.exit(0 if d.get('ok') else 1)"

EXPOSE 9000

CMD ["python", "-m", "ttssvc"]
