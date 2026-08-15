#!/usr/bin/env bash
# Virtual DJ - launcher. Usage: ./run.sh [--host 0.0.0.0] [--port 8420]
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "creating venv…"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

# First-run voice setup: fetch the default Piper models so the DJ can speak
# out of the box. Best-effort -- if there is no network here, the server still
# starts and can download voices later from the web UI.
if [ "${VDJ_NO_VOICE_DOWNLOAD:-}" != "1" ]; then
  echo "ensuring default voice models are present…"
  .venv/bin/python -m app.voices || echo "  (voice download skipped -- will retry at runtime)"
fi

exec .venv/bin/python -m app.main "$@"
