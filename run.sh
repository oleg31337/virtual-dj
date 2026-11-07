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

# Voice setup: ask the TTS service to make the configured voice usable so the
# DJ can speak out of the box. Best-effort -- with no service reachable yet the
# server still starts, and voices can be installed later from the web UI.
if [ "${VDJ_NO_VOICE_DOWNLOAD:-}" != "1" ]; then
  echo "ensuring the configured voice is installed on the TTS service…"
  .venv/bin/python -m app.voices || echo "  (voice install skipped -- will retry at runtime)"
fi

exec .venv/bin/python -m app.main "$@"
