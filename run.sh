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

exec .venv/bin/python -m app.main "$@"
