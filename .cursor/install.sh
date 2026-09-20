#!/usr/bin/env bash
# Idempotent dependency setup for the AI Team Chainlit app.
set -euo pipefail

cd "$(dirname "$0")/.."

# The stock image ships Python 3.12 but not always the venv seed package.
if [ ! -d .venv ]; then
  if ! python3 -m venv .venv 2>/dev/null; then
    py_minor="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    sudo apt-get update -qq
    sudo apt-get install -y "python${py_minor}-venv" || sudo apt-get install -y python3-venv
    python3 -m venv .venv
  fi
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
