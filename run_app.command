#!/bin/bash
# Double-click in Finder to start the Harness Commonizer desktop app.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating local Python environment…"
  PY=/usr/local/bin/python3.9
  [ -x "$PY" ] || PY=python3
  "$PY" -m venv .venv || { echo "Setup failed"; read; exit 1; }
fi
# (re)install dependencies whenever requirements.txt changed since the last install
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing / updating dependencies…"
  .venv/bin/pip install -q -r requirements.txt && cp requirements.txt .venv/requirements.installed \
    || { echo "Dependency install failed"; read; exit 1; }
fi
exec .venv/bin/python app.py
