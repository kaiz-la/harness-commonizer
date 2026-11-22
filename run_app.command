#!/bin/bash
# Double-click in Finder to start the Harness Commonizer desktop app.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating local Python environment…"
  PY=/usr/local/bin/python3.9
  [ -x "$PY" ] || PY=python3
  "$PY" -m venv .venv && .venv/bin/pip install -q -r requirements.txt || { echo "Setup failed"; read; exit 1; }
fi
exec .venv/bin/python app.py
