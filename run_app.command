#!/bin/bash
# Double-click in Finder to start the Harness Commonizer desktop app.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating local Python environment…"
  PY=/usr/local/bin/python3.9
  [ -x "$PY" ] || PY=python3
  "$PY" -m venv .venv || { echo "Setup failed"; read; exit 1; }
fi
CHECK="import pandas, numpy, scipy, sklearn, openpyxl, rapidfuzz, xlsxwriter, pyarrow, pptx"
# install when requirements.txt changed since the last install, or when a package does not import
if ! cmp -s requirements.txt .venv/requirements.installed || ! .venv/bin/python -c "$CHECK" 2>/dev/null; then
  echo "Installing / updating dependencies into .venv…"
  .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -c "$CHECK" \
    && cp requirements.txt .venv/requirements.installed \
    || { echo "Dependency install failed - see the messages above"; read; exit 1; }
fi
exec .venv/bin/python app.py
