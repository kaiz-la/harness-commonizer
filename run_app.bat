@echo off
rem Double-click in Explorer to start the Harness Commonizer desktop app (Windows).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating local Python environment...
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if not exist ".venv\Scripts\python.exe" (
    echo Could not create the environment. Install Python 3.9+ from python.org
    echo and tick "Add python.exe to PATH" and "tcl/tk and IDLE" during setup.
    pause & exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (echo Setup failed & pause & exit /b 1)
)
start "" ".venv\Scripts\pythonw.exe" app.py
