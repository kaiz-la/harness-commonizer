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
)
rem (re)install dependencies whenever requirements.txt changed since the last install
fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
if errorlevel 1 (
  echo Installing / updating dependencies...
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (echo Dependency install failed & pause & exit /b 1)
  copy /y requirements.txt ".venv\requirements.installed" >nul
)
start "" ".venv\Scripts\pythonw.exe" app.py
