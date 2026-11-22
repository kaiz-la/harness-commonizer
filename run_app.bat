@echo off
rem Double-click in Explorer to start the Harness Commonizer desktop app (Windows).
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo First run: creating local Python environment...
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if not exist "%VPY%" (
    echo Could not create the environment. Install Python 3.9+ from python.org
    echo and tick "Add python.exe to PATH" and "tcl/tk and IDLE" during setup.
    pause & exit /b 1
  )
  "%VPY%" -m pip install -q --upgrade pip
)
rem install when requirements.txt changed since the last install, or when a package does not import
set NEED=0
fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul || set NEED=1
"%VPY%" -c "import pandas, numpy, scipy, sklearn, openpyxl, rapidfuzz, xlsxwriter, pyarrow, pptx" >nul 2>nul || set NEED=1
if "%NEED%"=="1" (
  echo Installing / updating dependencies into .venv ...
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (echo. & echo Dependency install failed - see the messages above. & pause & exit /b 1)
  "%VPY%" -c "import pandas, numpy, scipy, sklearn, openpyxl, rapidfuzz, xlsxwriter, pyarrow, pptx"
  if errorlevel 1 (echo. & echo Packages still do not import - see the error above. & pause & exit /b 1)
  copy /y requirements.txt ".venv\requirements.installed" >nul
)
"%VPY%" -c "import tkinter" >nul 2>nul || (echo Tkinter is missing: reinstall Python from python.org with "tcl/tk and IDLE" ticked. & pause & exit /b 1)
start "" "%~dp0.venv\Scripts\pythonw.exe" app.py
