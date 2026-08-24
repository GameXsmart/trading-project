@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Start everything the engine needs, in three windows:
rem
rem    1. ingest    - live polling, funding rates, open interest, quality
rem                   scoring. This is the one that matters most right now:
rem                   the funding and open-interest tables are empty, so the
rem                   orderflow model has never actually run. They only fill
rem                   while this window is up.
rem    2. dashboard - the read-only API and web UI on http://127.0.0.1:8000
rem    3. cycle     - records predictions and resolves them on a loop, so the
rem                   learning loop has something to learn from.
rem
rem  Leave them running. News history and derivatives data accumulate only
rem  while ingest is alive, and that accumulation is the point: most of what
rem  this system cannot yet measure is a data problem, not a code problem.
rem
rem  System tools are called by full path throughout. Git Bash and similar
rem  put their own find.exe and timeout.exe on PATH, and those take different
rem  arguments - a script that silently picks the wrong one is worse than one
rem  that fails outright.
rem ===========================================================================

cd /d "%~dp0"
title Market Intelligence Engine - launcher

set "PY=%~dp0.venv\Scripts\python.exe"
set "TIMEOUT=%SystemRoot%\System32\timeout.exe"

echo.
echo   Crypto Market Intelligence Engine
echo   =================================
echo.

rem --- Python and virtual environment -------------------------------------
if not exist "%PY%" (
    echo   No virtual environment found. Creating one...
    set "LAUNCHER="
    where py >nul 2>&1 && set "LAUNCHER=py -3"
    if not defined LAUNCHER where python >nul 2>&1 && set "LAUNCHER=python"
    if not defined LAUNCHER (
        echo.
        echo   [!] Python is not installed, or not on PATH.
        echo       Install Python 3.12 or newer from https://python.org
        echo       and tick "Add python.exe to PATH" during setup.
        echo.
        pause
        exit /b 1
    )
    %LAUNCHER% -m venv "%~dp0.venv"
    if not exist "%PY%" (
        echo   [!] Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo   Installing dependencies. This takes a minute the first time...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -e ".[dev]" --quiet
    if errorlevel 1 (
        echo.
        echo   [!] Dependency install failed. Run this to see why:
        echo       "%PY%" -m pip install -e ".[dev]"
        echo.
        pause
        exit /b 1
    )
)

rem --- Database schema. Idempotent, safe on every launch. ------------------
echo   Preparing the database...
"%PY%" -m mie.cli db init >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [!] Database setup failed. Run this to see why:
    echo       "%PY%" -m mie.cli db init
    echo.
    pause
    exit /b 1
)

rem --- Backfill only on a genuinely empty database -------------------------
rem  The check runs in Python and reports through the exit code. Parsing the
rem  status table with `for /f` and `find` was the first thing I wrote and it
rem  broke on the space in this folder's path, then broke again by picking up
rem  Git Bash's find.exe. An exit code cannot be misparsed.
"%PY%" -m mie.cli have-history >nul 2>&1
if errorlevel 1 (
    echo.
    echo   No price history stored yet. Backfilling 1h, 4h and 1d...
    echo   This takes a few minutes and happens only once.
    echo.
    "%PY%" -m mie.cli backfill-all --timeframes 1d,4h,1h
    echo.
)

rem --- Launch the three long-running pieces --------------------------------
echo   Starting ingest, dashboard and prediction cycle...
echo.

start "MIE ingest" cmd /k ""%PY%" -m mie.cli run"
"%TIMEOUT%" /t 3 /nobreak >nul
start "MIE dashboard" cmd /k ""%PY%" -m mie.cli serve"
"%TIMEOUT%" /t 3 /nobreak >nul
start "MIE cycle" cmd /k ""%~dp0_cycle.bat""

echo   ---------------------------------------------------------------
echo.
echo     Dashboard:  http://127.0.0.1:8000
echo.
echo     Three windows are now running. Leave them open.
echo     Run stop-engine.bat to stop everything.
echo.
echo     What to expect: the dashboard will read "insufficient evidence"
echo     for a long time. That is the correct answer, not a fault - no
echo     model has demonstrated skill against a climatology baseline.
echo     What changes over the coming weeks is the data behind it.
echo.
echo   ---------------------------------------------------------------
echo.
"%TIMEOUT%" /t 20 /nobreak
endlocal
