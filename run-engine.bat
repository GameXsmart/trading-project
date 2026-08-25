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
rem  Done in a subroutine rather than inside an if-block. Batch expands %VAR%
rem  when it *parses* a parenthesised block, not when it runs, so a variable set
rem  inside the block is still empty by the time a later line in the same block
rem  uses it. That turned the venv command into " -m venv" and the first-run
rem  path failed for everyone who did not already have a virtual environment.
if not exist "%PY%" call :setup
if not exist "%PY%" exit /b 1

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
exit /b 0


rem ===========================================================================
rem  First-run setup: create the virtual environment and install dependencies.
rem ===========================================================================
:setup
echo   No virtual environment found. Setting one up.
echo.

set "LAUNCHER="
where py >nul 2>&1 && set "LAUNCHER=py -3"
if not defined LAUNCHER (
    where python >nul 2>&1 && set "LAUNCHER=python"
)
if not defined LAUNCHER (
    echo   [!] Python is not installed, or is not on PATH.
    echo.
    echo       Install Python 3.12 or newer from https://python.org
    echo       During setup, tick "Add python.exe to PATH" - the installer
    echo       leaves it unticked by default, and without it this cannot
    echo       find Python.
    echo.
    pause
    exit /b 1
)

echo   Creating the environment with: %LAUNCHER%
%LAUNCHER% -m venv "%~dp0.venv"
if not exist "%PY%" (
    echo.
    echo   [!] Could not create the virtual environment. Try running this
    echo       by hand to see the error:
    echo         %LAUNCHER% -m venv ".venv"
    echo.
    pause
    exit /b 1
)

echo   Installing dependencies. A few minutes the first time.
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -e ".[dev]" --quiet
if errorlevel 1 (
    echo.
    echo   [!] Dependency install failed. Run this to see why:
    echo         "%PY%" -m pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)
echo   Environment ready.
echo.
exit /b 0
