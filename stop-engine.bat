@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Stop everything run-engine.bat launched.
rem
rem  Closes the three windows by title rather than killing every python.exe on
rem  the machine, which would take out anything else you happen to be running.
rem ===========================================================================

cd /d "%~dp0"
title Market Intelligence Engine - stopping

set "TASKKILL=%SystemRoot%\System32\taskkill.exe"
set "TIMEOUT=%SystemRoot%\System32\timeout.exe"

echo.
echo   Stopping the engine...
echo.

set "STOPPED="
for %%W in ("MIE ingest" "MIE dashboard" "MIE cycle") do (
    "%TASKKILL%" /FI "WINDOWTITLE eq %%~W*" /T /F >nul 2>&1
    if not errorlevel 1 (
        echo     stopped %%~W
        set "STOPPED=1"
    )
)

if not defined STOPPED (
    echo     nothing was running.
) else (
    echo.
    echo   Stopped. Stored data is untouched - run-engine.bat picks up where it
    echo   left off, and no prediction already recorded can be revised.
)

echo.
"%TIMEOUT%" /t 6 /nobreak
endlocal
