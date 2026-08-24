@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  The prediction cycle. Launched by run-engine.bat; you should not need to run
rem  this yourself.
rem
rem  Every hour it records a prediction for each tracked asset, resolves any
rem  whose horizon has elapsed, reports whether anything was actually learned,
rem  and evaluates the alert rules.
rem
rem  Hourly because the default timeframe is 1h. Running more often would
rem  record predictions at the same bar, and those are dropped as duplicates
rem  by design - a re-run can neither inflate the sample nor revise what was
rem  already said.
rem ===========================================================================

cd /d "%~dp0"
title MIE cycle - predict, resolve, learn

set "PY=%~dp0.venv\Scripts\python.exe"
set "TIMEOUT=%SystemRoot%\System32\timeout.exe"
set "ASSETS=BTC ETH SOL"

if not exist "%PY%" (
    echo   [!] No virtual environment. Run run-engine.bat first.
    pause
    exit /b 1
)

:loop
echo.
echo ===========================================================================
echo   cycle starting  %date% %time%
echo ===========================================================================

for %%A in (%ASSETS%) do (
    echo.
    echo   --- recording a prediction for %%A ---
    "%PY%" -m mie.cli predict %%A --timeframe 1h --horizon 12
)

echo.
echo   --- resolving outcomes, and what was learned from them ---
"%PY%" -m mie.cli learn --timeframe 1h

echo.
echo   --- alert rules ---
"%PY%" -m mie.cli alerts --timeframe 1h

echo.
echo   cycle done at %time%. Next run in one hour.
echo   Close this window to stop the cycle; ingest and dashboard keep running.
echo.
"%TIMEOUT%" /t 3600 /nobreak >nul
goto loop
