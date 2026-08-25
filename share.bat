@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Put the dashboard on a public HTTPS address.
rem
rem  Uses a Cloudflare "quick tunnel": no account, no port forwarding, no
rem  firewall change. Cloudflare receives the traffic and forwards it down a
rem  connection this machine opens outward, so your home IP is never exposed
rem  and nothing is left listening on your router.
rem
rem  Three things to know:
rem    - the address changes every time you run this
rem    - it works only while this tunnel window and the dashboard are open
rem    - anyone with the link can open it, so treat the link as the password
rem
rem  What a visitor can do is read. Every write method returns 405 Method Not
rem  Allowed, because the service has no route that changes anything.
rem ===========================================================================

cd /d "%~dp0"
title MIE public link

set "PY=%~dp0.venv\Scripts\python.exe"
set "CF=%LOCALAPPDATA%\Programs\cloudflared\cloudflared.exe"
set "LOG=%TEMP%\mie_tunnel.log"
set "TIMEOUT=%SystemRoot%\System32\timeout.exe"

echo.
echo   Publishing the dashboard...
echo.

if not exist "%CF%" (
    echo   [!] cloudflared is not installed. Download it once from
    echo       https://github.com/cloudflare/cloudflared/releases/latest
    echo       and save cloudflared-windows-amd64.exe as:
    echo.
    echo       %CF%
    echo.
    pause
    exit /b 1
)

rem --- There has to be something to publish -------------------------------
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)" >nul 2>&1
if errorlevel 1 (
    echo   [!] The dashboard is not responding on port 8000.
    echo       Start it with run-engine.bat, give it a minute to load,
    echo       then run this again.
    echo.
    pause
    exit /b 1
)

if exist "%LOG%" del "%LOG%" >nul 2>&1

rem  The tunnel runs in its own window so it keeps going after this one closes,
rem  and so it can be watched or shut down on its own.
start "MIE tunnel" cmd /c ""%CF%" tunnel --url http://127.0.0.1:8000 --no-autoupdate > "%LOG%" 2>&1"

echo   Waiting for Cloudflare to assign an address...

"%PY%" "%~dp0scripts\wait_for_tunnel.py" "%LOG%"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

echo.
echo   Share that link. It stays live while the "MIE tunnel" window and the
echo   dashboard are both open, and it changes if you run this again.
echo.
echo   Close the "MIE tunnel" window to take it offline.
echo.
"%TIMEOUT%" /t 30 /nobreak
endlocal
