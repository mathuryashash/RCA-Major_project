@echo off
setlocal enabledelayedexpansion
title LocalRCA setup

rem  One double-click instead of "extract the whole ZIP, then find the exe two
rem  folders down". The two ways people actually fail at this are running the
rem  application from inside the archive, where the runtime beside it is not
rem  really there, and launching the collector by mistake because it is the
rem  first .exe they see. Both are checked below.

echo.
echo   LocalRCA setup
echo   ==============
echo.

set "HERE=%~dp0"
set "APP=%HERE%RCA-Desktop\RCA-Desktop.exe"

rem --- Running from inside the ZIP? Windows extracts to a temp folder to let
rem     you preview it, and everything appears to work until the app looks for
rem     the 1.1 GB of runtime that was never unpacked.
echo %HERE% | findstr /I "\\Temp\\ \\AppData\\Local\\Temp \\Temporary Internet" >nul
if %errorlevel%==0 (
    echo   [X] This is running from a temporary folder, which usually means
    echo       the ZIP was opened rather than extracted.
    echo.
    echo       Close this window, right-click the ZIP, choose "Extract All",
    echo       pick a folder such as C:\LocalRCA, and run this file from there.
    echo.
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo   [X] RCA-Desktop\RCA-Desktop.exe was not found next to this file.
    echo.
    echo       Extract the whole ZIP, keeping its folder structure, and run
    echo       this file from the extracted folder.
    echo.
    pause
    exit /b 1
)

rem --- Disk space. The extracted application is ~1.1 GB and the database grows
rem     by roughly 3.3 MB a day; running out later is a worse failure than
rem     refusing now.
for /f "tokens=3" %%a in ('dir /-c "%HERE%" ^| findstr /C:"bytes free"') do set FREE=%%a
if defined FREE (
    set /a FREEMB=!FREE:~0,-6! 2>nul
    if defined FREEMB if !FREEMB! LSS 500 (
        echo   [!] Only about !FREEMB! MB free on this drive.
        echo       LocalRCA needs room to grow its database ^(~3.3 MB/day^).
        echo.
    )
)

echo   Found: %APP%
echo.
echo   What happens next:
echo     - The application opens and asks what it may record.
echo     - Nothing is collected until you agree.
echo     - Agreeing registers a background collector to start at logon,
echo       adds a Start menu entry, and lists LocalRCA in Add/Remove Programs.
echo.
echo   The FIRST launch takes about a minute while Windows scans several
echo   thousand freshly extracted files. Later launches take a few seconds.
echo.
echo   This build is not code-signed, so Windows may show a SmartScreen
echo   warning. That is expected; see INSTALL.md for how to verify the
echo   download before trusting it.
echo.

choice /C YN /N /M "  Start LocalRCA now? [Y/N] "
if errorlevel 2 (
    echo.
    echo   Nothing was started. Run RCA-Desktop\RCA-Desktop.exe whenever you like.
    echo.
    pause
    exit /b 0
)

echo.
echo   Starting - the first launch is slow, please wait...
start "" "%APP%"
timeout /t 3 >nul
exit /b 0
