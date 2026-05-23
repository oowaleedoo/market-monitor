@echo off
title Market Monitor
set SCRIPT=%~dp0market_monitor.py
if exist "%~dp0.env" for /f "tokens=1,2 delims==" %%A in (%~dp0.env) do set %%A=%%B

reg query "HKCU\Software\Classes\marketmonitor" >nul 2>&1
if errorlevel 1 (
    reg add "HKCU\Software\Classes\marketmonitor"                    /ve /d "URL:Market Monitor" /f
    reg add "HKCU\Software\Classes\marketmonitor"                    /v "URL Protocol" /d "" /f
    reg add "HKCU\Software\Classes\marketmonitor\shell\open\command" /ve /d "pythonw \"%SCRIPT%\" --refresh" /f
)

py "%SCRIPT%"
if errorlevel 1 (
    echo.
    echo ERROR: script failed. Is Python installed?
    pause
)
