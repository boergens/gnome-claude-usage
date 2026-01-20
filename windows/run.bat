@echo off
REM Claude Usage System Tray - Windows launcher

cd /d "%~dp0"

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check for required packages
python -c "import pystray, PIL" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install pystray pillow torch
)

REM Run the tray app
python claude_usage_tray.py
