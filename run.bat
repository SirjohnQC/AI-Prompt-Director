@echo off
TITLE AI Prompt Director
CD /d "%~dp0"

ECHO ======================================================
ECHO      STARTING AI PROMPT DIRECTOR
ECHO ======================================================

REM 1. CHECK AND START OLLAMA
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
IF "%ERRORLEVEL%"=="0" (
    ECHO [OK] Ollama is already running.
) ELSE (
    ECHO [INFO] Ollama not running. Starting background service...
    REM Start Ollama minimized and hidden
    start /min "" ollama serve
    ECHO [OK] Ollama started.
)

REM 2. CHECK VENV
IF NOT EXIST "venv" (
    ECHO [ERROR] Not installed! Please run INSTALL.bat first.
    PAUSE
    EXIT
)

REM 3. LAUNCH APP
call venv\Scripts\activate
start "" "http://127.0.0.1:8000"
python main.py