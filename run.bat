@echo off
title AI Prompt Director
echo ===================================================
echo      AI PROMPT DIRECTOR - LAUNCHING
echo ===================================================

:: --- STEP 1: AUTO-START OLLAMA ---
echo [1/3] Checking Ollama Status...
tasklist | find /i "ollama.exe" >nul
if %errorlevel% neq 0 (
    echo    - Starting Ollama...
    start /B ollama serve >nul 2>&1
    timeout /t 5 >nul
) else (
    echo    - Ollama is running.
)

:: --- STEP 2: ACTIVATE VENV ---
if exist "venv\Scripts\activate.bat" (
    echo [2/3] Activating Virtual Environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] venv not found. Running in global scope.
)

:: --- STEP 3: LAUNCH APP ---
echo [3/3] Starting Web Server...
echo.
echo    UI: http://127.0.0.1:8000
echo.

start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"

:: Start Server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info

pause