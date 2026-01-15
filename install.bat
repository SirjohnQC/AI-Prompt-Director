@echo off
setlocal
TITLE AI Prompt Director - Ultimate Installer
color 0b

ECHO ===================================================
ECHO      AI PROMPT DIRECTOR - INSTALLATION WIZARD
ECHO ===================================================
ECHO.

:: --- STEP 1: CHECK PYTHON ---
echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [ERROR] Python is NOT installed or not in PATH.
    echo Please install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b
)
echo    - Python found.

:: --- STEP 2: SETUP VIRTUAL ENVIRONMENT ---
IF NOT EXIST "venv" (
    echo [2/5] Creating isolated virtual environment...
    python -m venv venv
) ELSE (
    echo [2/5] Virtual environment exists, skipping creation...
)

:: --- STEP 3: INSTALL PYTHON DEPENDENCIES ---
echo [3/5] Installing libraries into venv...
call venv\Scripts\activate
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [ERROR] Failed to install requirements.
    pause
    exit /b
)
echo    - Libraries installed successfully.

:: --- STEP 4: CHECK OLLAMA & MODELS ---
echo [4/5] Checking AI Environment...

:: Check Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0e
    echo [WARNING] Ollama is NOT installed.
    echo Opening download page...
    start https://ollama.com/download
    pause
    exit /b
)

:: Check Vision Model (qwen2.5-vl is standard, using it as safe default)
:: Note: If you prefer 'qwen3-vl', change the name below.
echo    - Checking Vision Model...
ollama list | findstr "qwen2.5-vl" >nul
if %errorlevel% neq 0 (
    echo      ...Pulling qwen2.5-vl (This may take a while)...
    ollama pull qwen2.5-vl
) else (
    echo      ...Vision model ready.
)

:: Check Writer Model
echo    - Checking Writer Model...
ollama list | findstr "llama3.2" >nul
if %errorlevel% neq 0 (
    echo      ...Pulling llama3.2...
    ollama pull llama3.2
) else (
    echo      ...Writer model ready.
)

:: --- STEP 5: CREATE DESKTOP SHORTCUT ---
echo [5/5] Creating Desktop Shortcut...

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"

echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%USERPROFILE%\Desktop\AI Prompt Director.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
:: IMPORTANT: Pointing to run.bat which handles the venv activation
echo oLink.TargetPath = "%~dp0run.bat" >> %SCRIPT%
echo oLink.WorkingDirectory = "%~dp0" >> %SCRIPT%
echo oLink.Description = "Launch AI Prompt Director" >> %SCRIPT%
:: echo oLink.IconLocation = "%~dp0app.ico" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%

cscript /nologo %SCRIPT%
del %SCRIPT%

ECHO.
color 0a
ECHO ===================================================
ECHO      INSTALLATION COMPLETE!
ECHO ===================================================
ECHO.
ECHO A shortcut "AI Prompt Director" has been created on your Desktop.
ECHO You can double-click it to start the tool.
ECHO.
pause