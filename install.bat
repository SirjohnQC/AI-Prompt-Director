@echo off
TITLE AI Prompt Director Installer
CLS

ECHO ======================================================
ECHO      INSTALLING AI PROMPT DIRECTOR...
ECHO ======================================================
ECHO.

REM 1. Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python is not installed!
    ECHO Please install Python from python.org and tick "Add to PATH".
    PAUSE
    EXIT /B
)

REM 2. Create Virtual Environment
IF NOT EXIST "venv" (
    ECHO [1/3] Creating isolated environment...
    python -m venv venv
) ELSE (
    ECHO [1/3] Environment exists, skipping...
)

REM 3. Install Dependencies
ECHO [2/3] Installing libraries (this may take a minute)...
call venv\Scripts\activate
pip install -r requirements.txt >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to install requirements. Check internet connection.
    PAUSE
    EXIT /B
)

REM 4. Create Desktop Shortcut (The Magic Part)
ECHO [3/3] Creating Desktop Shortcut...

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"

echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%USERPROFILE%\Desktop\AI Prompt Director.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%~dp0run_app.bat" >> %SCRIPT%
echo oLink.WorkingDirectory = "%~dp0" >> %SCRIPT%
echo oLink.Description = "Launch AI Prompt Director" >> %SCRIPT%

REM If you have an icon file named app.ico, uncomment the next line:
REM echo oLink.IconLocation = "%~dp0app.ico" >> %SCRIPT%

echo oLink.Save >> %SCRIPT%

cscript /nologo %SCRIPT%
del %SCRIPT%

ECHO.
ECHO ======================================================
ECHO      INSTALLATION COMPLETE!
ECHO ======================================================
ECHO.
ECHO A shortcut "AI Prompt Director" has been added to your Desktop.
ECHO You can close this window and double-click that shortcut now.
ECHO.
PAUSE