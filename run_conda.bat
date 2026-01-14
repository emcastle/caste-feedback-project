@echo off
setlocal
cd /d "%~dp0"

REM ===========================
REM CONFIG (edit these)
REM ===========================
set "ENV_NAME=feedback"
set "ENTRYPOINT=sanity_check.py"
REM ===========================

echo ==========================================================
echo   Conda Project Runner
echo   Env: %ENV_NAME%
echo   Folder: %CD%
echo ==========================================================

REM Handle install-only cases when .yml is updated
set "MODE=run"
if /I "%~1"=="install" set "MODE=install"

REM Ensure conda exists
where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda not found on PATH.
  echo Open "Anaconda Prompt" or ensure Anaconda is installed.
  pause
  exit /b 1
)

REM environment.yml must exist for this runner
if not exist "environment.yml" (
  echo ERROR: environment.yml not found in %CD%
  pause
  exit /b 1
)

REM Create env if missing
conda --no-plugins env list | findstr /i /c:"%ENV_NAME%" >nul
if errorlevel 1 (
  echo Creating env from environment.yml...
  conda --no-plugins env create -f environment.yml
  if errorlevel 1 (
    echo ERROR: failed to create env.
    pause
    exit /b 1
  )
) else (
  echo Env "%ENV_NAME%" already exists.
)

REM Update env from environment.yml
echo Updating env from environment.yml...
conda --no-plugins env update -n "%ENV_NAME%" -f environment.yml
if errorlevel 1 (
  echo ERROR: failed to update env.
  pause
  exit /b 1
)

REM Run sanity check / entrypoint without activation
if not exist "%ENTRYPOINT%" (
  echo ERROR: ENTRYPOINT not found: %ENTRYPOINT%
  pause
  exit /b 1
)

REM Handle updates only
if /I "MODE"=="install" (
  echo Install/update complete. Skipping entrypoint.
  pause
  exit /b 0
)

echo Running entrypoint: %ENTRYPOINT%
conda run -n "%ENV_NAME%" python "%ENTRYPOINT%"

set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" (
  echo FAILED (exit code %EXITCODE%)
  pause
  exit /b %EXITCODE%
)

echo SUCCESS
pause
exit /b 0
