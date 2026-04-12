@echo off
title TruthGuard AI
cd /d "%~dp0"

set "PYTHON_CMD=python"

REM Prefer the stable local venv first, then fall back to .venv, then system python.
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\.venv\Scripts\python.exe"
)

REM Suppress TensorFlow noise
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

echo Starting TruthGuard AI...
echo Using interpreter: %PYTHON_CMD%
echo.
"%PYTHON_CMD%" -m streamlit run app.py --server.headless true --server.port 8501 --server.fileWatcherType none --server.disconnectedSessionTTL 30
if errorlevel 1 (
    echo.
    echo Fallback: trying python app.py...
    "%PYTHON_CMD%" app.py
)

echo.
pause
