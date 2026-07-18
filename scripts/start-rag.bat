@echo off
setlocal

:: Path to local Miniconda activation script
set "CONDA_ACTIVATE=D:\program files\Miniconda\Scripts\activate.bat"
set "CONDA_DIR=D:\program files\Miniconda"

if not exist "%CONDA_ACTIVATE%" (
    echo [ERROR] Miniconda activate.bat not found at: %CONDA_ACTIVATE%
    pause
    exit /b 1
)

echo [INFO] Activating Conda base environment...
call "%CONDA_ACTIVATE%" "%CONDA_DIR%"

echo [INFO] Activating RAG environment: deepseek-ocr ...
call conda activate deepseek-ocr

echo [INFO] Starting Advanced RAG Service on 0.0.0.0:8000 ...
uvicorn src.app:app --host 0.0.0.0 --port 8000

pause
