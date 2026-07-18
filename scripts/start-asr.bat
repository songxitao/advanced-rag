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

echo [INFO] Activating ASR environment: asr_ui_env ...
call conda activate asr_ui_env

echo [INFO] Navigating to ASR directory E:\project\funclip-pro ...
cd /d E:\project\funclip-pro

echo [INFO] Starting SenseVoice ASR Service on 0.0.0.0:8001 ...
uvicorn asr_service:app --host 0.0.0.0 --port 8001

pause
