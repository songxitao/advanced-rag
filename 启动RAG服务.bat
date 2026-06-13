@echo off
:: 设置控制台为 GBK (936) 编码
chcp 936 > nul

echo ======================================================
echo          [START] 正在一键启动本地 Advanced RAG 服务
echo ======================================================
echo.

:: 切换到当前文件所在目录
cd /d "%~dp0"

echo [1/3] 正在激活 Miniconda 虚拟环境...
call "D:\program files\Miniconda\Scripts\activate.bat"

echo [2/3] 正在载入 deepseek-ocr 深度学习环境...
call conda activate deepseek-ocr

echo ======================================================
echo   请选择 RAG 模型的推理运行设备 (Device):
echo   [1] CPU 模式 (默认，占用显存为 0，防止显卡 OOM 溢出)
echo   [2] GPU 模式 (需 CUDA 支持，速度极快，需约 4-6GB 显存)
echo ======================================================
set /p dev_choice="请输入选项序号 [默认 1]: "

if "%dev_choice%"=="2" (
    set RAG_DEVICE=cuda
    echo [系统] 已设置运行设备为: GPU/CUDA 模式。
) else (
    set RAG_DEVICE=cpu
    echo [系统] 已设置运行设备为: CPU 模式。
)
echo.

echo [3/3] 正在启动 FastAPI 检索引擎服务...
echo API 文档地址: http://127.0.0.1:8000/docs
echo.

python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

echo.
echo ======================================================
echo  [STOP] 服务已停止运行。
echo ======================================================
pause
