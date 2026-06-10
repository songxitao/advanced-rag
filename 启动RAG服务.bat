@echo off
chcp 65001 > null
echo ======================================================
echo          🚀 正在一键启动本地 Advanced RAG 服务
echo ======================================================
echo.
cd /d "%~dp0"

echo [1/3] 正在激活 Miniconda 虚拟环境...
call "D:\program files\Miniconda\Scripts\activate.bat"

echo [2/3] 正在载入 deepseek-ocr 深度学习环境...
call conda activate deepseek-ocr

echo [3/3] 正在启动 FastAPI 检索引擎服务（热重载模式）...
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

echo.
echo ======================================================
echo  🛑 服务已停止运行。
echo ======================================================
pause
