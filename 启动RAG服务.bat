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

echo [3/3] 正在启动 FastAPI 检索引擎服务（热重载模式）...
echo API 文档地址: http://127.0.0.1:8000/docs
echo.

python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

echo.
echo ======================================================
echo  [STOP] 服务已停止运行。
echo ======================================================
pause
