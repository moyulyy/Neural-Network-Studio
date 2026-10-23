@echo off
setlocal
echo === NeuroStudio 环境安装 ===
call conda create -n neurostudio python=3.10 -y
if errorlevel 1 echo [提示] 环境可能已存在，继续安装依赖...
echo === 安装 CUDA 12.1 版 PyTorch ===
call conda run -n neurostudio python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :error
echo === 安装 PySide6（官方源） ===
call conda run -n neurostudio python -m pip install PySide6==6.8.3 --index-url https://pypi.org/simple
if errorlevel 1 goto :error
echo === 安装数据与测试依赖 ===
call conda run -n neurostudio python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :error
echo.
echo 安装完成。启动方式：
echo   conda activate neurostudio
echo   python run.py
echo 或直接双击 run.bat
goto :eof
:error
echo 安装过程中出现错误，请检查网络后重试。
exit /b 1
