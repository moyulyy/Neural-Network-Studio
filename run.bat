@echo off
call conda run -n neurostudio python "%~dp0run.py"
if errorlevel 1 (
    echo.
    echo 启动失败。请先运行 setup_env.bat 创建环境。
    pause
)
