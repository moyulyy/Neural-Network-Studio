@echo off
call conda run -n neurostudio python "%~dp0predict_app.py" %*
if errorlevel 1 pause
