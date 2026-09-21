@echo off
cd /d "%~dp0"
start "宠物寄养助手" powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
.venv\Scripts\python.exe -m app.web
pause
