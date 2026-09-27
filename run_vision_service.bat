@echo off
setlocal
cd /d "%~dp0"
py -3.11 vision_service.py --config config.toml
endlocal
