@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m PyInstaller --noconfirm --clean --onedir --name VisionService ^
  --add-data "config.toml;." ^
  --add-data "detect_coax.hdvp;." ^
  --add-data "best.pt;." ^
  --collect-all ultralytics ^
  vision_service.py

echo.
echo Build output: dist\VisionService\VisionService.exe
echo IMPORTANT: copy/edit config.toml beside VisionService.exe before running.
endlocal
