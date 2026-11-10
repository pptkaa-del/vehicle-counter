@echo off
chcp 65001 >nul
cd /d "%~dp0"
call venv\Scripts\activate
pip install pyinstaller
pyinstaller --noconfirm --windowed --name VehicleCounter ^
  --collect-all ultralytics --collect-all torchvision --hidden-import lap app.py
copy /y yolo11s.pt dist\VehicleCounter\ >nul
copy /y yolo11n.pt dist\VehicleCounter\ >nul
echo.
echo Готово: папка dist\VehicleCounter, запуск VehicleCounter.exe
echo Эту папку целиком можно копировать на другие компьютеры.
pause
