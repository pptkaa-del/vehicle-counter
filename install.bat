@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Установка окружения ===
python --version || (echo Установите Python 3.10-3.12 с python.org, отметив "Add Python to PATH" & pause & exit /b 1)
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
echo.
echo Есть видеокарта NVIDIA? Ставим PyTorch с CUDA (быстрее в 5-20 раз).
set /p GPU="Установить версию для NVIDIA GPU? (y/n): "
if /i "%GPU%"=="y" (
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)
pip install -r requirements.txt
echo Скачивание модели...
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11n.pt')"
echo.
echo Готово. Запуск: run.bat
pause
