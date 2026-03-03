@echo off
REM Egyptian ID OCR - Quick Start Script
REM Run this to set up and start the project

echo ================================================
echo Egyptian ID OCR - Quick Start
echo ================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Install dependencies
echo.
echo [2/5] Installing dependencies...
call venv\Scripts\pip.exe install easyocr --quiet
call venv\Scripts\pip.exe install -r requirements.txt --quiet

REM Download models
echo.
echo [3/5] Downloading EasyOCR models...
python scripts\download_models.py

echo.
echo [4/5] Downloading YOLO weights...
python scripts\download_weights.py

REM Run server
echo.
echo [5/5] Starting server...
echo.
echo ================================================
echo Server running at: http://localhost:8001
echo API docs at: http://localhost:8001/docs
echo ================================================
echo.
echo Press Ctrl+C to stop the server
echo.

call venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001

pause
