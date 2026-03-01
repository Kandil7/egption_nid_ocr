#!/bin/bash
# Egyptian ID OCR - Quick Start Script

echo "================================================"
echo "Egyptian ID OCR - Quick Start"
echo "================================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.11+"
    exit 1
fi

echo ""
echo "[1/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate venv
echo ""
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install easyocr --quiet 2>/dev/null
pip install -r requirements.txt --quiet 2>/dev/null
echo "[OK] Dependencies installed"

# Download models
echo ""
echo "[4/5] Downloading models..."
python scripts/download_models.py
python scripts/download_weights.py

# Run server
echo ""
echo "[5/5] Starting server..."
echo ""
echo "================================================"
echo "Server running at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo "================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
