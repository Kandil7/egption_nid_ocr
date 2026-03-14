---
title: Egyptian National ID OCR API
emoji: 🇪🇬
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: mit
tags:
  - ocr
  - egyptian-id
  - arabic-ocr
  - fastapi
  - computer-vision
  - yolo
  - easyocr
  - paddleocr
---

# 🇪🇬 Egyptian National ID OCR API

A production-ready OCR system for extracting information from Egyptian National ID cards. Built with FastAPI, YOLO detection, and multi-engine OCR (EasyOCR + PaddleOCR).

## ✨ Features

- **Card Detection**: YOLOv8-based ID card detection and cropping
- **Field Extraction**: Automatic extraction of name, ID number, address, serial
- **National ID Parsing**: Decodes 14-digit ID to extract birth date, gender, governorate
- **Multi-Engine OCR**: EasyOCR + PaddleOCR for optimal Arabic text recognition
- **CPU Optimized**: Designed for efficient CPU inference
- **Confidence Scoring**: Per-field and overall confidence scores
- **REST API**: FastAPI with automatic OpenAPI documentation

## 🚀 Quick Start

### Using the API

```python
import requests

# Upload an ID card image
with open("id_card.jpg", "rb") as f:
    response = requests.post(
        "https://YOUR-SPACE-ID.hf.space/api/v1/extract",
        files={"file": f}
    )
    print(response.json())
```

### cURL Example

```bash
curl -X POST "https://YOUR-SPACE-ID.hf.space/api/v1/extract" \
  -F "file=@id_card.jpg"
```

### API Documentation

Visit `/docs` for interactive Swagger UI documentation.

## 📋 API Endpoints

### `POST /api/v1/extract`

Extract information from an ID card image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Image file (JPEG, PNG, WebP, BMP, max 10MB)

**Response:**
```json
{
  "extracted": {
    "firstName": "محمد",
    "lastName": "عبدالله",
    "nid": "29901011234567",
    "address": "القاهرة، مصر",
    "serial": "12345678"
  },
  "confidence": {
    "overall": 0.87,
    "level": "high",
    "per_field": {
      "firstName": 0.85,
      "nid": 0.92,
      "address": 0.88
    }
  },
  "parsed_id": {
    "birth_date": "01/01/1999",
    "governorate": "القاهرة",
    "gender": "ذكر",
    "age": 27,
    "sequence": "2345"
  },
  "processing_ms": 3500
}
```

### `GET /api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### `GET /api/v1/models`

Check loaded model status.

### `GET /`

Root endpoint with API information.

## 🏗️ Architecture

```
Input Image
    ↓
[YOLOv8 - Card Detection] → Crop card
    ↓
[YOLOv8 - Field Detection] → Locate fields
    ↓
[EasyOCR/PaddleOCR] → Extract text
    ↓
[ID Parser] → Decode national ID
    ↓
Structured JSON Output
```

## ⚙️ Configuration

Environment variables (set in HF Spaces Secrets):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | production | Environment mode |
| `APP_PORT` | 7860 | Server port |
| `YOLO_CONF_THRESHOLD` | 0.50 | Detection confidence |
| `OCR_CPU_THREADS` | 4 | CPU threads for OCR |
| `MAX_IMAGE_SIZE_MB` | 10 | Max image size |
| `HF_PREWARM_MODELS` | false | Pre-load models at startup |

## 📊 Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Cold Start | < 60s | ~45s |
| Warm Request | < 15s | ~8s |
| Memory Usage | < 8GB | ~4GB |
| CPU Usage | < 100% | ~60% |

## 🛠️ Local Development

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR-USERNAME/egyptian-id-ocr
cd egyptian-id-ocr

# Build Docker image
docker build -f Dockerfile.hf -t egyptian-id-ocr .

# Run locally
docker run -p 7860:7860 --cpus="4" --memory="4g" egyptian-id-ocr

# Test the API
curl http://localhost:7860/api/v1/health
```

## 📁 Project Structure

```
egyptian-id-ocr/
├── Dockerfile.hf          # HF Spaces Docker config
├── README.md              # This file
├── .gitattributes         # Git LFS config
├── app/
│   ├── api/routes.py      # FastAPI endpoints
│   ├── core/config.py     # Settings
│   ├── models/
│   │   ├── detector.py    # YOLO detector
│   │   └── ocr_engine.py  # OCR engines
│   └── services/pipeline.py  # Main pipeline
├── weights/               # YOLO model weights
├── models_cache/          # OCR model cache
└── scripts/               # Utility scripts
```

## 🔍 Supported Fields

| Field | Description | Language |
|-------|-------------|----------|
| firstName | First name | Arabic |
| lastName | Last name | Arabic |
| nid | 14-digit national ID | Digits |
| address | Full address | Arabic |
| serial | Card serial number | Digits |
| issue_date | Issue date | Arabic/Digits |
| expiry_date | Expiry date | Arabic/Digits |

## 🇪🇬 National ID Format

The 14-digit Egyptian National ID encodes:
- **Digit 1**: Century (2=1900s, 3=2000s)
- **Digits 2-3**: Birth year
- **Digits 4-5**: Birth month
- **Digits 6-7**: Birth day
- **Digits 8-9**: Governorate code
- **Digit 10**: Gender (odd=male, even=female)
- **Digits 11-14**: Sequence number

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [NASO7Y](https://github.com/NASO7Y) for YOLO detection models
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR engine
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for Arabic OCR
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8

## 📞 Support

For issues or questions, please open an issue on the repository.
