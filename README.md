# Egyptian National ID OCR

A complete OCR system for extracting information from Egyptian National ID cards. Built with FastAPI, YOLO (NASO7Y), and EasyOCR.

## Features

- **Card Detection**: YOLO-based detection using NASO7Y models
- **Field Extraction**: Automatic extraction of name, ID number, address, serial
- **National ID Parsing**: Decodes the 14-digit ID to extract birth date, gender, governorate
- **Multi-Engine OCR**: EasyOCR (required) + PaddleOCR (optional for better Arabic)
- **CPU Optimized**: Designed for offline CPU inference
- **Confidence Scoring**: Per-field and overall confidence scores

## Architecture

```
Input Image
    ↓
[YOLOv8 - Card Detection] → Crop card
    ↓
[YOLOv8 - Field Detection] → Locate fields (firstName, lastName, nid, address, serial)
    ↓
[EasyOCR] → Extract text from each field
    ↓
[ID Parser] → Decode 14-digit national ID
    ↓
Structured JSON Output
```

## Quick Start (Windows)

```bash
# Double-click to run
run.bat
```

## Quick Start (Manual)

### 1. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install easyocr
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Download EasyOCR models (required)
python scripts/download_models.py

# Download YOLO weights from NASO7Y project
python scripts/download_weights.py
```

### 4. Run the Server

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 5. Test the API

```bash
# Using curl
curl -X POST http://localhost:8000/api/v1/extract -F "file=@path/to/id_card.jpg"

# Open in browser
# http://localhost:8000/docs
```

## API Endpoints

### `POST /api/v1/extract`

Extract information from an ID card image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Image file (JPEG, PNG, WebP, BMP)

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

### `GET /api/v1/models`

Check loaded model status.

## Configuration

Environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | development | Environment mode |
| `APP_PORT` | 8000 | Server port |
| `YOLO_CONF_THRESHOLD` | 0.50 | Detection confidence |
| `OCR_CPU_THREADS` | 4 | CPU threads for OCR |
| `MAX_IMAGE_SIZE_MB` | 10 | Max image size |

## Docker Deployment

```bash
# Build
docker build -t egyptian-id-ocr .

# Run
docker run -p 8000:8000 --cpus="4" --memory="4g" egyptian-id-ocr
```

## Project Structure

```
egyptian_id_ocr/
├── app/
│   ├── api/routes.py         # FastAPI endpoints
│   ├── core/config.py       # Settings
│   ├── core/logger.py       # Logging
│   ├── models/detector.py   # YOLO detector
│   ├── models/ocr_engine.py # OCR engines
│   ├── models/id_parser.py  # ID decoder
│   ├── services/pipeline.py  # Pipeline
│   ├── utils/image_utils.py # Preprocessing
│   └── utils/text_utils.py  # Text cleaning
├── scripts/
│   ├── download_weights.py   # Download NASO7Y YOLO
│   ├── download_models.py   # Download OCR models
│   └── test_basic.py       # Basic tests
├── tests/
├── venv/                    # Virtual environment
├── weights/                 # YOLO weights
├── models_cache/           # OCR model cache
├── .env
├── requirements.txt
├── Dockerfile
└── README.md
```

## Supported Fields

Based on NASO7Y YOLO model:

| Field | Description |
|-------|-------------|
| firstName | First name (Arabic) |
| lastName | Last name (Arabic) |
| nid | 14-digit national ID |
| address | Full address |
| serial | Card serial number |

Additional (back side):
| add_line_1 | Address line 1 |
| add_line_2 | Address line 2 |
| issue_date | Issue date |
| expiry_date | Expiry date |

## National ID Format

The 14-digit Egyptian National ID encodes:
- Digit 1: Century (2=1900s, 3=2000s)
- Digits 2-3: Birth year
- Digits 4-5: Birth month
- Digits 6-7: Birth day
- Digits 8-9: Governorate code
- Digit 10: Gender (odd=male, even=female)
- Digits 11-14: Sequence number

## Requirements

- Python 3.11+
- EasyOCR (required)
- PaddleOCR (optional - for better Arabic OCR)

## Testing

```bash
# Run basic tests
python scripts/test_basic.py

# Run pytest
pytest tests/ -v
```

## License

MIT License
