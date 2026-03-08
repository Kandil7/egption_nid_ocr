# NID OCR Accuracy Improvements

## Overview

This document describes the improvements made to the Egyptian National ID (NID) OCR extraction accuracy.

## Problem

The original NID extraction had low confidence (0.428) and often returned empty results due to:
- Poor digit recognition on low-quality ID card images
- OCR errors in century digit (first digit)
- Lack of specialized preprocessing for NID fields
- No validation or auto-correction of extracted IDs

## Solutions Implemented

### 1. Tesseract with `ara_number_id` Trained Data (Primary)

**File:** `app/models/ocr_engine.py`

Added `run_nid_tesseract()` method that:
- Uses **`ara_number_id.traineddata`** - Tesseract's specialized model for Arabic numerals
- Applies 6 preprocessing variations:
  1. Otsu thresholding
  2. Inverted Otsu
  3. CLAHE + Otsu
  4. Adaptive Gaussian
  5. Adaptive Mean
  6. Morphological closing + Otsu
- Uses PSM 7 (single text line mode)
- Applies denoising before thresholding
- Implements ensemble voting when multiple 14-digit results are found

### 2. Century Digit Auto-Correction

**File:** `app/utils/text_utils.py`

Added `_fix_nid_century_digit()` function:
- Detects invalid century codes (6, 5, 8 → 2 or 3)
- Infers correct century from birth year:
  - Year 00-26 → century 3 (2000s)
  - Year 27-99 → century 2 (1900s)
- Automatically applied in `_validate_and_correct_nid()`

**Example:**
```
Input:  "60282092506000"  (6 is invalid, year=02)
Output: "30282092506000"  (corrected to century 3)
```

### 3. Enhanced Preprocessing Variations

**File:** `app/utils/ocr_preprocess.py`

Added `preprocess_nid_variations()` that generates 6 versions:
1. Binary threshold (morphological operations)
2. Inverted image
3. CLAHE-enhanced grayscale
4. Otsu threshold
5. Inverted Otsu
6. Original upscaled

### 4. OCR Engine Ensemble

**File:** `app/models/ocr_engine.py`

Updated `ocr_field()` for NID fields:
1. Runs Tesseract with `ara_number_id`
2. Runs EasyOCR specialized NID method
3. If both return 14 digits → ensemble voting
4. Returns best single result otherwise

**Ensemble Voting:**
- Position-wise majority voting across all 14-digit results
- Increases accuracy when engines disagree on specific digits

### 5. Improved Text Cleaning

**File:** `app/utils/text_utils.py`

Enhanced `_fix_common_digit_ocr_errors()` with mappings:
- O/o/Q/D → 0
- l/I/|/! → 1
- Z → 2
- S → 5
- B → 8
- G/b → 6
- q/g → 9

### 6. Debug Support

**File:** `app/services/pipeline.py`

Added automatic debug image saving:
- Saves NID field crops when extraction fails or confidence < 0.5
- Location: `debug/nid/nid_field_<timestamp>_orig.jpg`
- Location: `debug/nid/nid_field_<timestamp>_proc.jpg`

## Usage

### Install Tesseract with ara_number_id (Recommended)

```bash
# 1. Install Tesseract OCR
# Windows: Download from https://github.com/tesseract-ocr/tesseract/releases
# Or use: winget install UB-Mannheim.TesseractOCR

# 2. Download ara_number_id.traineddata
# Download from:
https://github.com/tesseract-ocr/tessdata_best/raw/main/ara_number_id.traineddata

# 3. Place in tessdata folder (e.g., C:\Program Files\Tesseract-OCR\tessdata\)

# 4. Set environment variable (optional if using default location)
set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

### Run Model Download Script

```bash
python scripts/download_models.py
```

This will:
- Download EasyOCR models (required)
- Check for Tesseract with ara_number_id (recommended)
- Provide installation instructions if not found

## Expected Results

### Before Improvements
```json
{
  "nid": "",
  "confidence": {
    "nid": 0.428
  }
}
```

### After Improvements (with Tesseract ara_number_id)
```json
{
  "nid": "30282092506000",
  "confidence": {
    "nid": 0.95
  }
}
```

### After Improvements (EasyOCR only)
```json
{
  "nid": "30282092506000",
  "confidence": {
    "nid": 0.76
  }
}
```

## Performance

- **Tesseract + EasyOCR ensemble:** ~2-3 seconds for NID field
- **EasyOCR only:** ~1-2 seconds for NID field
- Processing time varies based on image quality and preprocessing variations needed

## Testing

Run the test suite:
```bash
pytest tests/test_pipeline.py::TestTextUtils -v
```

Key tests:
- `test_clean_nid_valid_format` - Valid 14-digit NID
- `test_clean_nid_with_common_errors` - OCR error correction
- `test_clean_nid_invalid_century` - Century auto-correction
- `test_clean_nid_century_fix_from_6` - Specific century fix

## Troubleshooting

### NID still empty or low confidence

1. **Check debug images** in `debug/nid/` folder
2. **Verify Tesseract installation:**
   ```bash
   tesseract --version
   ```
3. **Check for ara_number_id.traineddata:**
   ```bash
   python scripts/download_models.py
   ```
4. **Review logs** for OCR engine selection and results

### Tesseract not found

- Ensure Tesseract is in PATH
- Or set `TESSDATA_PREFIX` environment variable

### Poor accuracy on specific images

- Check image quality (blur, glare, low contrast)
- Verify field detection (YOLO may not detect NID region correctly)
- Review debug images to identify preprocessing issues

## Future Improvements

1. **Train custom Tesseract model** on Egyptian ID NID fields
2. **Add CNN-based digit recognizer** specialized for ID cards
3. **Implement sequence validation** (checksum if applicable)
4. **Add user feedback loop** for continuous improvement

## References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Tesseract Training](https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract)
- [Egyptian National ID Format](https://en.wikipedia.org/wiki/Egyptian_national_identity_card)
