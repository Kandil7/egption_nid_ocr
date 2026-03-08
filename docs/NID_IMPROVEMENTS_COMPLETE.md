# NID OCR Accuracy Improvements - Complete Guide

## Problem Summary

The Egyptian National ID (NID) extraction was returning incomplete results:
- Only 6 digits instead of 14 (e.g., "303046" instead of "303046XXXXXXXX")
- Processing time: 60+ seconds
- Low confidence: 0.588

## Root Causes Identified

1. **Card Detection Issue**: Model detects corners (front-up, front-bottom, etc.) not single "id_card" class
2. **Field Detection**: NID field might not be properly cropped
3. **OCR Limitations**: Single OCR engine failing on difficult images
4. **Processing Time**: Too many preprocessing variations (6+) running sequentially

## Solutions Implemented

### 1. Fixed Card Detection (`app/models/detector.py`)

**Problem**: Card model detects 8 corner/edge points, not a single bounding box.

**Solution**: Group all corner detections to calculate full card boundaries.

```python
# Calculate bounding box from all corner/edge detections
x1 = min(d.bbox[0] for d in card_dets)
y1 = min(d.bbox[1] for d in card_dets)
x2 = max(d.bbox[2] for d in card_dets)
y2 = max(d.bbox[3] for d in card_dets)
```

**Result**: Card properly cropped from corner detections.

### 2. Specialized NID Extractor (`app/models/nid_extractor.py`)

**New module** with multiple extraction strategies:

1. **Crop-based extraction**: If NID field is detected
2. **Multi-region scanning**: Scans 6 typical NID locations on card
3. **Full card scan**: Contour-based digit detection across entire card
4. **Smart candidate selection**: Chooses best result based on digit count, format validity, confidence

**Key features**:
- Multiple preprocessing variants (Otsu, Inverted, CLAHE, Adaptive)
- Contour analysis for digit region detection
- Horizontal line grouping
- Century digit auto-correction

### 3. Optimized OCR Engine (`app/models/ocr_engine.py`)

**New NID extraction pipeline**:

1. **Specialized NID extractor** (contour-based, multi-region)
2. **Tesseract with ara_number_id** (fallback)
3. **EasyOCR specialized** (fallback)
4. **Ensemble voting** (when multiple 14-digit results)

**Optimizations**:
- Reduced variations from 6 to 3 key ones (binary, CLAHE, Otsu)
- Removed slow multi-scale retry
- Early exit when 14 digits found with good confidence
- Detailed timing logs

**Expected processing time**: 3-5 seconds (down from 60s)

### 4. Enhanced Debug Support (`app/services/pipeline.py`)

**Debug image saving** now triggers when:
- NID is empty
- NID has less than 14 digits
- Confidence < 0.7

**Location**: `debug/nid/nid_field_<timestamp>_orig.jpg`

### 5. Updated Configuration (`app/core/config.py`)

Added complete class mappings:
- `CARD_CLASSES`: 8 corner/edge classes for card detection
- `NASO7Y_CLASSES`: 31 field classes including "invalid_*" variants

## Usage

### Restart Server

```bash
# Stop current (Ctrl+C)
python app/main.py
```

### Test Extraction

```bash
curl -X POST "http://localhost:8000/api/v1/extract" ^
  -F "file=@path/to/id_card.jpg"
```

### Debug Visualization

Open browser: `http://localhost:8000/debug/ocr-viewer`

Upload image to see:
- Card detection with corner points
- Field detection overlays
- Preprocessing variations
- OCR results from each engine
- Ensemble voting (if applicable)

### Check Debug Images

```bash
dir debug\nid\*.jpg
```

Shows the NID field crops that were extracted for OCR.

## Expected Results

### Before
```json
{
  "nid": "303046",
  "confidence": 0.588,
  "processing_ms": 60458
}
```

### After (Best Case - Tesseract + EasyOCR agree)
```json
{
  "nid": "303046XXXXXXXX",
  "confidence": 0.95,
  "processing_ms": 3500
}
```

### After (Fallback - Single engine)
```json
{
  "nid": "303046XXXXXXXX",
  "confidence": 0.76,
  "processing_ms": 4000
}
```

## Troubleshooting

### NID still incomplete

1. **Check debug images** in `debug/nid/`:
   - Is the NID field properly cropped?
   - Is the text visible and clear?

2. **Use debug viewer**:
   - Open `/debug/ocr-viewer`
   - Check which step fails
   - Review preprocessing variations

3. **Check logs** for:
   ```
   Card detected from X parts at [x1,y1,x2,y2] (avg conf: X.XX)
   NID extractor: Selected 'XXX' from XXX (digits=X, valid=X, conf=X.XX)
   NID OCR complete: 'XXX' (conf: X.XX, time: XXXXms)
   ```

### Card not detected

- Check if image shows full ID card
- Ensure good lighting and contrast
- Try `/debug/test-card-detection` endpoint with lower threshold

### Processing still slow

- Check logs for timing breakdown
- Verify EasyOCR models are cached (first run downloads them)
- Consider disabling Tesseract if not available

## Architecture

```
Input Image
    ↓
[Card Detection] → Group corners → Crop card
    ↓
[Field Detection] → YOLO → Crop NID field
    ↓
[NID Extractor] → Multi-region scan → Contour analysis
    │   ├─ If 14 digits found → Return
    │   └─ If partial → Continue
    ↓
[Tesseract ara_number_id] → 6 preprocessing variations
    │   ├─ If 14 digits → Return
    │   └─ If partial → Continue
    ↓
[EasyOCR] → 3 key variations
    ↓
[Ensemble Voting] → If both have 14 digits
    ↓
[Century Digit Fix] → Auto-correct based on birth year
    ↓
Final NID (14 digits)
```

## Files Modified

| File | Changes |
|------|---------|
| `app/models/detector.py` | Fixed card detection to use corner grouping |
| `app/models/nid_extractor.py` | **NEW**: Specialized NID extraction module |
| `app/models/ocr_engine.py` | Integrated NID extractor, optimized variations |
| `app/services/pipeline.py` | Enhanced debug image saving |
| `app/core/config.py` | Added complete class mappings |
| `app/utils/text_utils.py` | Century digit auto-correction |
| `app/utils/ocr_preprocess.py` | NID preprocessing variations |

## Testing

```bash
# Run NID-specific tests
pytest tests/test_pipeline.py::TestTextUtils -v

# All tests
pytest tests/test_pipeline.py -v
```

**Expected**: 9/9 NID text tests pass

## Next Steps for Further Improvement

1. **Train custom digit classifier** on Egyptian ID NID fields
2. **Add deep learning-based sequence recognition** (CRNN)
3. **Implement checksum validation** (if applicable)
4. **Collect user feedback** for continuous improvement
5. **Add image quality checks** before OCR

## References

- Egyptian NID Format: 14 digits (century + year + month + day + governorate + gender + sequence)
- Century codes: 2=1900s, 3=2000s
- Governorate codes: 01-35
- Gender: odd=male, even=female
