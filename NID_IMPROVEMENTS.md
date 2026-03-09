# NID OCR Accuracy Improvement Plan

## Current Issues (from logs)

1. **ONNX detects 0 fields despite high scores**
   - max_objectness=0.649, max_class_score=0.767
   - But returns 0 detections
   - Root cause: objectness × class_conf multiplication kills confidence

2. **Template ROI fallback when ONNX fails**
   - Uses fixed coordinates that may not align with actual NID position
   - Results in poor quality crops

3. **NID preprocessing may hurt OCR**
   - Enhancement can introduce artifacts
   - Modern OCR engines have their own preprocessing

## Comprehensive Solution

### 1. Fix ONNX Detection (DONE)
- Use class confidence directly (not objectness × class)
- Lower threshold to 0.15
- Objectness as filter only (> 0.1)

### 2. Improve NID Field Detection
- Add YOLO fallback when ONNX returns 0
- Use template ROIs only as last resort
- Log which fields were detected

### 3. RAW NID Images (DONE)
- No preprocessing for NID fields
- Let OCR engines use their own preprocessing
- Just convert to grayscale

### 4. Multi-Engine OCR (DONE)
- Tesseract (ara_number_id) - best for clean digits
- EasyOCR (digits_only) - good fallback
- PaddleOCR digits - if available
- Select best result (prefer 14 digits)

### 5. Better Error Handling
- Validate NID format (14 digits)
- Check checksum
- Try multiple PSM modes in Tesseract

## Expected Results

With these fixes:
- ONNX should detect 5-8 fields (including NID)
- NID extraction should work on most images
- Accuracy should improve from ~70% to ~90%+
