# PP-OCRv5 Arabic Model Usage Guide

## Overview

This document describes the usage of the **arabic_PP-OCRv5_mobile_rec** model for Egyptian ID name OCR in this project.

## Model Specifications

| Property | Value |
|----------|-------|
| **Model Name** | `arabic_PP-OCRv5_mobile_rec` |
| **Framework** | PaddlePaddle 3.x |
| **Architecture** | PP-OCRv5 (SVTR-HGNet backbone, dual-branch) |
| **Accuracy** | 81.27% |
| **Improvement vs PP-OCRv3** | +22.83% |
| **Model Size** | ~50-100 MB |
| **License** | Apache 2.0 |

### Supported Languages

The arabic_PP-OCRv5_mobile_rec model supports:
- **Arabic** (primary)
- Persian (fa)
- Uyghur (ug)
- Urdu (ur)
- Pashto (ps)
- Kurdish (ku)
- Sindhi (sd)
- Balochi (bal)
- English (en)

### Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Input Format** | BGR image (numpy array) or image path |
| **Optimal Text Height** | 48-64px for Arabic script |
| **rec_image_shape** | [3, 48, 320] (C, H, W) - auto-resized internally |
| **Preprocessing** | Internal normalization (handled by PaddleOCR) |
| **Model Cache** | `~/.paddlex/official_models/` |

## Installation

### Requirements

```bash
# PaddlePaddle 3.x (CPU)
pip install paddlepaddle==3.0.0

# PaddlePaddle 3.x (GPU with CUDA 11.8)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# PaddleOCR 3.x
pip install paddleocr
```

### Windows Long Path Warning

**Important:** On Windows, PaddleOCR 3.x requires the `modelscope` package which may fail to install due to Windows Long Path limitations.

**Symptoms:**
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

**Solutions:**

1. **Enable Long Paths (Requires Admin):**
   - Run Registry Editor as Administrator
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`
   - Reboot your computer

2. **Use WSL2 (Recommended for Development):**
   - Install WSL2 with Ubuntu
   - Install Python and dependencies in WSL2 environment
   - No long path issues in Linux

3. **Use Docker:**
   - Use the provided Dockerfile for containerized deployment
   - All dependencies pre-configured

### Verify Installation

```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed successfully')"
```

## Usage

### Basic Usage (PaddleOCR 3.x API)

```python
from paddleocr import PaddleOCR

# Initialize with Arabic language (loads arabic_PP-OCRv5_mobile_rec)
ocr = PaddleOCR(
    lang="ar",  # This loads arabic_PP-OCRv5_mobile_rec
    use_doc_orientation_classify=False,  # ID cards are already aligned
    use_doc_unwarping=False,  # No distortion correction needed
    use_textline_orientation=False,  # Text lines are horizontal
    show_log=False,  # Suppress verbose logging
)

# Run OCR on an image
image_path = "path/to/image.png"
result = ocr.predict(image_path)

# Process results
for res in result:
    if hasattr(res, 'dict') and res.dict():
        res_dict = res.dict()
        text = res_dict.get('rec_text', '')
        score = res_dict.get('rec_score', 0)
        print(f"Text: {text}, Confidence: {score:.4f}")
```

### Using TextRecognition Class (Direct Model Access)

```python
from paddleocr import TextRecognition

# Initialize with specific model
model = TextRecognition(
    model_name="arabic_PP-OCRv5_mobile_rec"
)

# Run prediction
output = model.predict(input="image.png", batch_size=1)

# Process results
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

### Command Line Usage

```bash
# Full OCR pipeline with Arabic model
paddleocr ocr -i image.png \
    --lang ar \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device cpu

# Text recognition only
paddleocr text_recognition \
    --model_name arabic_PP-OCRv5_mobile_rec \
    -i image.png
```

## Integration with This Project

### OCR Engine Configuration

The project's `PaddleOCREngine` class automatically loads the PP-OCRv5 Arabic model:

```python
from app.models.ocr_engine import PaddleOCREngine

engine = PaddleOCREngine()

# Check model info
print(engine.get_model_info())
# Output:
# {
#     'arabic': {
#         'name': 'arabic_PP-OCRv5_mobile_rec',
#         'accuracy': '81.27%',
#         'languages': 'Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English'
#     },
#     'digit': {
#         'name': 'en_PP-OCRv5_mobile_rec',
#         'languages': 'English, European digits'
#     }
# }

# Run Arabic OCR
from app.utils.ocr_preprocess import preprocess_for_arabic_names

image = cv2.imread("name_field.png")
preprocessed = preprocess_for_arabic_names(image, target_height=64)
result = engine.run_arabic(preprocessed)

print(f"Recognized: {result.text}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Latency: {result.latency_ms}ms")
```

### Preprocessing for Egyptian ID Names

```python
from app.utils.ocr_preprocess import (
    preprocess_for_paddleocr,
    preprocess_for_arabic_names,
    estimate_text_height,
    validate_image_for_ocr
)

# Option 1: Standard preprocessing
image = cv2.imread("name_field.png")
preprocessed = preprocess_for_paddleocr(
    image,
    field_type="arabic",
    target_height=64,  # Optimal for Arabic script
    enhance=True  # Apply CLAHE for better contrast
)

# Option 2: Specialized preprocessing for Arabic names
preprocessed = preprocess_for_arabic_names(
    image,
    target_height=64,
    sharpen=False  # Set True for blurry scans
)

# Check if image is suitable for OCR
if not validate_image_for_ocr(image):
    print("Image not suitable for OCR")

# Estimate text height
text_height = estimate_text_height(image)
print(f"Estimated text height: {text_height}px")
```

## Performance Benchmarks

### Expected Latency (CPU)

| Operation | Latency |
|-----------|---------|
| Model Loading | 2-5 seconds (first time) |
| Single Name Recognition | 50-150ms |
| Batch Recognition (4 names) | 200-400ms |

### Expected Latency (GPU)

| Operation | Latency |
|-----------|---------|
| Model Loading | 1-3 seconds (first time) |
| Single Name Recognition | 20-50ms |
| Batch Recognition (4 names) | 80-150ms |

### Accuracy Expectations

| Field Type | Expected Accuracy |
|------------|-------------------|
| firstName (Arabic) | 85-95% |
| lastName (Arabic) | 85-95% |
| address (Arabic) | 80-90% |
| Mixed Arabic/English | 75-85% |

**Note:** Accuracy depends on:
- Image quality (DPI, contrast, noise)
- Text height (optimal: 48-64px)
- Font clarity (standard ID fonts work best)
- Card condition (wear, damage, glare)

## Troubleshooting

### Model Not Loading

**Issue:** `ModuleNotFoundError: No module named 'modelscope'`

**Solution:**
```bash
pip install modelscope --upgrade
```

### Empty Results

**Issue:** OCR returns empty text

**Possible causes:**
1. Text height too small (< 20px)
2. Image too dark or too bright
3. Low contrast

**Solutions:**
```python
# 1. Upscale image
preprocessed = preprocess_for_paddleocr(image, target_height=64)

# 2. Check image quality
from app.utils.ocr_preprocess import validate_image_for_ocr
if not validate_image_for_ocr(image):
    print("Image quality issue detected")

# 3. Try with enhancement
preprocessed = preprocess_for_paddleocr(image, enhance=True)
```

### Low Confidence Scores

**Issue:** Confidence < 0.5

**Solutions:**
1. Increase text height to 64px
2. Apply CLAHE enhancement
3. Try denoising:
```python
preprocessed = preprocess_for_arabic_names(image, sharpen=False)
```

### Wrong Text Direction (LTR instead of RTL)

**Issue:** Arabic text recognized left-to-right

**Solution:** The project handles RTL correction automatically in `text_utils.sort_blocks_by_reading_direction()`. Ensure you're using the unified `ocr_field()` method.

## Model Cache Management

### Cache Location

- **Windows:** `C:\Users\<username>\.paddlex\official_models\`
- **Linux/Mac:** `~/.paddlex/official_models/`

### Clear Cache

To force re-download of models:

```bash
# Windows
rmdir /s /q %USERPROFILE%\.paddlex

# Linux/Mac
rm -rf ~/.paddlex
```

### Pre-download Models

To pre-download models for offline use:

```python
from paddleocr import PaddleOCR

# This will download the Arabic model
ocr = PaddleOCR(lang="ar", show_log=True)
```

## Configuration Options

### PaddleOCR Initialization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lang` | "ch" | Language code ("ar" for Arabic) |
| `use_doc_orientation_classify` | True | Document orientation detection |
| `use_doc_unwarping` | True | Document unwarping correction |
| `use_textline_orientation` | True | Text line orientation detection |
| `show_log` | True | Show verbose logging |
| `device` | "gpu:0" | Device for inference |

### Recommended for Egyptian IDs

```python
ocr = PaddleOCR(
    lang="ar",
    use_doc_orientation_classify=False,  # IDs are aligned
    use_doc_unwarping=False,  # No distortion
    use_textline_orientation=False,  # Horizontal text
    show_log=False,  # Cleaner logs
    device="cpu",  # or "gpu:0" if available
)
```

## References

- [Hugging Face Model Card](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [PP-OCRv5 Technical Report](https://arxiv.org/abs/2507.05595)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-14 | Initial documentation for PP-OCRv5 integration |
