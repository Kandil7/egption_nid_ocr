"""Test both ONNX and YOLO field detectors."""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from app.models.detector import EgyptianIDDetector
from app.core.logger import logger

# Load test image
img_path = 'tests/sample_ids/images/img_10_png.rf.d68824143abe697075c21dbd67223ef7.jpg'
image = cv2.imread(img_path)

if image is None:
    print(f"Failed to load image: {img_path}")
    sys.exit(1)

print(f"Loaded image: {image.shape}")
print("=" * 60)

# Initialize detector
detector = EgyptianIDDetector()

print("\n=== Testing ONNX Field Detector ===")
onnx_dets = detector.field_detector_onnx.detect(image, conf_threshold=0.18)
print(f"ONNX detected {len(onnx_dets)} fields:")
for det in onnx_dets[:10]:  # Show first 10
    print(f"  - {det.class_name:15s} (conf: {det.confidence:.3f}) bbox: {det.bbox}")

print("\n=== Testing YOLO Field Detector ===")
yolo_dets = detector.field_detector_yolo.detect(image, conf_threshold=0.25)
print(f"YOLO detected {len(yolo_dets)} fields:")
for det in yolo_dets[:10]:  # Show first 10
    print(f"  - {det.class_name:15s} (conf: {det.confidence:.3f}) bbox: {det.bbox}")

print("\n=== Testing crop_fields (ONNX first, YOLO fallback) ===")
fields = detector.crop_fields(image)
print(f"Extracted {len(fields)} fields: {list(fields.keys())}")

print("\n" + "=" * 60)
print("Test complete!")
