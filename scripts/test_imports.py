"""Quick test to verify imports work correctly."""
import sys
sys.path.insert(0, r'K:\business\projects_v2\egption_nid_ocr')

print("Testing imports...")

try:
    from app.models.ocr_engine import PaddleOCREngine
    print("[PASS] PaddleOCREngine imported successfully")
except Exception as e:
    print(f"[FAIL] PaddleOCREngine import failed: {e}")

try:
    from app.utils.ocr_preprocess import preprocess_for_paddleocr, preprocess_for_arabic_names
    print("[PASS] OCR preprocessing functions imported successfully")
except Exception as e:
    print(f"[FAIL] OCR preprocessing import failed: {e}")

print("\nAll imports completed!")
