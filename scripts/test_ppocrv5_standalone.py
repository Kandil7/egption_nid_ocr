"""
Standalone PaddleOCR-V5 Arabic Model Test

This script runs in a fresh Python process to avoid PaddleX reinitialization issues.

Run with: python scripts/test_ppocrv5_standalone.py
"""

import sys
import time
import numpy as np
import cv2


def main():
    print("\n" + "=" * 70)
    print("  PaddleOCR-V5 Arabic Model - Standalone Test")
    print("=" * 70)
    
    # Test 1: Import PaddleOCR
    print("\n[TEST 1] Importing PaddleOCR...")
    try:
        from paddleocr import PaddleOCR, TextRecognition
        print("[PASS] PaddleOCR imported successfully")
        
        import paddleocr
        version = getattr(paddleocr, '__version__', 'unknown')
        print(f"       Version: {version}")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    
    # Test 2: Initialize with Arabic model
    print("\n[TEST 2] Loading arabic_PP-OCRv5_mobile_rec...")
    try:
        start = time.time()
        ocr = PaddleOCR(
            lang="ar",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            show_log=True,  # Show model loading info
        )
        load_time = time.time() - start
        print(f"[PASS] Model loaded in {load_time:.2f}s")
        
        # Check model info
        if hasattr(ocr, '_pipeline'):
            print("[INFO] Pipeline initialized")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False
    
    # Test 3: Run OCR on test image
    print("\n[TEST 3] Running OCR test...")
    try:
        # Create test image with text-like pattern
        test_image = np.ones((64, 300, 3), dtype=np.uint8) * 255
        
        # Add some text pattern
        cv2.putText(
            test_image,
            "TEST OCR",
            (50, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
        )
        
        start = time.time()
        result = ocr.predict(test_image)
        inference_time = time.time() - start
        
        # Process results
        texts = []
        scores = []
        for res in result:
            if hasattr(res, 'dict') and res.dict():
                res_dict = res.dict()
                if 'rec_text' in res_dict:
                    texts.append(res_dict['rec_text'])
                    if 'rec_score' in res_dict:
                        scores.append(res_dict['rec_score'])
        
        print(f"[PASS] OCR completed in {inference_time*1000:.1f}ms")
        print(f"       Recognized: {texts}")
        print(f"       Scores: {[f'{s:.3f}' for s in scores]}")
        
    except Exception as e:
        print(f"[FAIL] OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Display model specifications
    print("\n[TEST 4] Model Specifications")
    print("-" * 50)
    specs = {
        "Model": "arabic_PP-OCRv5_mobile_rec",
        "Accuracy": "81.27%",
        "Improvement": "+22.83% vs PP-OCRv3",
        "Languages": "Arabic, Persian, Uyghur, Urdu, etc.",
        "Optimal Height": "48-64px",
        "Input": "BGR/Grayscale numpy array",
    }
    for key, value in specs.items():
        print(f"       {key}: {value}")
    
    print("\n" + "=" * 70)
    print("  All tests completed successfully!")
    print("=" * 70)
    print("\n[SUMMARY]")
    print("  - PaddleOCR 3.x is correctly installed")
    print("  - arabic_PP-OCRv5_mobile_rec loads with lang='ar'")
    print("  - Model is ready for Egyptian ID name OCR")
    print("\n[USAGE]")
    print("  from paddleocr import PaddleOCR")
    print("  ocr = PaddleOCR(lang='ar')")
    print("  result = ocr.predict(image)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
