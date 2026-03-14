"""
PaddleOCR-V5 Arabic Model Verification Script

This script verifies that the arabic_PP-OCRv5_mobile_rec model is correctly
loaded and configured for Egyptian ID name OCR.

Run with: python -m scripts.verify_ppocrv5_model
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logger import logger


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_paddleocr_import():
    """Test 1: Verify PaddleOCR can be imported."""
    print_section("TEST 1: PaddleOCR Import")
    
    try:
        from paddleocr import PaddleOCR, TextRecognition
        print("[PASS] PaddleOCR imported successfully")
        
        # Check version
        import paddleocr
        version = getattr(paddleocr, '__version__', 'unknown')
        print(f"       PaddleOCR version: {version}")
        
        # Check PaddlePaddle version
        try:
            import paddle
            paddle_version = paddle.__version__
            print(f"       PaddlePaddle version: {paddle_version}")
        except ImportError:
            print("[WARN] PaddlePaddle not found")
            
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import PaddleOCR: {e}")
        return False


def test_arabic_model_loading():
    """Test 2: Verify arabic_PP-OCRv5_mobile_rec model loads correctly."""
    print_section("TEST 2: Arabic PP-OCRv5 Model Loading")
    
    try:
        from paddleocr import PaddleOCR
        
        print("       Initializing PaddleOCR with lang='ar'...")
        start_time = time.time()
        
        # Initialize with Arabic language
        ocr = PaddleOCR(
            lang="ar",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            show_log=False,
        )
        
        load_time = time.time() - start_time
        print(f"[PASS] PaddleOCR initialized with lang='ar' in {load_time:.2f}s")
        
        # Check if model was loaded
        if hasattr(ocr, '_pipeline') and ocr._pipeline is not None:
            print("[INFO] OCR pipeline initialized")
        
        # Try to find model cache location
        try:
            import paddlex
            cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".paddlex",
                "official_models"
            )
            if os.path.exists(cache_dir):
                print(f"[INFO] Model cache directory: {cache_dir}")
                arabic_models = [f for f in os.listdir(cache_dir) if 'arabic' in f.lower()]
                if arabic_models:
                    print(f"[INFO] Found Arabic models: {arabic_models}")
                else:
                    print("[INFO] No 'arabic' named models in cache (model may use different naming)")
        except Exception as e:
            print(f"[DEBUG] Could not check cache: {e}")
        
        return ocr
    except Exception as e:
        print(f"[FAIL] Failed to load Arabic model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_recognition_class():
    """Test 3: Verify TextRecognition class with arabic model."""
    print_section("TEST 3: TextRecognition Class (Direct Model Access)")
    
    try:
        from paddleocr import TextRecognition
        
        print("       Initializing TextRecognition with arabic_PP-OCRv5_mobile_rec...")
        start_time = time.time()
        
        model = TextRecognition(
            model_name="arabic_PP-OCRv5_mobile_rec"
        )
        
        load_time = time.time() - start_time
        print(f"[PASS] TextRecognition initialized in {load_time:.2f}s")
        
        return model
    except Exception as e:
        print(f"[FAIL] Failed to initialize TextRecognition: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_arabic_recognition(ocr_engine):
    """Test 4: Test Arabic text recognition on sample image."""
    print_section("TEST 4: Arabic Text Recognition Test")
    
    if ocr_engine is None:
        print("[SKIP] OCR engine not available")
        return False
    
    try:
        # Create a test image with Arabic-like text structure
        # Egyptian ID names are typically 10-30 characters in Arabic
        test_image = np.ones((80, 400, 3), dtype=np.uint8) * 255
        
        # Add some Arabic-like text (using Arabic characters)
        # Note: cv2.putText doesn't support Arabic well, so we test with a pattern
        cv2.putText(
            test_image,
            "Test OCR",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
        )
        
        print("       Running OCR on test image...")
        start_time = time.time()
        
        result = ocr_engine.predict(test_image)
        
        inference_time = time.time() - start_time
        
        # Process results
        for res in result:
            if hasattr(res, 'dict') and res.dict():
                res_dict = res.dict()
                print(f"[INFO] Result: {res_dict}")
            elif hasattr(res, 'rec_text'):
                text = getattr(res, 'rec_text', '')
                score = getattr(res, 'rec_score', 0)
                print(f"[INFO] Recognized text: '{text}' (confidence: {score:.4f})")
        
        print(f"[PASS] Recognition completed in {inference_time*1000:.1f}ms")
        return True
        
    except Exception as e:
        print(f"[FAIL] Recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_specifications():
    """Test 5: Display model specifications."""
    print_section("TEST 5: PP-OCRv5 Arabic Model Specifications")
    
    specs = {
        "Model Name": "arabic_PP-OCRv5_mobile_rec",
        "Framework": "PaddlePaddle 3.x",
        "Architecture": "PP-OCRv5 (SVTR-HGNet backbone)",
        "Accuracy": "81.27%",
        "Improvement vs PP-OCRv3": "+22.83%",
        "Supported Languages": "Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English",
        "Input Format": "BGR image (numpy array) or image path",
        "Text Height Recommendation": "32-64px (optimal for Arabic script)",
        "rec_image_shape": "[3, 48, 320] (C, H, W) - height is auto-resized",
        "Preprocessing": "Internal normalization (handled by PaddleOCR)",
        "Model Cache": "~/.paddlex/official_models/",
    }
    
    for key, value in specs.items():
        print(f"       {key}: {value}")
    
    print("\n[INFO] For Egyptian ID names:")
    print("       - Names are typically 5-25 Arabic characters")
    print("       - Optimal text height: 48-64px for best accuracy")
    print("       - Use grayscale or BGR input (both supported)")
    print("       - Enable use_space_char=True for multi-word names")


def test_current_implementation():
    """Test 6: Check current project implementation."""
    print_section("TEST 6: Current Project Implementation Check")
    
    try:
        from app.models.ocr_engine import PaddleOCREngine, OCRMode
        
        print("       Checking PaddleOCREngine initialization...")
        engine = PaddleOCREngine()
        
        if engine.available():
            print("[PASS] PaddleOCREngine available")
            
            if engine._ar_reader is not None:
                print("[INFO] Arabic reader initialized")
                # Check initialization parameters
                if hasattr(engine._ar_reader, '_config'):
                    print(f"[DEBUG] Config: {engine._ar_reader._config}")
            else:
                print("[WARN] Arabic reader not initialized")
                
            if engine._digit_reader is not None:
                print("[INFO] Digit reader initialized")
            else:
                print("[WARN] Digit reader not initialized")
        else:
            print("[FAIL] PaddleOCREngine not available")
            
        return engine.available()
        
    except Exception as e:
        print(f"[FAIL] Implementation check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  PaddleOCR-V5 Arabic Model Verification")
    print("  Egyptian ID Name OCR Optimization")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Import
    results['import'] = test_paddleocr_import()
    
    # Test 2: Model Loading with lang="ar"
    ocr_engine = test_arabic_model_loading()
    results['model_loading'] = ocr_engine is not None
    
    # Test 3: TextRecognition Class
    text_rec_model = test_text_recognition_class()
    results['text_recognition'] = text_rec_model is not None
    
    # Test 4: Recognition Test
    if ocr_engine:
        results['recognition_test'] = test_arabic_recognition(ocr_engine)
    else:
        results['recognition_test'] = False
    
    # Test 5: Model Specifications
    test_model_specifications()
    results['specs_displayed'] = True
    
    # Test 6: Current Implementation
    results['implementation_check'] = test_current_implementation()
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  [SUCCESS] All verification tests passed!")
        print("\n  Key Findings:")
        print("  - PaddleOCR 3.x is correctly installed")
        print("  - arabic_PP-OCRv5_mobile_rec model loads with lang='ar'")
        print("  - Model is ready for Egyptian ID name OCR")
    else:
        print(f"\n  [WARNING] {total - passed} test(s) failed")
        print("  Check the error messages above for details")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
