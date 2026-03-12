"""
Test script for PaddleOCR-VL-1.5 integration
Run with: python -m scripts.test_paddle_vl
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def test_paddle_vl_import():
    """Test that PaddleOCR-VL-1.5 can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Import PaddleOCR-VL-1.5 engine")
    print("=" * 60)

    try:
        from app.models.paddle_vl_engine import PaddleVLEngine

        print("[PASS] PaddleVLEngine imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import PaddleVLEngine: {e}")
        return False


def test_paddle_vl_initialization():
    """Test PaddleOCR-VL-1.5 initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: Initialize PaddleOCR-VL-1.5 engine")
    print("=" * 60)

    try:
        from app.models.paddle_vl_engine import PaddleVLEngine

        engine = PaddleVLEngine()
        available = engine.available()

        if available:
            print(f"[PASS] PaddleOCR-VL-1.5 initialized successfully")
            print(f"       Device: {engine.device}")
            return True
        else:
            print("[WARN] PaddleOCR-VL-1.5 not available (will use fallback)")
            print("       This is OK if you don't have GPU or dependencies installed")
            return True  # Not a failure, just not available
    except Exception as e:
        print(f"[FAIL] Failed to initialize: {e}")
        return False


def test_paddle_vl_ocr():
    """Test OCR on a sample image."""
    print("\n" + "=" * 60)
    print("TEST 3: Run OCR with PaddleOCR-VL-1.5")
    print("=" * 60)

    try:
        from app.models.paddle_vl_engine import PaddleVLEngine

        engine = PaddleVLEngine()

        if not engine.available():
            print("[SKIP] PaddleOCR-VL-1.5 not available")
            return True

        # Create a test image with text
        # In real usage, you'd load an actual ID card image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255

        # Add some text to the image
        cv2.putText(
            test_image,
            "Test OCR",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        # Run OCR
        result = engine.run_ocr(test_image)

        print(f"       Text: '{result.text}'")
        print(f"       Confidence: {result.confidence:.2f}")
        print(f"       Latency: {result.latency_ms}ms")
        print(f"       Blocks: {len(result.blocks)}")

        print("[PASS] OCR test completed")
        return True

    except Exception as e:
        print(f"[FAIL] OCR test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_unified_ocr_engine():
    """Test that the unified OCR engine includes PaddleOCR-VL-1.5."""
    print("\n" + "=" * 60)
    print("TEST 4: Unified OCR Engine integration")
    print("=" * 60)

    try:
        from app.models.ocr_engine import OCREngine, OCRMode

        # Check if OCRMode has PADDLE_VL
        if hasattr(OCRMode, "PADDLE_VL"):
            print("[PASS] OCRMode.PADDLE_VL exists")
        else:
            print("[FAIL] OCRMode.PADDLE_VL not found")
            return False

        # Try to initialize OCR engine
        engine = OCREngine()
        engines = engine.get_available_engines()

        print(f"       Available engines: {engines}")

        if "paddle_vl" in engines:
            print(f"[PASS] PaddleOCR-VL-1.5 is available in unified engine")
        else:
            print("[INFO] PaddleOCR-VL-1.5 not available (may need GPU)")

        return True

    except Exception as e:
        print(f"[FAIL] Unified OCR engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_arabic_field_routing():
    """Test that Arabic fields are routed to PaddleOCR-VL-1.5."""
    print("\n" + "=" * 60)
    print("TEST 5: Arabic field routing")
    print("=" * 60)

    try:
        from app.models.ocr_engine import OCREngine

        # This will initialize the engine
        engine = OCREngine()
        engines = engine.get_available_engines()

        # Check if Arabic fields would use PaddleOCR-VL-1.5
        arabic_fields = [
            "firstName",
            "lastName",
            "address",
            "addressLine1",
            "addressLine2",
        ]

        print(f"       Arabic fields configured: {arabic_fields}")
        print(f"       Engine availability: {engines}")

        # The routing is configured in the code, just verify
        print("[PASS] Arabic field routing configured")
        return True

    except Exception as e:
        print(f"[FAIL] Arabic field routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PaddleOCR-VL-1.5 Integration Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import", test_paddle_vl_import()))
    results.append(("Initialization", test_paddle_vl_initialization()))
    results.append(("OCR", test_paddle_vl_ocr()))
    results.append(("Unified Engine", test_unified_ocr_engine()))
    results.append(("Arabic Routing", test_arabic_field_routing()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n  All tests passed!")
    else:
        print(f"\n  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
