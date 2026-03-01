"""
Script to download OCR models for offline use.
Run this after installing dependencies: pip install -r requirements.txt

Note: PaddleOCR is optional. EasyOCR is required.
"""

import os
import sys


def download_easyocr_models():
    """Download EasyOCR models."""
    print("Downloading EasyOCR models...")

    try:
        import easyocr

        # Initialize reader - this triggers model download
        reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=True)

        # Test with a simple image
        print("EasyOCR models downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading EasyOCR: {e}")
        return False


def download_paddleocr_models():
    """Download PaddleOCR models (optional)."""
    print("Downloading PaddleOCR models (optional)...")

    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="arabic", use_gpu=False, show_log=True)

        print("PaddleOCR models downloaded successfully!")
        return True
    except ImportError:
        print("PaddleOCR not installed (optional)")
        print("  Install with: pip install paddlepaddle paddleocr")
        return None  # None means optional, not failed
    except Exception as e:
        print(f"Error downloading PaddleOCR: {e}")
        return False


def main():
    """Download OCR models."""
    print("=" * 50)
    print("Downloading OCR models for offline use")
    print("=" * 50)

    # Create models cache directory
    cache_dir = "./models_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Download EasyOCR (required)
    print("\n--- EasyOCR (required) ---")
    easy_ok = download_easyocr_models()

    # Download PaddleOCR (optional)
    print("\n--- PaddleOCR (optional) ---")
    paddle_result = download_paddleocr_models()
    paddle_ok = paddle_result is True  # True = success, False = failed, None = skipped

    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"EasyOCR: {'OK' if easy_ok else 'FAILED'}")
    if paddle_result is None:
        print("PaddleOCR: SKIPPED (optional)")
    else:
        print(f"PaddleOCR: {'OK' if paddle_ok else 'FAILED'}")

    if easy_ok:
        print("\n[OK] EasyOCR ready! Project can run.")
        if not paddle_ok and paddle_result is not False:
            print("     (PaddleOCR is optional - project works with EasyOCR only)")
        sys.exit(0)
    else:
        print("\n[FAIL] EasyOCR is required. Please check errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
