"""
Script to download OCR models for offline use.
Run this after installing dependencies: pip install -r requirements.txt

Note: PaddleOCR is optional. EasyOCR is required.
Tesseract with ara_number_id is recommended for better NID accuracy.
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


def check_tesseract_ara_number_id():
    """Check if Tesseract with ara_number_id is available."""
    print("Checking Tesseract with ara_number_id...")

    try:
        import pytesseract

        # Check if Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception:
            print("Tesseract not found in PATH")
            print("  Download from: https://github.com/tesseract-ocr/tesseract/releases")
            return False

        # Check for ara_number_id trained data
        tessdata_prefix = os.environ.get('TESSDATA_PREFIX', '')
        if not tessdata_prefix:
            # Try default locations
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tessdata",
                r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
                "./weights",
                "./tessdata",
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'ara_number_id.traineddata')):
                    tessdata_prefix = path
                    break

        if tessdata_prefix:
            ara_path = os.path.join(tessdata_prefix, 'ara_number_id.traineddata')
            if os.path.exists(ara_path):
                print(f"✓ ara_number_id.traineddata found at: {tessdata_prefix}")
                return True
            else:
                print(f"✗ ara_number_id.traineddata not found at: {tessdata_prefix}")
        else:
            print("TESSDATA_PREFIX not set")

        print("\nTo improve NID accuracy, install Tesseract with ara_number_id:")
        print("  1. Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract/releases")
        print("  2. Download ara_number_id.traineddata:")
        print("     https://github.com/tesseract-ocr/tessdata_best/raw/main/ara_number_id.traineddata")
        print("  3. Place it in the tessdata folder")
        print("  4. Set TESSDATA_PREFIX environment variable")
        return False

    except ImportError:
        print("pytesseract not installed")
        print("  Install with: pip install pytesseract")
        return False
    except Exception as e:
        print(f"Error checking Tesseract: {e}")
        return False


def main():
    """Download OCR models."""
    print("=" * 50)
    print("Downloading OCR models for offline use")
    print("=" * 50)

    # Create models cache directory
    cache_dir = "./models_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Create weights directory for Tesseract
    weights_dir = "./weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Download EasyOCR (required)
    print("\n--- EasyOCR (required) ---")
    easy_ok = download_easyocr_models()

    # Download PaddleOCR (optional)
    print("\n--- PaddleOCR (optional) ---")
    paddle_result = download_paddleocr_models()
    paddle_ok = paddle_result is True  # True = success, False = failed, None = skipped

    # Check Tesseract (recommended for NID)
    print("\n--- Tesseract (recommended for NID) ---")
    tesseract_ok = check_tesseract_ara_number_id()

    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"EasyOCR: {'OK' if easy_ok else 'FAILED'}")
    if paddle_result is None:
        print("PaddleOCR: SKIPPED (optional)")
    else:
        print(f"PaddleOCR: {'OK' if paddle_ok else 'FAILED'}")
    print(f"Tesseract ara_number_id: {'OK' if tesseract_ok else 'NOT AVAILABLE'}")

    if easy_ok:
        print("\n[OK] EasyOCR ready! Project can run.")
        if tesseract_ok:
            print("[OK] Tesseract with ara_number_id available - best NID accuracy!")
        else:
            print("[INFO] Tesseract ara_number_id not available - NID accuracy may be lower")
        if not paddle_ok and paddle_result is not False:
            print("     (PaddleOCR is optional - project works with EasyOCR only)")
        sys.exit(0)
    else:
        print("\n[FAIL] EasyOCR is required. Please check errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
