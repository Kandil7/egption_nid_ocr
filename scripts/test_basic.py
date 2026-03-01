"""
Quick test script to verify basic functionality without models.
Run: python scripts/test_basic.py
"""

import sys
import os
import traceback
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from app.core.config import settings

        print(f"  [OK] config loaded (env: {settings.APP_ENV})")
    except Exception as e:
        print(f"  [FAIL] config failed: {e}")
        return False

    try:
        from app.core.logger import logger

        print("  [OK] logger initialized")
    except Exception as e:
        print(f"  [FAIL] logger failed: {e}")
        return False

    try:
        from app.models.id_parser import parse_national_id

        result = parse_national_id("29901011234567")
        assert result.valid
        print("  [OK] id_parser working")
    except Exception as e:
        print(f"  [FAIL] id_parser failed: {e}")
        traceback.print_exc()
        return False

    try:
        from app.utils.text_utils import clean_field

        cleaned = clean_field("2990101 12345 67", "nid")
        assert cleaned == "29901011234567", f"Got: {cleaned}"
        print("  [OK] text_utils working")
    except Exception as e:
        print(f"  [FAIL] text_utils failed: {e}")
        traceback.print_exc()
        return False

    try:
        from app.utils.image_utils import assess_quality
        import numpy as np

        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        quality = assess_quality(img)
        assert "overall" in quality
        print("  [OK] image_utils working")
    except Exception as e:
        print(f"  [FAIL] image_utils failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_id_parser():
    """Test ID parser with various inputs."""
    print("\nTesting ID parser...")

    from app.models.id_parser import parse_national_id

    # Test valid IDs - just check boolean results
    # Format: (id, should_be_valid)
    # 2 = 1900s, 3 = 2000s are valid century codes
    test_cases = [
        ("29901011234567", True),  # Valid male (1900s)
        ("30101021234567", True),  # Valid female (2000s)
        ("12345", False),  # Invalid length
        ("19901011234567", False),  # Invalid century (1)
        ("49901011234567", False),  # Invalid century (4)
    ]

    for nid, should_be_valid in test_cases:
        result = parse_national_id(nid)
        if result.valid != should_be_valid:
            print(f"  [FAIL] {nid} - expected valid={should_be_valid}, got {result.valid}")
            return False

    print("  [OK] ID parser tests passed")
    return True


def main():
    """Run basic tests."""
    print("=" * 50)
    print("Basic Functionality Tests")
    print("=" * 50)

    success = True

    if not test_imports():
        success = False

    if not test_id_parser():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("All basic tests passed!")
        print("\nNext steps:")
        print("  1. Download models: python scripts/download_weights.py")
        print("  2. Download OCR: python scripts/download_models.py")
        print("  3. Run server: uvicorn app.main:app --reload")
    else:
        print("Some tests failed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
