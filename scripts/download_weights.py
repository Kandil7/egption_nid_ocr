"""
Script to download YOLO weights from NASO7Y project.
Run this to get the trained YOLO models.
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path


# YOLO model files from NASO7Y project
# https://github.com/NASO7Y/OCR_Egyptian_ID
MODELS = {
    # Card detection
    "detect_id_card.pt": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id_card.pt",
    # Fields detection (firstName, lastName, serial, address, nid)
    "detect_odjects.pt": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_odjects.pt",
    # NID digit detection
    "detect_id.pt": "https://github.com/NASO7Y/OCR_Egyptian_ID/raw/main/detect_id.pt",
}


def download_file(url: str, dest: str) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading: {os.path.basename(dest)}")
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    """Download YOLO weights from NASO7Y."""
    print("=" * 60)
    print("Downloading YOLO Weights from NASO7Y")
    print("Source: https://github.com/NASO7Y/OCR_Egyptian_ID")
    print("=" * 60)

    # Create weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    # Download each model
    success_count = 0
    for filename, url in MODELS.items():
        dest = weights_dir / filename

        if dest.exists():
            print(f"\n{filename} already exists, skipping...")
            success_count += 1
            continue

        print(f"\n--- Downloading {filename} ---")
        if download_file(url, str(dest)):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{len(MODELS)} files")
    print("=" * 60)

    # List downloaded files
    print("\nDownloaded files:")
    for f in weights_dir.glob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    if success_count == len(MODELS):
        print("\n[OK] All models downloaded!")
        print("\nNext steps:")
        print("  1. Download OCR models: python scripts/download_models.py")
        print("  2. Run server: uvicorn app.main:app --reload")
    else:
        print("\n[FAIL] Some downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
