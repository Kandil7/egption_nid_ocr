import sys
import os
from pathlib import Path
import json

# Add parent dir to path so it can find "app"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.pipeline import get_pipeline

def test_image(img_path):
    print(f"\n=========================================")
    print(f"Testing: {img_path}")
    print(f"=========================================")
    
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        
    pipeline = get_pipeline()
    result = pipeline.process(img_bytes)
    
    # Print the specific NID field result
    if 'extracted' in result:
        print(f"Extracted NID: {result['extracted'].get('nid', '')}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        test_dir = Path("tests/sample_ids/images")
        for img_file in test_dir.glob("*.jpg"):
            test_image(str(img_file))
