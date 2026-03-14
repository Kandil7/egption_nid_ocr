"""Script to inspect PaddleOCR configuration and model loading."""
import inspect
from paddleocr import PaddleOCR

# Get constructor signature
sig = inspect.signature(PaddleOCR.__init__)
print("=" * 60)
print("PaddleOCR Constructor Parameters:")
print("=" * 60)
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    print(f"  {param_name}: {default}")
