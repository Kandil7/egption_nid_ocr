"""
Text Pre-Processing Utilities
Light preprocessing for EasyOCR - it handles most processing internally.
"""

import cv2
import numpy as np


def preprocess_for_easyocr(image: np.ndarray, field_type: str = "default") -> np.ndarray:
    """
    Light preprocessing for EasyOCR.
    EasyOCR works best with grayscale/color images, not binary.

    Args:
        image: Input image
        field_type: Type of field

    Returns:
        Preprocessed image ready for EasyOCR
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Upscale if too small (EasyOCR needs sufficient resolution)
    h, w = gray.shape
    if h < 32 or w < 32:
        scale = max(32 / h, 32 / w, 1.5)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Light enhancement for difficult images
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Convert back to BGR for EasyOCR (it expects color or grayscale)
    if len(image.shape) == 3:
        # Convert enhanced gray back to BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        result = enhanced

    return result
