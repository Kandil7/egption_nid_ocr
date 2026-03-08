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


def preprocess_nid_field(image: np.ndarray) -> np.ndarray:
    """
    Specialized preprocessing for NID (National ID) field.
    Optimized for digit recognition with enhanced contrast and noise removal.

    Args:
        image: Input image of NID field

    Returns:
        Preprocessed image optimized for digit OCR
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Upscale significantly (digits need high resolution)
    h, w = gray.shape
    target_height = max(64, h)
    scale = target_height / h
    upscaled = cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply denoising
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 3: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 4: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Step 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Step 6: Remove small noise
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(eroded)
    min_area = (eroded.shape[0] * eroded.shape[1]) * 0.001

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, -1)

    # Return enhanced grayscale as BGR (EasyOCR-friendly, not binarized)
    # Heavy binarization hurts EasyOCR — use CLAHE-enhanced version instead
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr


def preprocess_nid_multi_scale(image: np.ndarray) -> list:
    """
    Generate multiple preprocessed versions of NID field at different scales.

    Args:
        image: Input image of NID field

    Returns:
        List of preprocessed images at different scales
    """
    scales = [1.0, 1.5, 2.0]
    results = []

    for scale_factor in scales:
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        preprocessed = preprocess_nid_field(resized)
        results.append(preprocessed)

    return results


def preprocess_nid_variations(image: np.ndarray) -> list:
    """
    Generate multiple preprocessing variations for difficult NID images.

    Args:
        image: Input image of NID field

    Returns:
        List of varied preprocessed images
    """
    variations = []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    target_height = max(64, h)
    scale = target_height / h
    upscaled = cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)

    # Variation 1: Standard preprocessed
    variations.append(preprocess_nid_field(image))

    # Variation 2: Inverted
    inverted = cv2.bitwise_not(upscaled)
    variations.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))

    # Variation 3: CLAHE grayscale
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(upscaled)
    variations.append(cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR))

    # Variation 4: Otsu
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variations.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # Variation 5: Inverted Otsu
    _, otsu_inv = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variations.append(cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR))

    # Variation 6: Original upscaled
    variations.append(cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR))

    return variations
