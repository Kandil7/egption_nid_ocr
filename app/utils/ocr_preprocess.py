"""
Text Pre-Processing Utilities

Optimized for PP-OCRv5 Arabic model (arabic_PP-OCRv5_mobile_rec).

PP-OCRv5 Model Specifications:
- Input: BGR or grayscale image (numpy array)
- Optimal text height: 48-64px for Arabic script
- rec_image_shape: [3, 48, 320] (auto-resized internally)
- Normalization: Handled internally by PaddleOCR
- Supported formats: BGR (cv2 default), RGB, grayscale

Key Insights for Egyptian ID Names:
- Arabic names typically have 5-25 characters
- Text height should be 48-64px for optimal recognition
- PaddleOCR handles most preprocessing internally
- Light enhancement (CLAHE) can improve low-contrast images
- Avoid binarization - PP-OCRv5 works best with grayscale/color

Reference: https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec
"""

import cv2
import numpy as np


def preprocess_for_paddleocr(
    image: np.ndarray,
    field_type: str = "arabic",
    target_height: int = 64,
    enhance: bool = True
) -> np.ndarray:
    """
    Preprocess image for PP-OCRv5 Arabic model.
    
    PP-OCRv5 handles most preprocessing internally, but these optimizations
    can improve accuracy for Egyptian ID names:
    
    1. Resize to optimal text height (48-64px)
    2. Light contrast enhancement (CLAHE) for low-quality scans
    3. Preserve grayscale/color (no binarization)
    
    Args:
        image: Input image (BGR or grayscale)
        field_type: Type of field ("arabic", "digits", "mixed")
        target_height: Target text height in pixels (48-64 optimal for Arabic)
        enhance: Apply CLAHE enhancement for low-contrast images
        
    Returns:
        Preprocessed image ready for PaddleOCR
        
    Example:
        >>> image = cv2.imread("name_field.png")
        >>> preprocessed = preprocess_for_paddleocr(image, target_height=64)
        >>> result = paddle_ocr.predict(preprocessed)
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get original dimensions
    h, w = gray.shape
    
    # ============================================
    # Step 1: Resize to optimal height
    # ============================================
    # PP-OCRv5 works best with text height 48-64px for Arabic
    # Egyptian ID names are typically 10-30mm tall on card
    # At 300 DPI scan, this translates to ~120-360px
    # We resize to target_height for consistent recognition
    
    if h < target_height or h > target_height * 2:
        # Calculate scale to achieve target height
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
        
        # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)
    
    # ============================================
    # Step 2: Light contrast enhancement (optional)
    # ============================================
    # CLAHE improves recognition for low-contrast scans
    # Common in Egyptian ID cards due to:
    # - Varying scan quality
    # - Card wear/aging
    # - Poor lighting during capture
    
    if enhance:
        # Apply CLAHE with conservative settings
        # clipLimit=2.0: Moderate contrast enhancement
        # tileGridSize=4x4: Fine-grained local adaptation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Use enhanced image if it has better contrast
        # Compare standard deviation (higher = more contrast)
        if enhanced.std() > gray.std():
            gray = enhanced
    
    # ============================================
    # Step 3: Convert back to BGR for PaddleOCR
    # ============================================
    # PaddleOCR accepts both grayscale and BGR
    # BGR is preferred for consistency with detection output
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return result


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


def preprocess_for_arabic_names(
    image: np.ndarray,
    target_height: int = 64,
    sharpen: bool = False
) -> np.ndarray:
    """
    Specialized preprocessing for Arabic names on Egyptian IDs.
    
    Optimized for:
    - firstName and lastName fields
    - Arabic script with 5-25 characters
    - Typical ID card scan quality (300-600 DPI)
    
    Args:
        image: Cropped name field image (BGR or grayscale)
        target_height: Target text height (64px recommended)
        sharpen: Apply mild sharpening for blurry scans
        
    Returns:
        Preprocessed image optimized for arabic_PP-OCRv5_mobile_rec
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # ============================================
    # Resize to optimal height
    # ============================================
    if h != target_height:
        scale = target_height / h
        new_w = int(w * scale)
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, (new_w, target_height), interpolation=interpolation)
    
    # ============================================
    # Denoising (for noisy scans)
    # ============================================
    # Egyptian IDs often have noise from:
    # - Card texture/patterns
    # - Scan artifacts
    # - Compression artifacts
    
    # Fast NL-means denoising
    # h=5: Moderate denoising strength
    gray = cv2.fastNlMeansDenoising(gray, h=5)
    
    # ============================================
    # Sharpening (optional, for blurry scans)
    # ============================================
    if sharpen:
        # Mild unsharp masking
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # ============================================
    # CLAHE for contrast
    # ============================================
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    
    # Convert to BGR for PaddleOCR
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return result


def estimate_text_height(image: np.ndarray) -> int:
    """
    Estimate the text height in an image.
    
    Useful for determining if resizing is needed before OCR.
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        Estimated text height in pixels
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to get text regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get median height of contours (approximates text height)
    heights = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 5 and w > 5:  # Filter noise
            heights.append(h)
    
    if heights:
        return int(np.median(heights))
    return 0


def validate_image_for_ocr(image: np.ndarray, min_height: int = 20) -> bool:
    """
    Validate that an image is suitable for OCR.
    
    Checks:
    - Minimum dimensions
    - Not all black/white
    - Has some content
    
    Args:
        image: Input image
        min_height: Minimum acceptable height in pixels
        
    Returns:
        True if image is suitable for OCR
    """
    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape
    
    # Check minimum size
    if h < min_height or w < min_height:
        return False
    
    # Check for empty image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Check if image has content (not all black or white)
    mean_val = gray.mean()
    if mean_val < 10 or mean_val > 245:
        return False
    
    # Check variance (flat images have no content)
    if gray.std() < 5:
        return False
    
    return True
