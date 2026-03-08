"""
Egyptian ID Card - Full Preprocessing & ROI Extraction Pipeline
يغطي: تحسين الصورة، تصحيح الزاوية، Perspective Transform، واستخراج الـ ROIs
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from app.core.logger import logger


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────


@dataclass
class CardROIs:
    """كل الـ ROIs المستخرجة من البطاقة"""

    full_card: np.ndarray  # البطاقة كاملة بعد التصحيح
    name_ar: Optional[np.ndarray] = None  # الاسم بالعربي
    name_en: Optional[np.ndarray] = None  # الاسم بالإنجليزي
    address: Optional[np.ndarray] = None  # العنوان
    id_number: Optional[np.ndarray] = None  # الرقم القومي
    birth_date: Optional[np.ndarray] = None  # تاريخ الميلاد
    gender: Optional[np.ndarray] = None  # النوع
    nationality: Optional[np.ndarray] = None  # الجنسية
    face: Optional[np.ndarray] = None  # الصورة الشخصية
    quality_score: float = 0.0  # جودة الصورة (0-1)


# ─────────────────────────────────────────────────────────
# Step 1: Image Quality Assessment
# ─────────────────────────────────────────────────────────


def assess_quality(image: np.ndarray) -> dict:
    """
    تقييم جودة الصورة قبل أي معالجة
    Returns: dict بـ scores و overall quality
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness — Laplacian variance (كلما أعلى كلما أوضح)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0)  # normalize to 0-1

    # 2. Brightness — mean pixel value
    brightness = float(np.mean(gray)) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2  # أحسن لو قريب من 0.5

    # 3. Contrast — standard deviation
    contrast = float(np.std(gray)) / 128.0
    contrast_score = min(contrast, 1.0)

    # 4. Glare detection — check for blown-out highlights
    overexposed = np.sum(gray > 240) / gray.size
    glare_score = 1.0 - min(overexposed * 10, 1.0)

    overall = np.mean([sharpness, brightness_score, contrast_score, glare_score])

    return {
        "overall": round(float(overall), 3),
        "sharpness": round(float(sharpness), 3),
        "brightness": round(float(brightness_score), 3),
        "contrast": round(float(contrast_score), 3),
        "glare": round(float(glare_score), 3),
        "acceptable": overall > 0.35,  # minimum threshold
    }


# ─────────────────────────────────────────────────────────
# Step 2: Basic Preprocessing
# ─────────────────────────────────────────────────────────


def resize_to_standard(image: np.ndarray, target_width: int = 1200) -> np.ndarray:
    """Resize للعرض المعياري مع الحفاظ على الـ aspect ratio"""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, new_h), interpolation=interp)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """إزالة الضوضاء مع الحفاظ على حواف الحروف"""
    # Bilateral filter: يحافظ على الحواف أحسن من Gaussian
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """تحسين التباين باستخدام CLAHE على قناة L في LAB"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Unsharp masking — يحسن حدة الحروف"""
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1 + strength, gaussian, -strength, 0)


# ─────────────────────────────────────────────────────────
# Step 3: Card Detection & Perspective Correction
# ─────────────────────────────────────────────────────────


def order_points(pts: np.ndarray) -> np.ndarray:
    """ترتيب 4 نقاط: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: أصغر مجموع
    rect[2] = pts[np.argmax(s)]  # bottom-right: أكبر مجموع
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: أصغر فرق
    rect[3] = pts[np.argmax(diff)]  # bottom-left: أكبر فرق
    return rect


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Perspective Transform — يصحح زاوية التصوير ويعطي صورة مستقيمة
    كأنك صورت البطاقة من فوق مباشرة
    """
    rect = order_points(corners)
    tl, tr, br, bl = rect

    # حساب العرض والارتفاع الجديد
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def find_card_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    إيجاد حواف البطاقة في الصورة باستخدام Contour Detection
    Returns: numpy array بـ 4 نقاط أو None لو مش لاقي
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing للـ edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Dilate عشان نوصل الـ edges المنفصلة
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # إيجاد الـ contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    img_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # البطاقة لازم تكون على الأقل 15% من الصورة
        if area < img_area * 0.15:
            continue

        # تقريب الـ contour لشكل مضلع
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # لو وجدنا مستطيل (4 نقاط)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    logger.warning("Could not find card corners via contour detection")
    return None


def auto_detect_and_warp(image: np.ndarray) -> np.ndarray:
    """
    تلقائياً: اكتشف البطاقة + صحح الـ perspective
    لو مش قادر يكتشف → يرجع الصورة كما هي
    """
    resized = resize_to_standard(image, 1200)
    corners = find_card_corners(resized)

    if corners is not None:
        warped = perspective_transform(resized, corners)
        logger.info(f"✅ Perspective corrected: {warped.shape}")
        return warped

    # Fallback: deskew بس
    logger.info("Using deskew fallback")
    return deskew(resized)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    تصحيح الانحراف الطفيف (rotation) باستخدام Projection Profile
    أفضل للصور المسحوبة scan مع انحراف بسيط [web:23]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # Threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    # تجاهل الزوايا الصغيرة جداً (أقل من 0.5 درجة)
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    logger.info(f"Deskewed by {angle:.2f}°")
    return rotated


# ─────────────────────────────────────────────────────────
# Step 4: ROI Extraction (Relative Coordinates)
# ─────────────────────────────────────────────────────────

# Egyptian ID Card Layout based on canonical schema
# Source: User provided schema + NASO7Y project
# Coordinates: (x_ratio, y_ratio, w_ratio, h_ratio)
# Rectified size: 1024x640

# Front side fields (NASO7Y: firstName, lastName, serial, address, nid)
FRONT_ROIS = {
    # Photo (left side)
    "face": (0.02, 0.08, 0.25, 0.70),
    # Names (right side, upper)
    "firstName": (0.30, 0.08, 0.68, 0.20),
    "lastName": (0.30, 0.28, 0.68, 0.18),
    # National ID (middle/bottom)
    "nid": (0.28, 0.46, 0.68, 0.14),
    # Other fields
    "serial": (0.30, 0.65, 0.68, 0.10),
    "address": (0.05, 0.78, 0.90, 0.18),
    # Logo placeholder
    "front_logo": (0.85, 0.02, 0.13, 0.08),
}

# Back side fields
BACK_ROIS = {
    # Address lines
    "add_line_1": (0.05, 0.10, 0.90, 0.12),
    "add_line_2": (0.05, 0.24, 0.90, 0.12),
    # National ID back
    "back_nid": (0.10, 0.40, 0.80, 0.12),
    # Serial and issue info
    "serial_num": (0.05, 0.55, 0.40, 0.10),
    "issue_code": (0.50, 0.55, 0.20, 0.10),
    # Dates
    "issue_date": (0.05, 0.70, 0.40, 0.10),
    "expiry_date": (0.50, 0.70, 0.45, 0.10),
    # Logo placeholder
    "back_logo": (0.85, 0.02, 0.13, 0.08),
}


def extract_roi(image: np.ndarray, roi: tuple) -> np.ndarray:
    """استخراج ROI واحد من الصورة"""
    h, w = image.shape[:2]
    x_r, y_r, w_r, h_r = roi
    x = int(x_r * w)
    y = int(y_r * h)
    rw = int(w_r * w)
    rh = int(h_r * h)

    # Padding صغير حول الـ ROI
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    rw = min(w - x, rw + pad * 2)
    rh = min(h - y, rh + pad * 2)

    crop = image[y : y + rh, x : x + rw]
    return crop if crop.size > 0 else image


def extract_all_rois(card_image: np.ndarray, side: str = "front") -> dict[str, np.ndarray]:
    """استخرج كل الـ ROIs من البطاقة"""
    rois_map = FRONT_ROIS if side == "front" else BACK_ROIS
    return {field: extract_roi(card_image, coords) for field, coords in rois_map.items()}


# ─────────────────────────────────────────────────────────
# Step 5: Per-Field Preprocessing for OCR
# ─────────────────────────────────────────────────────────


def preprocess_text_field(image: np.ndarray, field_type: str = "arabic") -> np.ndarray:
    """
    Optimized preprocessing with NID-specific enhancements.

    Key insight: EasyOCR and PaddleOCR have built-in preprocessing.
    Heavy transforms (binarization, aggressive denoising) often hurt performance.
    
    NID fields get specialized treatment for digit recognition.

    Args:
        image: Input image
        field_type: Type of field (nid, firstName, lastName, address, etc.)

    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale only if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Upscale strategy - only if truly needed (minimum 48px height)
    if h < 48:
        scale = 48 / h
        gray = cv2.resize(gray, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)
        h = 48

    # Field-specific lightweight preprocessing
    if field_type in ["nid", "front_nid", "back_nid", "id_number", "serial", "serial_num", "issue_code"]:
        # NID-specific preprocessing for optimal digit recognition
        return _preprocess_nid_field(gray, h, w)

    elif field_type in ["firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2"]:
        # Arabic text: minimal denoising, preserve text structure
        if h < 80:
            scale = 80 / h
            gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)

        # Only apply very light bilateral filter if noisy (high variance)
        if np.var(gray) > 2000:  # High variance indicates noise
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
        return gray

    elif field_type in ["issue_date", "expiry_date", "dob"]:
        # Dates: similar to digits
        if h < 64:
            scale = 64 / h
            gray = cv2.resize(gray, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)
        return gray

    elif field_type == "face":
        # Photo: return original color image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

    elif field_type in ["front_logo", "back_logo"]:
        # Logos: return grayscale
        return gray

    # Default: return grayscale with minimal processing
    if h < 80:
        scale = 80 / h
        gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)
    return gray


def _preprocess_nid_field(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Specialized preprocessing for NID/digit fields.

    Optimized for Egyptian NID format (14 digits) with:
    - Aggressive upscaling for small digits
    - Contrast enhancement for low-contrast images
    - Adaptive thresholding for Tesseract
    - Morphological operations to connect broken digit segments

    Args:
        gray: Grayscale or BGR image
        h: Height
        w: Width

    Returns:
        Preprocessed grayscale image optimized for digit OCR
    """
    # Ensure grayscale
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
    
    # Step 1: Upscale significantly for better digit recognition
    # NID digits are often small - upscale to at least 120px height
    target_height = max(120, h)
    if h < target_height:
        scale = target_height / h
        gray = cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)
        h, w = target_height, int(w * scale)

    # Step 2: Check contrast and enhance if needed
    # Calculate standard deviation (measure of contrast)
    contrast = np.std(gray)
    logger.info(f"NID preprocessing: contrast={contrast:.1f} (low if <30)")
    
    if contrast < 30:
        # Low contrast - apply aggressive enhancement
        logger.info("NID: Applying aggressive contrast enhancement")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)
        
        # Additional contrast boost
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=0)
    else:
        # Good contrast - mild enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

    # Return enhanced grayscale (single channel)
    # EasyOCR and PaddleOCR both accept grayscale images
    return enhanced


def adaptive_preprocess(image: np.ndarray, quality_score: float) -> np.ndarray:
    """
    Apply preprocessing based on image quality assessment.
    
    High quality (>0.7): Minimal processing
    Medium quality (0.4-0.7): Light enhancement
    Low quality (<0.4): Aggressive enhancement
    
    Args:
        image: Input image (BGR)
        quality_score: Quality score from 0 to 1
        
    Returns:
        Preprocessed image
    """
    if quality_score > 0.7:
        # High quality - let OCR handle it
        return image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif quality_score > 0.4:
        # Medium quality - light enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    else:
        # Low quality - aggressive enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16, 16))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Light sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(enhanced, -1, kernel)


def remove_background_lines(image: np.ndarray) -> np.ndarray:
    """
    إزالة الخطوط الأفقية والعمودية الموجودة في خلفية البطاقة
    يحسن دقة الـ OCR بشكل ملحوظ [web:23]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # إزالة الخطوط الأفقية
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # إزالة الخطوط العمودية
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # دمج الخطوط وطرحهم من الصورة
    lines_mask = cv2.add(h_lines, v_lines)
    result = cv2.subtract(thresh, lines_mask)

    return cv2.bitwise_not(result)


# ─────────────────────────────────────────────────────────
# Step 6: Side Detection (Front vs Back)
# ─────────────────────────────────────────────────────────


def detect_card_side(image: np.ndarray) -> str:
    """
    تحديد هل البطاقة وجه أمامي أم خلفي
    بيعتمد على: وجود الصورة الشخصية (circle/oval region) للأمامي
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # تفحص المنطقة اليسارية (حيث الصورة الشخصية في الأمامي)
    left_region = gray[:, : int(w * 0.30)]

    # إيجاد circles (الصورة الشخصية غالباً دايرة أو شبه دايرة)
    circles = cv2.HoughCircles(
        left_region,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )

    if circles is not None:
        return "front"

    # طريقة بديلة: تفحص الكثافة اللونية في منطقة الصورة
    face_region = image[:, : int(w * 0.30)]
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    variance = np.var(face_gray)

    # الصورة الشخصية = variance عالي (وجه بيفرق)
    return "front" if variance > 500 else "back"


# ─────────────────────────────────────────────────────────
# Step 7: Main Pipeline Entry Point
# ─────────────────────────────────────────────────────────


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array"""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — invalid format or corrupted file")
    return img


def full_preprocess_pipeline(
    image: np.ndarray, use_yolo: bool = False, yolo_fields: Optional[dict] = None
) -> CardROIs:
    """
    Pipeline كامل من الصورة الخام لـ CardROIs جاهزة للـ OCR

    Args:
        image: الصورة الخام
        use_yolo: لو True استخدم YOLO للـ field detection
        yolo_fields: الـ fields المكتشفة من YOLO (لو use_yolo=True)

    Returns:
        CardROIs object بكل الحقول المستخرجة والمعالجة
    """
    # ── 1. Quality Assessment ──────────────────────────
    quality = assess_quality(image)
    logger.info(f"Image quality: {quality['overall']:.2f} | {quality}")

    if not quality["acceptable"]:
        logger.warning(f"Low quality image (score: {quality['overall']:.2f}), proceeding anyway")

    # ── 2. Resize + Basic Enhancement ──────────────────
    image = resize_to_standard(image, target_width=1200)
    image = remove_noise(image)
    image = enhance_contrast(image)

    # ── 3. Perspective Correction ──────────────────────
    card = auto_detect_and_warp(image)

    # ── 4. Final Sharpening ───────────────────────────
    card = sharpen_image(card, strength=1.2)

    # ── 5. Detect Side ────────────────────────────────
    side = detect_card_side(card)
    logger.info(f"Detected card side: {side}")

    # ── 6. Extract ROIs ──────────────────────────────
    if use_yolo and yolo_fields:
        # استخدم YOLO للـ precision العالية
        raw_rois = yolo_fields
    else:
        # استخدم الـ relative coordinates كـ fallback
        raw_rois = extract_all_rois(card, side=side)

    # ── 7. Remove Background Lines ───────────────────
    card_clean = remove_background_lines(card)

    # ── 8. Per-Field Preprocessing ───────────────────
    processed_rois = {}
    for field_name, roi_img in raw_rois.items():
        if roi_img is None or roi_img.size == 0:
            continue
        processed = preprocess_text_field(roi_img, field_type=field_name)
        processed_rois[field_name] = processed
        logger.debug(f"Processed ROI '{field_name}': {processed.shape}")

    return CardROIs(
        full_card=card,
        name_ar=processed_rois.get("name_ar"),
        name_en=processed_rois.get("name_en"),
        address=processed_rois.get("address"),
        id_number=processed_rois.get("id_number"),
        birth_date=processed_rois.get("birth_date"),
        gender=processed_rois.get("gender"),
        nationality=processed_rois.get("nationality"),
        face=processed_rois.get("face"),
        quality_score=quality["overall"],
    )
