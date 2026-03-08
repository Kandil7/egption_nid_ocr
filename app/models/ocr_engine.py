"""
OCR Engine for Egyptian ID Card
EasyOCR-based with light preprocessing.
"""

import os
import time
import numpy as np
import cv2
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List

from app.core.config import settings
from app.core.logger import logger
from app.utils.ocr_preprocess import preprocess_nid_field, preprocess_nid_multi_scale, preprocess_nid_variations
from app.models.nid_extractor import get_nid_extractor


class OCRMode(str, Enum):
    """OCR engine modes."""

    EASYOCR = "easyocr"
    PADDLE_AR = "paddle_ar"
    TESSERACT = "tesseract"


@dataclass
class OCRResult:
    """Result from OCR operation."""

    text: str
    confidence: float
    engine_used: OCRMode
    latency_ms: int


class EasyOCREngine:
    """
    EasyOCR engine - primary OCR for this project.
    """

    def __init__(self):
        """Initialize EasyOCR engine."""
        logger.info("Loading EasyOCR...")

        import easyocr

        self.reader = easyocr.Reader(
            ["ar", "en"],
            gpu=False,
            quantize=True,
            model_storage_directory=settings.MODELS_CACHE_DIR,
            verbose=False,
        )
        logger.info("EasyOCR ready")

    def run(self, image_np: np.ndarray, digits_only: bool = False) -> OCRResult:
        """Run OCR on the given image."""
        t0 = time.time()

        try:
            # Use original image for best results
            # EasyOCR handles preprocessing internally
            kwargs = dict(
                detail=1,
                decoder="greedy",
                beamWidth=1,
            )

            if digits_only:
                kwargs["allowlist"] = "0123456789"

            results = self.reader.readtext(image_np, **kwargs)

            if not results:
                # Try with slight preprocessing if empty
                gray = (
                    cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                    if len(image_np.shape) == 3
                    else image_np
                )
                if gray.shape[0] < 50:
                    # Upscale small images
                    scale = 50 / gray.shape[0]
                    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    if len(image_np.shape) == 3:
                        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    results = self.reader.readtext(gray, **kwargs)

            if not results:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used=OCRMode.EASYOCR,
                    latency_ms=int((time.time() - t0) * 1000),
                )

            # Extract text and confidence
            texts = []
            confs = []
            for r in results:
                if len(r) >= 3:
                    texts.append(r[1])
                    confs.append(r[2])

            text = " ".join(texts)
            conf = float(np.mean(confs)) if confs else 0.0

            return OCRResult(
                text=text,
                confidence=conf,
                engine_used=OCRMode.EASYOCR,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)

    def run_nid(self, image_np: np.ndarray) -> OCRResult:
        """
        Run OCR on NID field with optimized preprocessing.
        Tries original image first, then variations only if needed.

        Args:
            image_np: Input image of NID field

        Returns:
            OCRResult with best recognition result
        """
        t0 = time.time()
        best_result = OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)
        kwargs = dict(detail=1, decoder="greedy", beamWidth=1, allowlist="0123456789")

        try:
            # Strategy 1: Try original image first (fastest path)
            # Upscale if small
            img = image_np
            if len(img.shape) == 3:
                h, w = img.shape[:2]
            else:
                h, w = img.shape
            
            if h < 80:
                scale = 80 / h
                img = cv2.resize(img, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)
            
            results = self.reader.readtext(img, **kwargs)
            if results:
                texts = [r[1] for r in results if len(r) >= 3]
                confs = [r[2] for r in results if len(r) >= 3]
                text = " ".join(texts)
                conf = float(np.mean(confs)) if confs else 0.0
                digit_count = sum(c.isdigit() for c in text)
                
                if digit_count >= 14 and conf > 0.5:
                    # Perfect result on first try — skip variations
                    return OCRResult(
                        text=text,
                        confidence=conf,
                        engine_used=OCRMode.EASYOCR,
                        latency_ms=int((time.time() - t0) * 1000),
                    )
                
                score = conf * (min(digit_count, 14) / 14.0)
                if score > best_result.confidence:
                    best_result = OCRResult(
                        text=text, confidence=conf,
                        engine_used=OCRMode.EASYOCR,
                        latency_ms=int((time.time() - t0) * 1000),
                    )

            # Strategy 2: Only try variations if first pass was poor
            if len(re.sub(r'\D', '', best_result.text)) < 10:
                variations = preprocess_nid_variations(image_np)
                # Try only CLAHE and Otsu (indices 2, 3) — skip redundant ones
                for idx in [2, 3]:
                    if idx >= len(variations):
                        continue
                    variant_img = variations[idx]
                    results = self.reader.readtext(variant_img, **kwargs)

                    if results:
                        texts = [r[1] for r in results if len(r) >= 3]
                        confs = [r[2] for r in results if len(r) >= 3]
                        text = " ".join(texts)
                        conf = float(np.mean(confs)) if confs else 0.0
                        digit_count = sum(c.isdigit() for c in text)
                        score = conf * (min(digit_count, 14) / 14.0)

                        if score > best_result.confidence:
                            best_result = OCRResult(
                                text=text, confidence=conf,
                                engine_used=OCRMode.EASYOCR,
                                latency_ms=int((time.time() - t0) * 1000),
                            )

                        if digit_count >= 14 and conf > 0.5:
                            break

            # Strategy 3: If still poor, try without allowlist (catch Arabic-Indic digits)
            if best_result.confidence < 0.4 or len(re.sub(r'\D', '', best_result.text)) < 10:
                results = self.reader.readtext(img, detail=1, decoder="greedy", beamWidth=1)
                
                if results:
                    texts = [r[1] for r in results if len(r) >= 3]
                    confs = [r[2] for r in results if len(r) >= 3]
                    text = " ".join(texts)
                    conf = float(np.mean(confs)) if confs else 0.0
                    
                    from app.utils.text_utils import _normalize_digits, _fix_common_digit_ocr_errors
                    text = _normalize_digits(text)
                    text = _fix_common_digit_ocr_errors(text)
                    
                    digit_count = sum(c.isdigit() for c in text)
                    if digit_count >= 10:
                        score = conf * (min(digit_count, 14) / 14.0)
                        if score > best_result.confidence:
                            best_result = OCRResult(
                                text=text, confidence=conf,
                                engine_used=OCRMode.EASYOCR,
                                latency_ms=int((time.time() - t0) * 1000),
                            )

            logger.info(f"NID OCR complete: '{best_result.text}' (conf: {best_result.confidence:.2f}, time: {best_result.latency_ms}ms)")
            return best_result

        except Exception as e:
            logger.error(f"EasyOCR NID error: {e}")
            return best_result


class PaddleOCREngine:
    """
    PaddleOCR-based recognizer.

    We use:
      - Arabic model for Arabic / mixed-name and address fields.
      - Optionally English/multilingual models for other fields in future.

    Detection is handled separately (YOLO / template ROIs), so we only enable recognition here.
    """

    def __init__(self) -> None:
        """Initialize PaddleOCR engines."""
        self._ar_reader = None
        self._digit_reader = None

        try:
            from paddleocr import PaddleOCR  # type: ignore

            use_gpu = bool(getattr(settings, "PADDLE_USE_GPU", False))

            # Arabic text recognizer (names, address)
            logger.info("Loading PaddleOCR Arabic recognizer (PP-OCRv3/PP-OCRv4 mobile)...")
            ar_kwargs = dict(
                lang="ar",
                use_angle_cls=False,
                det=False,
                rec=True,
                use_gpu=use_gpu,
                show_log=False,
            )
            if getattr(settings, "PADDLE_AR_REC_MODEL_DIR", ""):
                ar_kwargs["rec_model_dir"] = settings.PADDLE_AR_REC_MODEL_DIR
            self._ar_reader = PaddleOCR(**ar_kwargs)
            logger.info("PaddleOCR Arabic recognizer ready")

            # Digit / Latin recognizer (PP-OCRv4 mobile_rec)
            logger.info("Loading PaddleOCR digit/Latin recognizer (PP-OCRv4 mobile_rec)...")
            digit_kwargs = dict(
                det=False,
                rec=True,
                use_angle_cls=False,
                use_gpu=use_gpu,
                show_log=False,
            )
            if getattr(settings, "PADDLE_DIGIT_REC_MODEL_DIR", ""):
                digit_kwargs["rec_model_dir"] = settings.PADDLE_DIGIT_REC_MODEL_DIR
            self._digit_reader = PaddleOCR(**digit_kwargs)
            logger.info("PaddleOCR digit/Latin recognizer ready")
        except Exception as e:
            logger.warning(f"Could not initialize PaddleOCR engines: {e}")
            self._ar_reader = None
            self._digit_reader = None

    def available(self) -> bool:
        """Return True if at least one PaddleOCR reader is ready."""
        return self._ar_reader is not None or self._digit_reader is not None

    def run_arabic(self, image_np: np.ndarray) -> OCRResult:
        """Run Arabic recognition on a cropped field image."""
        t0 = time.time()

        if self._ar_reader is None:
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=0,
            )

        try:
            # PaddleOCR accepts numpy arrays directly.
            # We disable detection (det=False in constructor) and let it only run recognition.
            ocr_out = self._ar_reader.ocr(image_np, cls=False)

            texts = []
            confs = []

            # ocr_out is typically: [[ [box], (text, score) ], ... ]
            for line in ocr_out:
                for box, (txt, score) in line:
                    if txt:
                        texts.append(txt)
                        confs.append(float(score))

            text = " ".join(texts)
            conf = float(np.mean(confs)) if confs else 0.0

            return OCRResult(
                text=text,
                confidence=conf,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            logger.error(f"PaddleOCR Arabic error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=int((time.time() - t0) * 1000),
            )

    def run_digits(self, image_np: np.ndarray) -> OCRResult:
        """Run digit / Latin recognition on a cropped field image using PP-OCRv4-style model."""
        t0 = time.time()

        if self._digit_reader is None:
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=0,
            )

        try:
            ocr_out = self._digit_reader.ocr(image_np, cls=False)

            # Some PaddleOCR versions may return None or an empty list on failure.
            if not ocr_out:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used=OCRMode.PADDLE_AR,
                    latency_ms=int((time.time() - t0) * 1000),
                )

            texts = []
            confs = []
            for line in ocr_out:
                if not line:
                    continue
                for item in line:
                    if not item or len(item) < 2:
                        continue
                    # item is typically: [box, (text, score)]
                    box, result = item[0], item[1]
                    if not result or len(result) < 2:
                        continue
                    txt, score = result[0], result[1]
                    if txt:
                        texts.append(txt)
                        confs.append(float(score))

            text = " ".join(texts)
            conf = float(np.mean(confs)) if confs else 0.0

            return OCRResult(
                text=text,
                confidence=conf,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            logger.error(f"PaddleOCR digit/Latin error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.PADDLE_AR,
                latency_ms=int((time.time() - t0) * 1000),
            )


class TesseractEngine:
    """
    Tesseract OCR for digit recognition.
    Uses Arabic trained data for better number recognition.
    """

    def __init__(self):
        """Initialize Tesseract engine."""
        self._client = None
        
        try:
            import pytesseract
            
            # Set custom tessdata directory via environment variable
            tessdata_dir = os.path.abspath(getattr(settings, "TESSDATA_DIR", "./weights"))
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            
            # Set Tesseract executable path
            tesseract_path = self._find_tesseract()
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            logger.info(f"Tesseract OCR ready - tessdata dir: {tessdata_dir}, tesseract path: {tesseract_path}")
            self._client = pytesseract
        except Exception as e:
            logger.warning(f"Could not initialize Tesseract OCR: {e}")
            self._client = None

    def _find_tesseract(self) -> str:
        """Find Tesseract executable path."""
        # Common Windows paths
        paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Try PATH
        return "tesseract"

    def run_digits(self, image_np: np.ndarray) -> OCRResult:
        """Run digit recognition on a cropped field image."""
        t0 = time.time()

        if self._client is None:
            logger.debug("Tesseract client not available, skipping digit OCR")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.TESSERACT,
                latency_ms=0,
            )

        try:
            # Convert to grayscale if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_np.copy()

            # Apply thresholding for better digit recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Configure tesseract for digits only
            custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"

            # Try with Arabic number trained data if available
            try:
                text = self._client.image_to_string(thresh, config=custom_config, lang='ara_number_id')
            except:
                # Fallback to English digits
                text = self._client.image_to_string(thresh, config=custom_config, lang='eng')

            # Clean text - keep only digits
            cleaned_text = re.sub(r'\D', '', text.strip())

            logger.debug(f"Tesseract digit OCR - raw: '{text}', cleaned: '{cleaned_text}'")

            # Calculate confidence based on text length (NID should be 14 digits)
            confidence = min(1.0, len(cleaned_text) / 14.0) if cleaned_text else 0.0

            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                engine_used=OCRMode.TESSERACT,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=OCRMode.TESSERACT,
                latency_ms=int((time.time() - t0) * 1000),
            )

    def run_nid_tesseract(self, image_np: np.ndarray) -> OCRResult:
        """
        Run NID recognition using Tesseract with ara_number_id trained data.
        Uses multiple preprocessing approaches for best results.

        Args:
            image_np: Input image of NID field

        Returns:
            OCRResult with best recognition result
        """
        t0 = time.time()
        best_result = OCRResult(text="", confidence=0.0, engine_used=OCRMode.TESSERACT, latency_ms=0)

        if self._client is None:
            logger.debug("Tesseract client not available for NID OCR")
            return best_result

        try:
            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_np.copy()

            # Upscale for better recognition (3x for small text)
            h, w = gray.shape
            target_height = max(80, h)  # Increased minimum height
            scale = target_height / h
            upscaled = cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(upscaled, None, h=15, templateWindowSize=7, searchWindowSize=21)

            # Preprocessing variations - comprehensive set
            variations = []

            # 1. Otsu threshold on denoised
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variations.append(("otsu", otsu))

            # 2. Inverted Otsu
            _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variations.append(("otsu_inv", otsu_inv))

            # 3. CLAHE + Otsu
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(denoised)
            _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variations.append(("clahe_otsu", clahe_otsu))

            # 4. Adaptive Gaussian
            adaptive = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
            variations.append(("adaptive_gauss", adaptive))

            # 5. Adaptive Mean
            adaptive_mean = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
            variations.append(("adaptive_mean", adaptive_mean))

            # 6. Morphological closing + Otsu (connects broken digit parts)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            _, closed_otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variations.append(("morph_otsu", closed_otsu))

            # Tesseract config for NID - single line of digits
            custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

            results_by_digits = {}  # Store results by digit count for ensemble

            for name, thresh_img in variations:
                try:
                    # Use ara_number_id trained data
                    text = self._client.image_to_string(thresh_img, config=custom_config, lang='ara_number_id')
                    cleaned = re.sub(r'\D', '', text.strip())

                    # Fix common OCR errors
                    from app.utils.text_utils import _normalize_digits, _fix_common_digit_ocr_errors, _fix_nid_century_digit
                    cleaned = _normalize_digits(cleaned)
                    cleaned = _fix_common_digit_ocr_errors(cleaned)

                    digit_count = len(cleaned)
                    logger.debug(f"Tesseract NID ({name}): raw='{text}', cleaned='{cleaned}', digits={digit_count}")

                    # Store for ensemble voting
                    if digit_count > 0:
                        results_by_digits[digit_count] = results_by_digits.get(digit_count, [])
                        results_by_digits[digit_count].append(cleaned)

                        # Update best if this has more digits
                        if digit_count > len(re.sub(r'\D', '', best_result.text)):
                            conf = min(1.0, digit_count / 14.0)
                            best_result = OCRResult(
                                text=_fix_nid_century_digit(cleaned),
                                confidence=conf,
                                engine_used=OCRMode.TESSERACT,
                                latency_ms=int((time.time() - t0) * 1000),
                            )

                            # Early exit for perfect result
                            if digit_count == 14:
                                logger.debug(f"Tesseract NID early exit: '{best_result.text}'")
                                break

                except Exception as e:
                    logger.debug(f"Tesseract NID ({name}) error: {e}")
                    continue

            # Ensemble: If we have multiple 14-digit results, use voting
            if 14 in results_by_digits and len(results_by_digits[14]) >= 2:
                voted = self._vote_nid_results(results_by_digits[14])
                if voted:
                    logger.debug(f"Tesseract NID ensemble voted: '{voted}'")
                    best_result = OCRResult(
                        text=_fix_nid_century_digit(voted),
                        confidence=0.95,
                        engine_used=OCRMode.TESSERACT,
                        latency_ms=int((time.time() - t0) * 1000),
                    )

            # If no result, try without lang restriction
            if len(re.sub(r'\D', '', best_result.text)) < 10:
                try:
                    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = self._client.image_to_string(otsu, config=custom_config, lang='eng')
                    cleaned = re.sub(r'\D', '', text.strip())
                    cleaned = _normalize_digits(cleaned)
                    cleaned = _fix_common_digit_ocr_errors(cleaned)

                    logger.debug(f"Tesseract NID (eng fallback): raw='{text}', cleaned='{cleaned}'")

                    if len(cleaned) >= 10:
                        best_result = OCRResult(
                            text=_fix_nid_century_digit(cleaned),
                            confidence=min(1.0, len(cleaned) / 14.0),
                            engine_used=OCRMode.TESSERACT,
                            latency_ms=int((time.time() - t0) * 1000),
                        )
                except Exception as e:
                    logger.debug(f"Tesseract NID eng fallback error: {e}")

            return best_result

        except Exception as e:
            logger.error(f"Tesseract NID error: {e}")
            return best_result

    def _vote_nid_results(self, results: list) -> str:
        """
        Vote on multiple NID results to get the most likely correct one.

        Args:
            results: List of 14-digit NID strings

        Returns:
            Voted NID string
        """
        if not results:
            return ""

        # Position-wise voting
        voted_digits = []
        for pos in range(14):
            digit_votes = [r[pos] for r in results if len(r) == 14 and pos < len(r)]
            if digit_votes:
                # Most common digit at this position
                voted = max(set(digit_votes), key=digit_votes.count)
                voted_digits.append(voted)
            else:
                voted_digits.append('0')

        return ''.join(voted_digits)


class OCREngine:
    """
    Unified OCR engine using EasyOCR, PaddleOCR, and Tesseract.
    """

    _instance = None
    _easy: Optional[EasyOCREngine] = None
    _paddle: Optional[PaddleOCREngine] = None
    _tesseract: Optional[TesseractEngine] = None

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize OCR engine."""
        if self._initialized:
            return

        logger.info("Initializing OCR Engine...")

        # Initialize EasyOCR (digits + fallback)
        self._easy = EasyOCREngine()

        # Initialize PaddleOCR (Arabic names, address)
        try:
            paddle_engine = PaddleOCREngine()
            if paddle_engine.available():
                self._paddle = paddle_engine
            else:
                self._paddle = None
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR engine: {e}")
            self._paddle = None

        # Initialize Tesseract (digits fallback)
        try:
            self._tesseract = TesseractEngine()
        except Exception as e:
            logger.warning(f"Failed to initialize Tesseract OCR: {e}")
            self._tesseract = None

        self._initialized = True
        logger.info("OCR Engine ready")

    def ocr_field(self, image: np.ndarray, field_name: str) -> OCRResult:
        """
        Run OCR on a field.

        Strategy:
          - NID fields: Specialized NID extractor + Tesseract + EasyOCR ensemble
          - Other digit-only fields (serial, codes): Paddle digits → Tesseract → EasyOCR
          - Arabic-rich text fields (names, addresses): PaddleOCR Arabic; fall back to EasyOCR.
          - Other fields: EasyOCR default.
        """
        nid_fields = {"nid", "front_nid", "back_nid", "id_number"}
        other_digit_fields = {"serial", "serial_num", "issue_code"}
        arabic_fields = {"firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"}

        # NID fields - use specialized extractor + ensemble
        if field_name in nid_fields:
            logger.debug(f"OCR field '{field_name}' - using specialized NID extractor + ensemble")
            t0 = time.time()
            
            # 1. Specialized NID extractor (contour-based, multi-region)
            nid_extractor = get_nid_extractor()
            
            # Simple wrapper to use EasyOCR reader as recognize_func
            def recognize(img):
                if not self._easy: return ""
                res = self._easy.reader.readtext(img, detail=0, allowlist="0123456789")
                return "".join(res)
                
            extracted_nid, nid_conf = nid_extractor.extract(image, recognize_func=recognize)
            
            if extracted_nid and len(extracted_nid) == 14:
                logger.info(f"NID extractor found: '{extracted_nid}' (conf: {nid_conf:.2f})")
                return OCRResult(
                    text=extracted_nid,
                    confidence=nid_conf,
                    engine_used=OCRMode.TESSERACT,
                    latency_ms=int((time.time() - t0) * 1000),
                )
            
            # 2. Fallback to Tesseract + EasyOCR ensemble
            tesseract_result = OCRResult(text="", confidence=0.0, engine_used=OCRMode.TESSERACT, latency_ms=0)
            if self._tesseract and self._tesseract._client is not None:
                tesseract_result = self._tesseract.run_nid_tesseract(image)
                if tesseract_result.text and len(tesseract_result.text) == 14:
                    logger.debug(f"Tesseract NID: '{tesseract_result.text}' (conf: {tesseract_result.confidence:.2f})")
                    return tesseract_result

            # 3. EasyOCR specialized NID method
            easy_result = OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)
            if self._easy:
                easy_result = self._easy.run_nid(image)
                # Clean and normalize EasyOCR result
                from app.utils.text_utils import _normalize_digits, _fix_common_digit_ocr_errors, _fix_nid_century_digit
                cleaned = _normalize_digits(easy_result.text)
                cleaned = _fix_common_digit_ocr_errors(cleaned)
                cleaned = _fix_nid_century_digit(cleaned)
                easy_result.text = cleaned

            # 4. Ensemble voting if we have multiple 14-digit results
            results_14 = []
            if tesseract_result.text and len(tesseract_result.text) == 14:
                results_14.append(tesseract_result.text)
            if easy_result.text and len(easy_result.text) == 14:
                results_14.append(easy_result.text)

            if len(results_14) >= 2:
                voted = self._vote_nid_results(results_14)
                logger.info(f"NID ensemble: {results_14} → voted: '{voted}'")
                return OCRResult(
                    text=voted,
                    confidence=0.95,
                    engine_used=OCRMode.TESSERACT,
                    latency_ms=int((time.time() - t0) * 1000),
                )

            # 5. Return best single result
            if tesseract_result.text and len(tesseract_result.text) >= 10:
                return tesseract_result
            if easy_result.text and len(easy_result.text) >= 10:
                return easy_result
            if tesseract_result.text:
                return tesseract_result
            return easy_result

        # Other digit-only fields (serial, issue code, etc.)
        if field_name in other_digit_fields:
            logger.debug(f"OCR field '{field_name}' - digit field, checking engines...")

            # 1. Prefer Paddle digit recognizer (PP-OCRv4 mobile_rec or custom) if available
            if self._paddle and self._paddle._digit_reader is not None:
                logger.debug("Using PaddleOCR digits")
                paddle_result = self._paddle.run_digits(image)
                # Only use Paddle result if it returns actual digits
                if paddle_result.text and len(paddle_result.text) > 0:
                    logger.debug(f"PaddleOCR digits result: '{paddle_result.text}'")
                    return paddle_result
                logger.debug("PaddleOCR digits returned empty, trying Tesseract...")

            # 2. Try Tesseract with Arabic number trained data
            if self._tesseract and self._tesseract._client is not None:
                logger.debug("Using Tesseract digits")
                tesseract_result = self._tesseract.run_digits(image)
                if tesseract_result.text:
                    logger.debug(f"Tesseract digits result: '{tesseract_result.text}'")
                    return tesseract_result
                logger.debug("Tesseract digits returned empty, trying EasyOCR...")

            # 3. Fallback to EasyOCR with numeric allowlist
            if self._easy:
                logger.debug("Using EasyOCR digits")
                easy_result = self._easy.run(image, digits_only=True)
                if easy_result.text:
                    logger.debug(f"EasyOCR digits result: '{easy_result.text}'")
                return easy_result

            logger.warning(f"No digit OCR engine available for field '{field_name}'")
            return OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)

        # Arabic / mixed-name/address fields → PaddleOCR Arabic first
        if field_name in arabic_fields and self._paddle and self._paddle.available():
            paddle_result = self._paddle.run_arabic(image)
            # If Paddle returns something with non-zero confidence, use it
            if paddle_result.text and paddle_result.confidence > 0:
                return paddle_result

        # Fallback / default path → EasyOCR without digit-only restriction
        if self._easy:
            return self._easy.run(image, digits_only=False)

        return OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)

    def get_available_engines(self) -> Dict[str, bool]:
        """Check which engines are available."""
        return {
            "easyocr": self._easy is not None and self._easy.reader is not None,
            "paddle_ar": self._paddle is not None and self._paddle._ar_reader is not None,
            "paddle_digits": self._paddle is not None and self._paddle._digit_reader is not None,
            "tesseract": self._tesseract is not None and self._tesseract._client is not None,
        }
