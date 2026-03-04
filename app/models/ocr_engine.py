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
from typing import Optional, Dict

from app.core.config import settings
from app.core.logger import logger


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
          - Digit-only fields (NID, serial, codes): Prefer Paddle digits → Tesseract → EasyOCR
          - Arabic-rich text fields (names, addresses): Prefer PaddleOCR Arabic; fall back to EasyOCR.
          - Other fields: EasyOCR default.
        """
        digit_fields = {"nid", "front_nid", "back_nid", "id_number", "serial", "serial_num", "issue_code"}
        arabic_fields = {"firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"}

        # Digit-only fields
        if field_name in digit_fields:
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
