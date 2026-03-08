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
        """
        Run digit recognition on a cropped field image.
        
        Optimized for Egyptian NID using ara_number_id.traineddata
        which specializes in Arabic-Indic and European digit recognition.
        """
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

            # Log image properties for debugging
            logger.info(f"Tesseract input: shape={gray.shape}, dtype={gray.dtype}, min={gray.min()}, max={gray.max()}")

            # Apply thresholding for better digit recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Upscale for better recognition (NID digits are often small)
            h, w = thresh.shape
            if h < 100:
                scale = 100 / h
                thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                logger.info(f"Tesseract upscaled: {h}x{w} → {thresh.shape[0]}x{thresh.shape[1]}")

            # Configure tesseract for digits only with single text line mode
            # --psm 7: Treat the image as a single text line
            custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

            # Use ara_number_id.traineddata - specialized for Arabic numbers
            # This trained data recognizes both Arabic-Indic (٠١٢٣) and European (0123) digits
            text = ""
            confidence = 0.0
            
            try:
                # Get detailed OCR result with confidence
                data = self._client.image_to_data(
                    thresh, 
                    config=custom_config, 
                    lang='ara_number_id',
                    output_type=self._client.Output.DICT
                )
                
                # Extract text and confidence from results
                for i, box_text in enumerate(data['text']):
                    if box_text.strip():
                        text += box_text
                        conf = float(data['conf'][i])
                        confidence = max(confidence, conf)
                
                logger.info(f"Tesseract ara_number_id result: '{text}' (conf: {confidence:.1f}%, latency: {int((time.time() - t0) * 1000)}ms)")
                
            except Exception as e:
                logger.warning(f"ara_number_id failed: {e}, falling back to eng")
                # Fallback to English digits
                data = self._client.image_to_data(
                    thresh, 
                    config=custom_config, 
                    lang='eng',
                    output_type=self._client.Output.DICT
                )
                
                for i, box_text in enumerate(data['text']):
                    if box_text.strip():
                        text += box_text
                        conf = float(data['conf'][i])
                        confidence = max(confidence, conf)

            # Clean text - keep only digits
            cleaned_text = re.sub(r'\D', '', text.strip())
            
            # Normalize Arabic-Indic digits to European
            arabic_indic = "٠١٢٣٤٥٦٧٨٩"
            european = "0123456789"
            translation_table = str.maketrans(arabic_indic, european)
            cleaned_text = cleaned_text.translate(translation_table)

            logger.info(f"Tesseract digit OCR - raw: '{text}', cleaned: '{cleaned_text}' (len={len(cleaned_text)})")

            # Calculate confidence based on text length (NID should be 14 digits)
            # and Tesseract's confidence score
            length_score = min(1.0, len(cleaned_text) / 14.0) if cleaned_text else 0.0
            tesseract_conf = confidence / 100.0 if confidence > 0 else 0.0
            
            # Combined confidence: weight Tesseract's confidence higher
            final_confidence = (tesseract_conf * 0.7 + length_score * 0.3) if cleaned_text else 0.0

            return OCRResult(
                text=cleaned_text,
                confidence=final_confidence,
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

    def _vote_nid_results(self, results: list[str]) -> str:
        """Vote character-by-character on multiple 14-digit NID candidates."""
        if not results:
            return ""
        if len(results) == 1:
            return results[0]

        # Ensure all strings are 14 digits by padding or truncating if necessary
        # Usually they are already filtered to 14, but just in case
        padded_results = [r[:14].ljust(14, '0') for r in results]
        
        voted = []
        for i in range(14):
            chars_at_i = [r[i] for r in padded_results]
            # Get most common character
            most_common = max(set(chars_at_i), key=chars_at_i.count)
            voted.append(most_common)
            
        return "".join(voted)

    def ocr_field(self, image: np.ndarray, field_name: str) -> OCRResult:
        """
        Run OCR on a field with optimized engine routing.
        
        Strategy:
          - NID fields: Tesseract (ara_number_id) → Paddle digits → EasyOCR
          - Other digit fields: Paddle digits → Tesseract → EasyOCR
          - Arabic text fields: PaddleOCR Arabic → EasyOCR
          - Other fields: EasyOCR default
          
        Note: ara_number_id.traineddata is specifically trained for Arabic number
        recognition and provides superior accuracy for Egyptian NID digits.
        """
        # NID-specific fields (14-digit national ID)
        nid_fields = {"nid", "front_nid", "back_nid", "id_number"}
        
        # Other digit fields (serial, codes, etc.)
        digit_fields = {"serial", "serial_num", "issue_code"}
        
        # Arabic text fields
        arabic_fields = {"firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"}

        # NID fields - prioritize ensemble voting
        if field_name in nid_fields:
            logger.debug(f"OCR field '{field_name}' - NID field, using ensemble voting")
            t0 = time.time()
            candidates = []
            best_confidence = 0.0
            
            # 1. Tesseract with ara_number_id.traineddata
            if self._tesseract and self._tesseract._client is not None:
                tesseract_result = self._tesseract.run_digits(image)
                if tesseract_result.text and len(tesseract_result.text) >= 10:
                    candidates.append(("tesseract", tesseract_result.text, tesseract_result.confidence))
                    best_confidence = max(best_confidence, tesseract_result.confidence)
            
            # 2. Paddle digit recognizer
            if self._paddle and self._paddle._digit_reader is not None:
                paddle_result = self._paddle.run_digits(image)
                if paddle_result.text and len(paddle_result.text) >= 10:
                    # Clean and normalize paddle result just in case
                    p_text = re.sub(r'\D', '', paddle_result.text)
                    if len(p_text) >= 10:
                        candidates.append(("paddle", p_text, paddle_result.confidence))
                        best_confidence = max(best_confidence, paddle_result.confidence)
            
            # 3. EasyOCR with numeric allowlist
            if self._easy:
                # We can try multi-scale for EasyOCR
                easy_result = self._easy.run(image, digits_only=True)
                if easy_result.text and len(easy_result.text) >= 10:
                    e_text = re.sub(r'\D', '', easy_result.text)
                    if len(e_text) >= 10:
                        candidates.append(("easyocr", e_text, easy_result.confidence))
                        best_confidence = max(best_confidence, easy_result.confidence)

            # Analyze candidates
            if not candidates:
                return OCRResult(text="", confidence=0.0, engine_used=OCRMode.TESSERACT, latency_ms=int((time.time() - t0) * 1000))

            results_14 = [c[1] for c in candidates if len(c[1]) == 14]
            
            if len(results_14) >= 2:
                # We have multiple 14-digit results, perform ensemble voting
                voted_text = self._vote_nid_results(results_14)
                logger.info(f"NID ensemble: {results_14} → voted: '{voted_text}'")
                
                # Check if voted text passes checksum
                from app.models.id_parser import validate_nid_checksum
                if validate_nid_checksum(voted_text):
                    best_confidence = min(0.99, best_confidence + 0.1) # Boost for valid checksum
                
                return OCRResult(
                    text=voted_text,
                    confidence=best_confidence,
                    engine_used=OCRMode.TESSERACT,
                    latency_ms=int((time.time() - t0) * 1000),
                )
            elif len(results_14) == 1:
                # Only one 14-digit result
                return OCRResult(
                    text=results_14[0],
                    confidence=best_confidence,
                    engine_used=OCRMode.TESSERACT,
                    latency_ms=int((time.time() - t0) * 1000),
                )
            else:
                # No 14-digit results, pick the longest/best candidate
                best_candidate = max(candidates, key=lambda x: (len(x[1]), x[2]))
                return OCRResult(
                    text=best_candidate[1],
                    confidence=best_candidate[2],
                    engine_used=OCRMode.TESSERACT, # Proxy for ensemble
                    latency_ms=int((time.time() - t0) * 1000),
                )

        # Other digit-only fields
        if field_name in digit_fields:
            logger.debug(f"OCR field '{field_name}' - digit field")
            
            # 1. Paddle digit recognizer
            if self._paddle and self._paddle._digit_reader is not None:
                paddle_result = self._paddle.run_digits(image)
                if paddle_result.text and len(paddle_result.text) > 0:
                    return paddle_result
            
            # 2. Tesseract with ara_number_id
            if self._tesseract and self._tesseract._client is not None:
                tesseract_result = self._tesseract.run_digits(image)
                if tesseract_result.text:
                    return tesseract_result
            
            # 3. EasyOCR fallback
            if self._easy:
                return self._easy.run(image, digits_only=True)
            
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
