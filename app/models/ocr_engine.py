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
                kwargs["allowlist"] = "0123456789٠١٢٣٤٥٦٧٨٩"

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
            if ocr_out and len(ocr_out) > 0 and ocr_out[0] is not None:
                for line in ocr_out:
                    if not line: continue
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
            if not ocr_out or (len(ocr_out) > 0 and ocr_out[0] is None):
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

            # Upscale for better recognition (NID digits are often small)
            h, w = gray.shape
            if h < 100:
                scale = 100 / h
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                logger.info(f"Tesseract upscaled: {h}x{w} → {gray.shape[0]}x{gray.shape[1]}")

            # Do not use char_whitelist with ara_number_id, it causes empty results
            # Try multiple PSM modes and pick the best result
            best_text = ""
            best_confidence = 0.0
            best_psm = 7
            
            for psm in [7, 13, 6]:  # 7=single line, 13=raw line, 6=single block
                config = f"--oem 3 --psm {psm}"
                text = ""
                confidence = 0.0
                
                try:
                    data = self._client.image_to_data(
                        gray, 
                        config=config, 
                        lang='ara_number_id',
                        output_type=self._client.Output.DICT
                    )
                    
                    for i, box_text in enumerate(data['text']):
                        if box_text.strip():
                            text += box_text
                            conf = float(data['conf'][i])
                            confidence = max(confidence, conf)
                    
                    # Normalize Arabic-Indic digits to European immediately
                    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
                    european = "0123456789"
                    translation_table = str.maketrans(arabic_indic, european)
                    cleaned = re.sub(r'[^0-9٠-٩]', '', text.strip())
                    cleaned = cleaned.translate(translation_table)
                    
                    logger.info(f"Tesseract psm={psm}: raw='{text}', cleaned='{cleaned}' (len={len(cleaned)}, conf={confidence:.1f}%)")
                    
                    # Pick best: prefer results closer to 14 digits, then highest confidence
                    # We use a scoring function: confidence minus a penalty for every character away from 14
                    def _score_cand(txt, conf):
                        if not txt: return -999.0
                        return conf - (abs(len(txt) - 14) * 20.0)
                        
                    current_score = _score_cand(cleaned, confidence)
                    best_score = _score_cand(best_text, best_confidence)
                    
                    if current_score > best_score:
                        best_text = cleaned
                        best_confidence = confidence
                        best_psm = psm
                        
                except Exception as e:
                    logger.debug(f"Tesseract psm={psm} failed: {e}")
                    continue
            
            # If ara_number_id failed entirely, try eng as fallback
            if not best_text:
                try:
                    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
                    data = self._client.image_to_data(
                        gray, 
                        config=config, 
                        lang='eng',
                        output_type=self._client.Output.DICT
                    )
                    for i, box_text in enumerate(data['text']):
                        if box_text.strip():
                            best_text += box_text
                            conf = float(data['conf'][i])
                            best_confidence = max(best_confidence, conf)
                    best_text = re.sub(r'\D', '', best_text.strip())
                    logger.info(f"Tesseract eng fallback: '{best_text}' (conf={best_confidence:.1f}%)")
                except Exception as e:
                    logger.warning(f"Tesseract eng fallback failed: {e}")

            cleaned_text = best_text
            confidence = best_confidence
            logger.info(f"Tesseract best result (psm={best_psm}): '{cleaned_text}' (len={len(cleaned_text)}, conf={confidence:.1f}%)")

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

    def ocr_field(self, image: np.ndarray, field_name: str, raw_image: Optional[np.ndarray] = None) -> OCRResult:
        """
        Run OCR on a field with optimized engine routing.

        Strategy:
          - NID fields: Multi-engine voting (Tesseract + EasyOCR + Paddle)
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

        # NID fields - use multi-engine approach
        if field_name in nid_fields:
            logger.info(f"NID OCR: Starting multi-engine extraction for '{field_name}'")
            t0 = time.time()
            results = []
            
            # 1. Tesseract with ara_number_id (best for clean digits)
            if self._tesseract and self._tesseract._client is not None:
                tesseract_result = self._tesseract.run_digits(image)
                if tesseract_result.text:
                    results.append(("tesseract", tesseract_result.text, tesseract_result.confidence))
                    logger.info(f"NID OCR: Tesseract result='{tesseract_result.text}' (len={len(tesseract_result.text)}, conf={tesseract_result.confidence:.2f})")
            
            # 2. EasyOCR with digits_only (good fallback)
            if self._easy and not any(len(r[1]) >= 14 for r in results):
                easy_result = self._easy.run(image, digits_only=True)
                if easy_result.text:
                    results.append(("easyocr", easy_result.text, easy_result.confidence))
                    logger.info(f"NID OCR: EasyOCR result='{easy_result.text}' (len={len(easy_result.text)}, conf={easy_result.confidence:.2f})")
            
            # 3. PaddleOCR digits (if available)
            if self._paddle and self._paddle._digit_reader is not None:
                paddle_result = self._paddle.run_digits(image)
                if paddle_result.text:
                    results.append(("paddle", paddle_result.text, paddle_result.confidence))
                    logger.info(f"NID OCR: PaddleOCR result='{paddle_result.text}' (len={len(paddle_result.text)}, conf={paddle_result.confidence:.2f})")
            
            # Select best result
            if results:
                # Prefer 14-digit results
                for engine, text, conf in results:
                    if len(text) == 14:
                        logger.info(f"NID OCR: Selected {engine} result (14 digits): '{text}'")
                        return OCRResult(
                            text=text,
                            confidence=conf,
                            engine_used=getattr(OCRMode, engine.upper(), OCRMode.EASYOCR),
                            latency_ms=int((time.time() - t0) * 1000),
                        )
                
                # If no 14-digit result, use highest confidence
                best = max(results, key=lambda x: (len(x[1]), x[2]))
                logger.info(f"NID OCR: Selected {best[0]} result (best confidence): '{best[1]}' (len={len(best[1])})")
                return OCRResult(
                    text=best[1],
                    confidence=best[2],
                    engine_used=getattr(OCRMode, best[0].upper(), OCRMode.EASYOCR),
                    latency_ms=int((time.time() - t0) * 1000),
                )
            
            logger.warning(f"NID OCR: All engines failed to extract digits")
            return OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=int((time.time() - t0) * 1000))

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
