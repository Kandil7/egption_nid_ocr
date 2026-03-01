"""
OCR Engine for Egyptian ID Card
EasyOCR-based with light preprocessing.
"""

import os
import time
import numpy as np
import cv2
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict

from app.core.config import settings
from app.core.logger import logger


class OCRMode(str, Enum):
    """OCR engine modes."""

    EASYOCR = "easyocr"


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


class OCREngine:
    """
    Unified OCR engine using EasyOCR.
    """

    _instance = None
    _easy: Optional[EasyOCREngine] = None

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

        # Initialize EasyOCR
        self._easy = EasyOCREngine()

        self._initialized = True
        logger.info("OCR Engine ready")

    def ocr_field(self, image: np.ndarray, field_name: str) -> OCRResult:
        """
        Run OCR on a field.

        Args:
            image: Field image
            field_name: Name of the field

        Returns:
            OCRResult with extracted text and metadata
        """
        # Determine if this is a digit-only field
        digits_only = field_name in ["nid", "serial", "serial_num", "issue_code"]

        # Run OCR
        if self._easy:
            return self._easy.run(image, digits_only=digits_only)

        return OCRResult(text="", confidence=0.0, engine_used=OCRMode.EASYOCR, latency_ms=0)

    def get_available_engines(self) -> Dict[str, bool]:
        """Check which engines are available."""
        return {
            "easyocr": self._easy is not None and self._easy.reader is not None,
        }
