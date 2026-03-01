"""
Egyptian ID Extraction Pipeline Service
Orchestrates the complete OCR pipeline from image input to structured output.
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.models.id_parser import parse_national_id
from app.utils.image_utils import (
    decode_image,
    deskew,
    full_preprocess_pipeline,
    extract_all_rois,
    preprocess_text_field,
)
from app.utils.text_utils import clean_field
from app.core.config import settings
from app.core.logger import logger


# Field classification
ARABIC_FIELDS = {"name_ar", "address", "nationality", "gender"}
DIGIT_FIELDS = {"id_number"}


@dataclass
class ExtractionResult:
    """Result of ID extraction with confidence scores."""

    extracted: Dict[str, str]
    confidence: Dict[str, Any]
    processing_ms: int
    parsed_id: Optional[Dict[str, Any]] = None


class IDExtractionPipeline:
    """
    Complete pipeline for Egyptian ID card extraction.
    Uses singleton pattern to ensure models are loaded once.
    """

    _instance = None
    _detector: Optional[EgyptianIDDetector] = None
    _ocr: Optional[OCREngine] = None

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the pipeline with models."""
        if self._initialized:
            return

        logger.info("Initializing ID Extraction Pipeline...")
        self._initialized = True

    @classmethod
    def initialize(cls):
        """Initialize models (called at startup)."""
        if cls._instance is None:
            cls._instance = cls()

        if cls._detector is None:
            logger.info("Loading detection models...")
            try:
                cls._detector = EgyptianIDDetector()
            except Exception as e:
                logger.warning(f"Could not load detector: {e}")
                cls._detector = None

        if cls._ocr is None:
            logger.info("Loading OCR models...")
            try:
                cls._ocr = OCREngine()
            except Exception as e:
                logger.warning(f"Could not load OCR: {e}")
                cls._ocr = None

        logger.info("Pipeline initialized")

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Process an image and extract ID information.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary with extracted data, confidence scores, and processing time
        """
        start_time = time.time()

        # 1. Decode image
        try:
            image = decode_image(image_bytes)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return {"error": f"Failed to decode image: {str(e)}", "processing_ms": 0}

        # 2. Preprocess (deskew)
        try:
            image = deskew(image)
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")

        # 3. Detect card (if detector available)
        card_image = image
        if self._detector:
            try:
                card_image = self._detector.crop_card(image)
            except Exception as e:
                logger.warning(f"Card detection failed: {e}")

        # 4. Detect fields (if detector available)
        yolo_fields = {}
        if self._detector:
            try:
                yolo_fields = self._detector.crop_fields(card_image)
            except Exception as e:
                logger.warning(f"Field detection failed: {e}")

        # 5. Extract ROIs (fallback to coordinates)
        if not yolo_fields:
            try:
                from app.utils.image_utils import detect_card_side

                side = detect_card_side(card_image)
                yolo_fields = extract_all_rois(card_image, side=side)
                # Convert to tuple format for OCR
                yolo_fields = {k: (v, 0.5) for k, v in yolo_fields.items()}
            except Exception as e:
                logger.warning(f"ROI extraction failed: {e}")
                yolo_fields = {}

        if not yolo_fields:
            return {
                "error": "No fields detected",
                "processing_ms": int((time.time() - start_time) * 1000),
            }

        # 6. OCR for each field
        extracted = {}
        confidence_scores = {}

        for field_name, (field_img, det_conf) in yolo_fields.items():
            try:
                # Preprocess for OCR
                processed = preprocess_text_field(field_img, field_type=field_name)

                # Run OCR
                if self._ocr:
                    ocr_result = self._ocr.ocr_field(processed, field_name)
                    text = ocr_result.text
                    ocr_conf = ocr_result.confidence
                else:
                    # Fallback: return empty
                    text = ""
                    ocr_conf = 0.0

                # Clean text
                cleaned = clean_field(text, field_name)
                extracted[field_name] = cleaned

                # Calculate confidence
                conf = (det_conf + ocr_conf) / 2 if det_conf > 0 else ocr_conf
                confidence_scores[field_name] = round(conf, 3)

            except Exception as e:
                logger.error(f"OCR failed for field {field_name}: {e}")
                extracted[field_name] = ""
                confidence_scores[field_name] = 0.0

        # 7. Parse national ID
        parsed_info = None
        if "id_number" in extracted and extracted["id_number"]:
            try:
                parsed_info = parse_national_id(extracted["id_number"])
                if not parsed_info.valid:
                    logger.warning(f"Invalid ID parsed: {parsed_info.error}")
            except Exception as e:
                logger.error(f"ID parsing failed: {e}")

        # 8. Calculate overall confidence
        if confidence_scores:
            avg_conf = float(np.mean(list(confidence_scores.values())))
        else:
            avg_conf = 0.0

        level = "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"

        # 9. Build response
        processing_ms = int((time.time() - start_time) * 1000)

        result = {
            "extracted": extracted,
            "confidence": {
                "overall": round(avg_conf, 3),
                "level": level,
                "per_field": confidence_scores,
            },
            "processing_ms": processing_ms,
        }

        if parsed_info and parsed_info.valid:
            result["parsed_id"] = {
                "birth_date": parsed_info.birth_date,
                "governorate": parsed_info.governorate,
                "gender": parsed_info.gender,
                "age": parsed_info.age,
                "sequence": parsed_info.sequence,
            }

        logger.info(f"Extraction complete in {processing_ms}ms")
        return result


def get_pipeline() -> IDExtractionPipeline:
    """Get or create the pipeline instance."""
    return IDExtractionPipeline()
