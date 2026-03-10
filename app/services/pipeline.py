"""
Egyptian ID Extraction Pipeline Service
Orchestrates the complete OCR pipeline from image input to structured output.
"""

import os
import cv2
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.models.id_parser import parse_national_id
from app.utils.image_utils import (
    decode_image,
    deskew,
    extract_all_rois,
    preprocess_text_field,
    resize_to_standard,
    remove_noise,
    enhance_contrast,
    auto_detect_and_warp,
    remove_background_lines,
    detect_card_side,
)
from app.utils.text_utils import clean_field
from app.utils.cache import ocr_cache
from app.core.config import settings
from app.core.logger import logger


# Field classification
ARABIC_FIELDS = {"name_ar", "address", "nationality", "gender"}
DIGIT_FIELDS = {"id_number"}


class FieldValidator:
    """Cross-field validation for ID data consistency."""

    @staticmethod
    def validate(parsed_id: Optional[Dict], extracted: Dict[str, str]) -> Dict[str, str]:
        """
        Validate consistency between extracted fields and parsed ID data.

        Returns dict of validation errors/warnings.
        """
        errors = {}

        if not parsed_id or not parsed_id.get("valid"):
            return errors

        # Validate NID length
        if "id_number" in extracted:
            nid = extracted["id_number"]
            if len(nid) != 14:
                errors["nid_length"] = f"Expected 14 digits, got {len(nid)}"
            elif not nid.isdigit():
                errors["nid_format"] = "NID contains non-digit characters"
            elif not parsed_id.get("checksum_valid", False):
                errors["nid_checksum"] = "NID checksum validation failed"

        # Validate date formats
        for date_field in ["issue_date", "expiry_date"]:
            if date_field in extracted and extracted[date_field]:
                date_text = extracted[date_field]
                if len(date_text) < 8:  # Too short for any date format
                    errors[f"{date_field}_format"] = "Date appears incomplete"

        return errors


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
    _executor: Optional[ThreadPoolExecutor] = None

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

        # Initialize thread pool executor
        if IDExtractionPipeline._executor is None:
            IDExtractionPipeline._executor = ThreadPoolExecutor(
                max_workers=settings.OCR_CPU_THREADS
            )

        # Auto-initialize models if not already done
        if IDExtractionPipeline._detector is None:
            logger.info("Loading detection models...")
            try:
                IDExtractionPipeline._detector = EgyptianIDDetector()
            except Exception as e:
                logger.warning(f"Could not load detector: {e}")
                IDExtractionPipeline._detector = None

        if IDExtractionPipeline._ocr is None:
            logger.info("Loading OCR models...")
            try:
                IDExtractionPipeline._ocr = OCREngine()
            except Exception as e:
                logger.warning(f"Could not load OCR: {e}")
                IDExtractionPipeline._ocr = None

        self._initialized = True
        logger.info("Pipeline initialized")

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

    def _process_single_field(
        self, field_name: str, field_img: np.ndarray, det_conf: float
    ) -> Tuple[str, str, float]:
        """
        Process a single field - extracted for parallelization.

        Args:
            field_name: Name of the field
            field_img: Field image numpy array
            det_conf: Detection confidence from YOLO

        Returns:
            Tuple of (field_name, cleaned_text, confidence)
        """
        try:
            # Check cache first
            cached = ocr_cache.get(field_img, field_name)
            if cached is not None:
                logger.debug(f"Cache HIT for {field_name}")
                return field_name, cached["text"], cached["confidence"]

            # Log field image properties for debugging NID issues
            if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                logger.info(f"NID field '{field_name}': shape={field_img.shape}, det_conf={det_conf:.2f}")

            # Preprocess for OCR
            processed = preprocess_text_field(field_img, field_type=field_name)

            # Debug: save NID processed image to disk so we can see what Tesseract sees
            if field_name in ["nid", "id_number", "front_nid", "back_nid"]:
                import os
                debug_dir = os.path.join("debug", "nid_viz")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{field_name}_proc_{int(time.time()*1000)}.jpg"), processed)

            # Log NID preprocessing result
            if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                logger.info(f"NID preprocessed: shape={processed.shape}, dtype={processed.dtype}")

            # Run OCR
            if self._ocr:
                ocr_result = self._ocr.ocr_field(processed, field_name, raw_image=field_img)
                text = ocr_result.text
                ocr_conf = ocr_result.confidence
                
                # Log NID OCR result
                if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                    logger.info(f"NID OCR result: text='{text}' (len={len(text)}), conf={ocr_conf:.2f}, engine={ocr_result.engine_used}")
            else:
                # Fallback: return empty
                text = ""
                ocr_conf = 0.0

            # Clean text
            cleaned = clean_field(text, field_name)

            # Cache the result
            ocr_cache.set(field_img, field_name, {"text": text, "confidence": ocr_conf})

            # Calculate confidence
            conf = (det_conf + ocr_conf) / 2 if det_conf > 0 else ocr_conf
            return field_name, cleaned, round(conf, 3)

        except Exception as e:
            logger.error(f"OCR failed for field {field_name}: {e}")
            return field_name, "", 0.0

    def _process_fields_parallel(
        self, yolo_fields: Dict[str, Tuple[np.ndarray, float]]
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Process all fields in parallel using thread pool.

        Args:
            yolo_fields: Dictionary of field_name -> (image, detection_confidence)

        Returns:
            Tuple of (extracted_texts, confidence_scores)
        """
        extracted = {}
        confidence_scores = {}

        # Fields we actually OCR to reduce runtime
        ocr_fields = {
            "nid",
            "id_number",
            "front_nid",
            "back_nid",
            "firstName",
            "lastName",
            "address",
            "addressLine1",
            "addressLine2",
            "serial",
            "serial_num",
            "issue_date",
            "expiry_date",
        }

        # Filter to essential fields
        fields_to_process = {
            k: v for k, v in yolo_fields.items() if k in ocr_fields
        }

        # Submit all tasks
        futures = {
            self._executor.submit(
                self._process_single_field,
                field_name,
                field_img,
                det_conf,
            ): field_name
            for field_name, (field_img, det_conf) in fields_to_process.items()
        }

        # Collect results as they complete
        for future in as_completed(futures):
            field_name, text, conf = future.result()
            extracted[field_name] = text
            confidence_scores[field_name] = conf

        return extracted, confidence_scores

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

        # 2. Preprocess (deskew) - temporarily disabled, causes issues
        # try:
        #     image = deskew(image)
        # except Exception as e:
        #     logger.warning(f"Deskew failed: {e}")

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

        # 5. Extract ROIs (fallback to template-based layout on normalized card)
        if not yolo_fields:
            try:
                # Layout-aware preprocessing for fallback path:
                # - Resize to canonical width
                # - Denoise & enhance contrast
                # - Perspective correction & deskew
                # - Background line removal
                normalized = resize_to_standard(card_image, target_width=settings.TARGET_IMAGE_WIDTH)
                normalized = remove_noise(normalized)
                normalized = enhance_contrast(normalized)
                normalized = auto_detect_and_warp(normalized)
                normalized = remove_background_lines(normalized)

                # Ensure normalized is BGR (3 channels) for extract_all_rois
                if len(normalized.shape) == 2:  # Grayscale
                    normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
                    logger.info("Converted grayscale to BGR for ROI extraction")

                # Detect side (front/back) on normalized card and use template ROIs
                side = detect_card_side(normalized)
                logger.info(f"Template ROI fallback using side='{side}'")
                template_rois = extract_all_rois(normalized, side=side)

                # Convert to tuple format for OCR, assign mid-level detection confidence
                yolo_fields = {k: (v, 0.5) for k, v in template_rois.items()}
                card_image = normalized
            except Exception as e:
                logger.warning(f"ROI extraction failed: {e}")
                yolo_fields = {}

        if not yolo_fields:
            return {
                "error": "No fields detected",
                "processing_ms": int((time.time() - start_time) * 1000),
            }

        # 6. OCR for each field (using parallel processing)
        extracted: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}

        # Use parallel processing
        extracted, confidence_scores = self._process_fields_parallel(yolo_fields)

        # 6b. Combine address lines into single address field if needed
        if "addressLine1" in extracted or "addressLine2" in extracted:
            address_parts = []
            address_conf = []
            
            if "addressLine1" in extracted and extracted["addressLine1"]:
                address_parts.append(extracted["addressLine1"])
                address_conf.append(confidence_scores.get("addressLine1", 0.5))
            
            if "addressLine2" in extracted and extracted["addressLine2"]:
                address_parts.append(extracted["addressLine2"])
                address_conf.append(confidence_scores.get("addressLine2", 0.5))
            
            if address_parts:
                # Combine address lines
                combined_address = " ".join(address_parts)
                combined_conf = float(np.mean(address_conf)) if address_conf else 0.5
                
                # Store as 'address' field (or keep separate lines too)
                if "address" not in extracted or not extracted["address"]:
                    extracted["address"] = combined_address
                    confidence_scores["address"] = combined_conf
                    logger.info(f"Combined address lines: {combined_address}")

        # 7. Parse national ID
        parsed_info = None
        if "id_number" in extracted and extracted["id_number"]:
            try:
                parsed_info = parse_national_id(extracted["id_number"])
                if not parsed_info.valid:
                    logger.warning(f"Invalid ID parsed: {parsed_info.error}")
            except Exception as e:
                logger.error(f"ID parsing failed: {e}")

        # 7b. Cross-field validation
        validator = FieldValidator()
        validation_errors = validator.validate(parsed_info, extracted)
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")

        # 8. Calculate overall confidence
        if confidence_scores:
            avg_conf = float(np.mean(list(confidence_scores.values())))
        else:
            avg_conf = 0.0

        # Adjust confidence based on NID checksum validation
        if parsed_info and parsed_info.valid:
            if "id_number" in confidence_scores:
                if parsed_info.checksum_valid:
                    # Boost confidence if checksum is valid
                    confidence_scores["id_number"] = min(
                        1.0, confidence_scores.get("id_number", 0) + 0.15
                    )
                    logger.debug(f"NID checksum valid - boosted confidence")
                else:
                    # Reduce confidence if checksum invalid
                    confidence_scores["id_number"] = max(
                        0.0, confidence_scores.get("id_number", 0) - 0.20
                    )
                    logger.warning(f"NID checksum invalid - reduced confidence")
            
            # Recalculate overall confidence
            avg_conf = float(np.mean(list(confidence_scores.values())))

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
                "checksum_valid": parsed_info.checksum_valid,  # Include checksum status
            }

        logger.info(f"Extraction complete in {processing_ms}ms")
        return result


def get_pipeline() -> IDExtractionPipeline:
    """Get or create the pipeline instance."""
    return IDExtractionPipeline()
