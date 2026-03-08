"""
Egyptian ID Extraction Pipeline Service
Orchestrates the complete OCR pipeline from image input to structured output.

Supports:
- Single image (front OR back)
- Dual-side single image (both sides in one image)
- Multi-image upload (2 separate images: front + back)
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.models.id_parser import parse_national_id
from app.models.nid_extractor import get_nid_extractor, NIDExtractionResult
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
from app.utils.text_utils import clean_field, _is_valid_nid_format, _normalize_digits
from app.core.config import settings
from app.core.logger import logger

# Import new services
from app.services.side_classifier import (
    get_side_classifier,
    SideClassification,
    CardSide,
    SideClassifier
)
from app.services.dual_side_processor import (
    get_dual_side_processor,
    DualSideProcessor,
    DualSideResult
)
from app.services.field_router import (
    get_field_router,
    FieldRouter,
    validate_field,
    is_field_expected_for_side,
    get_fields_for_side,
)


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
    
    # New fields for multi-side processing
    side_info: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class IDExtractionPipeline:
    """
    Complete pipeline for Egyptian ID card extraction.
    Uses singleton pattern to ensure models are loaded once.
    
    Supports:
    - Single image processing (front or back)
    - Dual-side image processing (both sides in one image)
    - Multi-image processing (separate front and back images)
    """

    _instance = None
    _detector: Optional[EgyptianIDDetector] = None
    _ocr: Optional[OCREngine] = None
    _side_classifier: Optional[SideClassifier] = None
    _dual_processor: Optional[DualSideProcessor] = None

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

        if IDExtractionPipeline._side_classifier is None:
            IDExtractionPipeline._side_classifier = get_side_classifier()

        if IDExtractionPipeline._dual_processor is None:
            IDExtractionPipeline._dual_processor = get_dual_side_processor(
                detector=IDExtractionPipeline._detector,
                ocr_engine=IDExtractionPipeline._ocr
            )

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

        if cls._side_classifier is None:
            cls._side_classifier = get_side_classifier()

        if cls._dual_processor is None:
            cls._dual_processor = get_dual_side_processor(
                detector=cls._detector,
                ocr_engine=cls._ocr
            )

        logger.info("Pipeline initialized")

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Process a single image and extract ID information.
        Automatically detects if image is front, back, or both sides.

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

        # 2. Classify the image side
        classification = self._classify_image(image)
        logger.info(f"Image classified as: {classification.side.value} (confidence: {classification.confidence:.2f})")

        # 3. Route to appropriate processor
        if classification.side == CardSide.BOTH:
            # Dual-side image - use dual-side processor
            result = self._process_dual_side(image, classification)
        else:
            # Single side - use standard processing
            side = "front" if classification.side == CardSide.FRONT else "back"
            result = self._process_single_side(image, side, classification)

        # Add side info to result
        result["side_info"] = {
            "detected_side": classification.side.value,
            "classification_confidence": round(classification.confidence, 3),
            "classification_details": classification.details
        }

        processing_ms = int((time.time() - start_time) * 1000)
        result["processing_ms"] = processing_ms
        
        logger.info(f"Extraction complete in {processing_ms}ms")
        return result

    def process_multi_image(
        self, 
        front_bytes: Optional[bytes] = None,
        back_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Process multiple images (front and/or back) and merge results.
        
        Args:
            front_bytes: Front side image bytes (optional)
            back_bytes: Back side image bytes (optional)
            
        Returns:
            Dictionary with merged extracted data, confidence scores, and processing time
        """
        start_time = time.time()
        result = {
            "extracted": {},
            "confidence": {},
            "warnings": [],
            "side_info": {"source": "multi_image"}
        }
        
        front_data = None
        back_data = None
        
        # Process front image if provided
        if front_bytes:
            try:
                front_image = decode_image(front_bytes)
                classification = self._classify_image(front_image)
                
                # Force treat as front even if classification differs
                front_data = self._process_single_side(front_image, "front", classification)
                logger.info(f"Front image processed in {front_data.get('processing_ms', 0)}ms")
            except Exception as e:
                logger.error(f"Failed to process front image: {e}")
                result["warnings"].append(f"Front image processing failed: {str(e)}")
        
        # Process back image if provided
        if back_bytes:
            try:
                back_image = decode_image(back_bytes)
                classification = self._classify_image(back_image)
                
                # Force treat as back even if classification differs
                back_data = self._process_single_side(back_image, "back", classification)
                logger.info(f"Back image processed in {back_data.get('processing_ms', 0)}ms")
            except Exception as e:
                logger.error(f"Failed to process back image: {e}")
                result["warnings"].append(f"Back image processing failed: {str(e)}")
        
        # Merge results
        if front_data and back_data:
            result = self._merge_multi_image_results(front_data, back_data, result)
        elif front_data:
            result = front_data
            result["side_info"]["source"] = "front_only"
        elif back_data:
            result = back_data
            result["side_info"]["source"] = "back_only"
        else:
            result["error"] = "No valid images provided"
            result["processing_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        result["processing_ms"] = int((time.time() - start_time) * 1000)
        logger.info(f"Multi-image processing complete in {result['processing_ms']}ms")
        return result

    def _classify_image(self, image: np.ndarray) -> SideClassification:
        """Classify image as front, back, or both sides."""
        classifier = get_side_classifier()
        return classifier.classify(image)

    def _process_dual_side(self, image: np.ndarray, 
                           classification: SideClassification) -> Dict[str, Any]:
        """Process a dual-side image using the dual-side processor."""
        processor = get_dual_side_processor(
            detector=self._detector,
            ocr_engine=self._ocr
        )
        
        dual_result: DualSideResult = processor.process(image)
        
        # Convert to standard result format
        result = {
            "extracted": dual_result.extracted,
            "confidence": dual_result.confidence,
            "cross_validation": dual_result.cross_validation,
            "warnings": dual_result.warnings,
        }
        
        # Add parsed_id if available
        if dual_result.parsed_id:
            result["parsed_id"] = dual_result.parsed_id
        
        return result

    def _process_single_side(self, image: np.ndarray, side: str,
                             classification: SideClassification) -> Dict[str, Any]:
        """
        Process a single side image with side-aware field filtering.

        Args:
            image: Input image
            side: "front" or "back"
            classification: Side classification result

        Returns:
            Dictionary with extracted data and metadata
        """
        start_time = time.time()
        warnings = []

        # 1. Preprocess (deskew) - temporarily disabled, causes issues
        # try:
        #     image = deskew(image)
        # except Exception as e:
        #     logger.warning(f"Deskew failed: {e}")

        # 2. Detect card (if detector available)
        card_image = image
        if self._detector:
            try:
                card_image = self._detector.crop_card(image)
            except Exception as e:
                logger.warning(f"Card detection failed: {e}")

        # 3. Detect fields with side-aware filtering (if detector available)
        yolo_fields = {}
        field_metadata = {"dual_side_indicator": False, "side_classification": {}}
        
        if self._detector:
            try:
                # Use side-aware field detection
                yolo_fields = self._detector.crop_fields(
                    card_image,
                    expected_side=side,
                    strict_side_filter=False  # Don't strictly filter, but mark unexpected
                )
                
                # Get field metadata for dual-side detection
                if self._detector.field_detector.session is not None:
                    detections, metadata = self._detector.detect_fields_with_metadata(
                        card_image, expected_side=side
                    )
                    field_metadata = metadata
                    
                    # Check if ONNX detected fields from both sides (indicates potential dual-side)
                    if metadata.get("is_dual_side_indicator"):
                        warnings.append(
                            f"Detected fields from both sides in single-side image. "
                            f"Front: {metadata['side_classification'].get('front', [])}, "
                            f"Back: {metadata['side_classification'].get('back', [])}. "
                            f"Consider using dual-side processing."
                        )
                        logger.warning(
                            f"Potential dual-side image detected during single-side processing. "
                            f"Expected: {side}, but found fields from both sides."
                        )
                        
            except Exception as e:
                logger.warning(f"Field detection failed: {e}")

        # 4. Extract ROIs (fallback to template-based layout on normalized card)
        if not yolo_fields:
            try:
                # Layout-aware preprocessing for fallback path
                normalized = resize_to_standard(card_image, target_width=settings.TARGET_IMAGE_WIDTH)
                normalized = remove_noise(normalized)
                normalized = enhance_contrast(normalized)
                normalized = auto_detect_and_warp(normalized)
                normalized = remove_background_lines(normalized)

                # Use provided side or detect
                template_side = side
                if template_side == "back" or classification.side == CardSide.BACK:
                    template_side = "back"
                else:
                    template_side = "front"

                logger.info(f"Template ROI fallback using side='{template_side}'")
                template_rois = extract_all_rois(normalized, side=template_side)

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
                "warnings": warnings,
            }

        # 5. OCR for each field with side-aware validation
        extracted: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}
        field_validations: Dict[str, Dict[str, Any]] = {}

        # Fields we actually OCR to reduce runtime
        ocr_fields = {
            "nid", "id_number", "front_nid", "back_nid",
            "firstName", "lastName", "address", "add_line_1", "add_line_2",
            "serial", "serial_num", "issue_date", "expiry_date",
            "dob", "gender", "job_title",
        }

        # Get expected field IDs for this side (for validation)
        expected_field_ids = get_fields_for_side(side)

        for field_name, (field_img, det_conf) in yolo_fields.items():
            # Skip non-essential ROIs
            if field_name not in ocr_fields:
                continue
            
            try:
                # Preprocess for OCR
                processed = preprocess_text_field(field_img, field_type=field_name)

                # Run OCR
                if self._ocr:
                    ocr_result = self._ocr.ocr_field(processed, field_name)
                    text = ocr_result.text
                    ocr_conf = ocr_result.confidence
                else:
                    text = ""
                    ocr_conf = 0.0

                # Clean text
                cleaned = clean_field(text, field_name)
                extracted[field_name] = cleaned

                # Calculate base confidence
                conf = (det_conf + ocr_conf) / 2 if det_conf > 0 else ocr_conf

                # Boost confidence for expected fields on this side
                field_router = get_field_router()
                onnx_id = field_router.get_onnx_class_id(field_name)
                
                if onnx_id is not None:
                    # Check if field is expected for this side
                    if is_field_expected_for_side(onnx_id, side):
                        conf = min(1.0, conf * 1.15)  # 15% boost for expected fields
                        logger.debug(
                            f"Boosted confidence for expected field {field_name}: {conf:.2f}"
                        )
                    
                    # Validate field value
                    validation = validate_field(onnx_id, cleaned)
                    if validation.is_valid:
                        if validation.warnings:
                            logger.debug(
                                f"Field {field_name} validation warnings: {validation.warnings}"
                            )
                    else:
                        warnings.append(
                            f"Field '{field_name}' validation failed: {validation.error_message}"
                        )
                        # Reduce confidence for invalid fields
                        conf *= 0.7

                # Boost NID confidence if format is valid
                if field_name in ["nid", "front_nid", "back_nid", "id_number"] and cleaned:
                    if _is_valid_nid_format(cleaned):
                        conf = max(conf, 0.85)
                        logger.debug(f"NID format validation passed - boosted confidence to {conf:.2f}")
                    elif len(cleaned) >= 10:
                        conf = max(conf, 0.5)

                # Debug: Save NID field image if extraction failed or low confidence
                if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                    if not cleaned or len(cleaned) < 14 or conf < 0.7:
                        try:
                            debug_dir = Path("debug/nid")
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            timestamp = int(time.time() * 1000)
                            field_path = debug_dir / f"nid_field_{timestamp}_orig.jpg"
                            cv2.imwrite(str(field_path), field_img)
                            processed_path = debug_dir / f"nid_field_{timestamp}_proc.jpg"
                            cv2.imwrite(str(processed_path), processed)
                            logger.warning(
                                f"NID debug images saved (text='{cleaned}', "
                                f"len={len(cleaned)}, conf={conf:.2f})"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to save NID debug image: {e}")

                confidence_scores[field_name] = round(conf, 3)

            except Exception as e:
                logger.error(f"OCR failed for field {field_name}: {e}")
                extracted[field_name] = ""
                confidence_scores[field_name] = 0.0

        # 6. Parse national ID
        parsed_info = None
        nid = (
            extracted.get("nid", "")
            or extracted.get("front_nid", "")
            or extracted.get("back_nid", "")
            or extracted.get("id_number", "")
        )
        if nid:
            try:
                parsed_info = parse_national_id(nid)
                if not parsed_info.valid:
                    warnings.append(f"Invalid ID parsed: {parsed_info.error}")
                    logger.warning(f"Invalid ID parsed: {parsed_info.error}")
            except Exception as e:
                logger.error(f"ID parsing failed: {e}")
                warnings.append(f"ID parsing error: {str(e)}")

        # 7. Calculate overall confidence
        if confidence_scores:
            avg_conf = float(np.mean(list(confidence_scores.values())))
        else:
            avg_conf = 0.0

        level = "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"

        # 8. Build response with enhanced side info
        result = {
            "extracted": extracted,
            "confidence": {
                "overall": round(avg_conf, 3),
                "level": level,
                "per_field": confidence_scores,
            },
            "side_info": {
                "processed_side": side,
                "classification": classification.side.value,
                "field_metadata": field_metadata,
            },
            "warnings": warnings,
        }

        if parsed_info and parsed_info.valid:
            result["parsed_id"] = {
                "birth_date": parsed_info.birth_date,
                "governorate": parsed_info.governorate,
                "gender": parsed_info.gender,
                "age": parsed_info.age,
                "sequence": parsed_info.sequence,
            }

        return result

    def _merge_multi_image_results(
        self, 
        front_data: Dict[str, Any], 
        back_data: Dict[str, Any],
        base_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge results from front and back images.
        
        Priority:
        - Front: firstName, lastName, nid, address, serial
        - Back: add_line_1, add_line_2, issue_date, expiry_date
        - NID: cross-validate between both sides
        
        Args:
            front_data: Result from front image processing
            back_data: Result from back image processing
            base_result: Base result dictionary to update
            
        Returns:
            Merged result dictionary
        """
        front_extracted = front_data.get("extracted", {})
        back_extracted = back_data.get("extracted", {})
        front_conf = front_data.get("confidence", {}).get("per_field", {})
        back_conf = back_data.get("confidence", {}).get("per_field", {})
        
        merged_extracted = {}
        merged_confidence = {}
        all_confidences = []
        cross_validation = {}
        
        # Front-priority fields
        front_priority = {"firstName", "lastName", "address", "serial"}
        for field in front_priority:
            if field in front_extracted and front_extracted[field]:
                merged_extracted[field] = front_extracted[field]
                if field in front_conf:
                    merged_confidence[field] = front_conf[field]
                    all_confidences.append(front_conf[field])
        
        # Back-priority fields
        back_priority = {"add_line_1", "add_line_2", "issue_date", "expiry_date"}
        for field in back_priority:
            if field in back_extracted and back_extracted[field]:
                merged_extracted[field] = back_extracted[field]
                if field in back_conf:
                    merged_confidence[field] = back_conf[field]
                    all_confidences.append(back_conf[field])
        
        # NID cross-validation
        front_nid = front_extracted.get("front_nid", "") or front_extracted.get("nid", "")
        back_nid = back_extracted.get("back_nid", "") or back_extracted.get("nid", "")
        
        front_nid_conf = front_conf.get("front_nid", front_conf.get("nid", 0.0))
        back_nid_conf = back_conf.get("back_nid", back_conf.get("nid", 0.0))
        
        nid_result = self._cross_validate_nid(front_nid, front_nid_conf, back_nid, back_nid_conf)
        
        if nid_result["nid"]:
            merged_extracted["nid"] = nid_result["nid"]
            merged_confidence["nid"] = nid_result["confidence"]
            all_confidences.append(nid_result["confidence"])
        
        cross_validation["nid"] = nid_result
        
        # Merge address if both sides have address info
        if front_extracted.get("address") and back_extracted.get("add_line_1"):
            # Combine addresses
            address_parts = []
            if back_extracted.get("add_line_1"):
                address_parts.append(back_extracted["add_line_1"])
            if back_extracted.get("add_line_2"):
                address_parts.append(back_extracted["add_line_2"])
            if front_extracted.get("address") and front_extracted["address"] not in ' '.join(address_parts):
                address_parts.append(front_extracted["address"])
            
            merged_extracted["address"] = ' | '.join(address_parts)
            merged_confidence["address"] = max(
                front_conf.get("address", 0.5),
                back_conf.get("add_line_1", 0.5)
            )
        
        # Calculate overall confidence
        if all_confidences:
            avg_conf = float(np.mean(all_confidences))
            level = "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"
        else:
            avg_conf = 0.0
            level = "unknown"
        
        # Parse NID
        parsed_info = None
        nid = merged_extracted.get("nid", "")
        if nid:
            try:
                parsed_info = parse_national_id(nid)
            except Exception as e:
                logger.warning(f"NID parsing failed: {e}")
        
        base_result["extracted"] = merged_extracted
        base_result["confidence"] = {
            "overall": round(avg_conf, 3),
            "level": level,
            "per_field": merged_confidence,
        }
        base_result["cross_validation"] = cross_validation
        base_result["side_info"]["source"] = "multi_image_merged"
        
        if parsed_info and parsed_info.valid:
            base_result["parsed_id"] = {
                "birth_date": parsed_info.birth_date,
                "governorate": parsed_info.governorate,
                "gender": parsed_info.gender,
                "age": parsed_info.age,
                "sequence": parsed_info.sequence,
            }
        
        return base_result

    def _cross_validate_nid(
        self,
        front_nid: str,
        front_conf: float,
        back_nid: str,
        back_conf: float
    ) -> Dict[str, Any]:
        """Cross-validate NID from front and back sources."""
        result = {
            "nid": "",
            "confidence": 0.0,
            "source": "none",
            "front_nid": front_nid,
            "back_nid": back_nid,
            "match_status": "none"
        }
        
        # Normalize NIDs
        front_nid = _normalize_digits(front_nid) if front_nid else ""
        back_nid = _normalize_digits(back_nid) if back_nid else ""
        
        # Case 1: Only front NID
        if front_nid and not back_nid:
            conf = max(front_conf, 0.85) if _is_valid_nid_format(front_nid) else front_conf
            result["nid"] = front_nid
            result["confidence"] = conf
            result["source"] = "front"
            result["match_status"] = "front_only"
            return result
        
        # Case 2: Only back NID
        if back_nid and not front_nid:
            conf = max(back_conf, 0.85) if _is_valid_nid_format(back_nid) else back_conf
            result["nid"] = back_nid
            result["confidence"] = conf
            result["source"] = "back"
            result["match_status"] = "back_only"
            return result
        
        # Case 3: Both NIDs
        if front_nid and back_nid:
            if front_nid == back_nid:
                # Match - boost confidence
                avg_conf = (front_conf + back_conf) / 2
                conf = max(avg_conf, front_conf, back_conf)
                if _is_valid_nid_format(front_nid):
                    conf = max(conf, 0.95)
                
                result["nid"] = front_nid
                result["confidence"] = conf
                result["source"] = "both_matched"
                result["match_status"] = "match"
                logger.info(f"NID match: {front_nid} (conf={conf:.2f})")
                return result
            else:
                # Mismatch - use front with reduced confidence
                conf = front_conf * 0.8
                if _is_valid_nid_format(front_nid):
                    conf = max(conf, 0.7)
                
                result["nid"] = front_nid
                result["confidence"] = conf
                result["source"] = "front_mismatch"
                result["match_status"] = "mismatch"
                logger.warning(f"NID mismatch: front={front_nid}, back={back_nid}")
                return result
        
        return result


def get_pipeline() -> IDExtractionPipeline:
    """Get or create the pipeline instance."""
    return IDExtractionPipeline()
