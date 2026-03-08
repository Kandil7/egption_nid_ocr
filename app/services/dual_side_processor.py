"""
Dual-Side Processor for Egyptian ID Cards
Handles images containing both front and back sides of the ID card.

Features:
- Split dual-side images into front and back halves
- Process each side separately with optimized pipelines
- Merge results with intelligent field prioritization
- Cross-validate NID between sides for accuracy
- Side-aware field routing for improved accuracy
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

from app.core.logger import logger
from app.core.config import settings
from app.services.side_classifier import SideClassifier, CardSide, SideClassification
from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.models.id_parser import parse_national_id
from app.services.field_router import (
    get_field_router,
    FieldRouter,
    validate_field,
    is_field_expected_for_side,
    get_fields_for_side,
)
from app.utils.image_utils import (
    resize_to_standard,
    remove_noise,
    enhance_contrast,
    auto_detect_and_warp,
    remove_background_lines,
    detect_card_side,
    extract_all_rois,
    preprocess_text_field,
)
from app.utils.text_utils import clean_field, _is_valid_nid_format, _normalize_digits


@dataclass
class DualSideResult:
    """Result from processing dual-side ID card image."""
    
    # Combined extracted data
    extracted: Dict[str, str] = field(default_factory=dict)
    
    # Confidence scores
    confidence: Dict[str, Any] = field(default_factory=dict)
    
    # Processing time in milliseconds
    processing_ms: int = 0
    
    # Side-specific results
    front_result: Optional[Dict[str, Any]] = None
    back_result: Optional[Dict[str, Any]] = None
    
    # Split information
    split_info: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation results
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Parsed ID information
    parsed_id: Optional[Dict[str, Any]] = None
    
    # Any warnings or issues
    warnings: List[str] = field(default_factory=list)


class DualSideProcessor:
    """
    Processor for dual-side Egyptian ID card images.
    
    Pipeline:
    1. Detect and split the image into front and back halves
    2. Process each half through the standard extraction pipeline
    3. Merge results with field prioritization
    4. Cross-validate NID between sides
    5. Return combined result with metadata
    """
    
    # Field priority mapping
    # Front side fields (higher priority for these fields)
    FRONT_PRIORITY_FIELDS = {
        "firstName", "lastName", "nid", "front_nid", "address", "serial"
    }
    
    # Back side fields (higher priority for these fields)
    BACK_PRIORITY_FIELDS = {
        "add_line_1", "add_line_2", "back_nid", "issue_date", "expiry_date"
    }
    
    # Fields that appear on both sides (need cross-validation)
    OVERLAPPING_FIELDS = {"nid", "front_nid", "back_nid", "id_number", "address"}
    
    def __init__(self, detector: Optional[EgyptianIDDetector] = None,
                 ocr_engine: Optional[OCREngine] = None):
        """
        Initialize the dual-side processor.
        
        Args:
            detector: YOLO/ONNX detector for field detection
            ocr_engine: OCR engine for text recognition
        """
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.side_classifier = SideClassifier()
        
        # OCR fields to extract (reduced set for speed)
        self.ocr_fields = {
            "nid", "id_number", "front_nid", "back_nid",
            "firstName", "lastName", "address", "add_line_1", "add_line_2",
            "serial", "serial_num", "issue_date", "expiry_date"
        }
    
    def process(self, image: np.ndarray) -> DualSideResult:
        """
        Process a dual-side ID card image.
        
        Args:
            image: Input image containing both front and back sides
            
        Returns:
            DualSideResult with merged extraction results
        """
        start_time = time.time()
        result = DualSideResult()
        
        try:
            # Step 1: Classify and determine split orientation
            classification = self.side_classifier.classify(image)
            
            if classification.side != CardSide.BOTH:
                result.warnings.append(
                    f"Image classified as {classification.side.value}, not both sides. "
                    f"Confidence: {classification.confidence:.2f}"
                )
                # Still try to process as dual-side if aspect ratio suggests it
                if classification.confidence < 0.7:
                    logger.warning(
                        f"Proceeding with dual-side processing despite classification: "
                        f"{classification.side.value}"
                    )
                else:
                    # Not a dual-side image, return early
                    result.warnings.append("Not a dual-side image, use standard pipeline")
                    result.processing_ms = int((time.time() - start_time) * 1000)
                    return result
            
            result.split_info = classification.details
            
            # Step 2: Split the image
            orientation = classification.details.get("split_orientation", None)
            front_image, back_image = self.side_classifier.split_dual_side_image(
                image, orientation
            )
            
            result.split_info["orientation"] = orientation
            result.split_info["front_size"] = {"width": front_image.shape[1], "height": front_image.shape[0]}
            result.split_info["back_size"] = {"width": back_image.shape[1], "height": back_image.shape[0]}
            
            logger.info(
                f"Split dual-side image: front={front_image.shape}, back={back_image.shape}"
            )
            
            # Step 3: Process each side
            front_result = self._process_single_side(front_image, "front")
            back_result = self._process_single_side(back_image, "back")
            
            result.front_result = front_result
            result.back_result = back_result
            
            # Step 4: Merge results
            merged, cross_val = self._merge_results(front_result, back_result)
            
            result.extracted = merged["extracted"]
            result.confidence = merged["confidence"]
            result.cross_validation = cross_val
            
            # Step 5: Parse NID
            nid = merged["extracted"].get("nid", "") or merged["extracted"].get("id_number", "")
            if nid:
                try:
                    parsed = parse_national_id(nid)
                    if parsed.valid:
                        result.parsed_id = {
                            "birth_date": parsed.birth_date,
                            "governorate": parsed.governorate,
                            "gender": parsed.gender,
                            "age": parsed.age,
                            "sequence": parsed.sequence,
                        }
                except Exception as e:
                    logger.warning(f"NID parsing failed: {e}")
                    result.warnings.append(f"NID parsing failed: {str(e)}")
            
            # Step 6: Calculate processing time
            result.processing_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Dual-side processing complete in {result.processing_ms}ms"
            )
            
        except Exception as e:
            logger.error(f"Dual-side processing failed: {e}")
            result.warnings.append(f"Processing error: {str(e)}")
            result.processing_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    def _process_single_side(self, image: np.ndarray, side: str) -> Dict[str, Any]:
        """
        Process a single side of the ID card.
        
        Args:
            image: Image of one side
            side: "front" or "back"
            
        Returns:
            Dictionary with extracted data and metadata
        """
        start_time = time.time()
        
        # Preprocess the image
        try:
            normalized = self._preprocess_card(image)
        except Exception as e:
            logger.warning(f"Preprocessing failed for {side}: {e}")
            normalized = image
        
        # Detect fields using YOLO/ONNX
        fields = {}
        if self.detector:
            try:
                fields = self.detector.crop_fields(normalized)
            except Exception as e:
                logger.warning(f"Field detection failed for {side}: {e}")
        
        # Fallback to template-based ROI extraction
        if not fields:
            try:
                template_rois = extract_all_rois(normalized, side=side)
                fields = {k: (v, 0.5) for k, v in template_rois.items()}
                logger.debug(f"Using template ROIs for {side} side")
            except Exception as e:
                logger.warning(f"Template ROI extraction failed for {side}: {e}")
        
        # OCR for each field
        extracted = {}
        confidence_scores = {}
        
        for field_name, (field_img, det_conf) in fields.items():
            if field_name not in self.ocr_fields:
                continue
            
            try:
                # Preprocess for OCR
                processed = preprocess_text_field(field_img, field_type=field_name)
                
                # Run OCR
                if self.ocr_engine:
                    ocr_result = self.ocr_engine.ocr_field(processed, field_name)
                    text = ocr_result.text
                    ocr_conf = ocr_result.confidence
                else:
                    text = ""
                    ocr_conf = 0.0
                
                # Clean text
                cleaned = clean_field(text, field_name)
                extracted[field_name] = cleaned
                
                # Calculate confidence
                conf = (det_conf + ocr_conf) / 2 if det_conf > 0 else ocr_conf
                
                # Boost NID confidence if format is valid
                if field_name in ["nid", "front_nid", "back_nid", "id_number"] and cleaned:
                    if _is_valid_nid_format(cleaned):
                        conf = max(conf, 0.85)
                    elif len(cleaned) >= 10:
                        conf = max(conf, 0.5)
                
                confidence_scores[field_name] = round(conf, 3)
                
            except Exception as e:
                logger.error(f"OCR failed for {side}.{field_name}: {e}")
                extracted[field_name] = ""
                confidence_scores[field_name] = 0.0
        
        # Calculate overall confidence
        avg_conf = float(np.mean(list(confidence_scores.values()))) if confidence_scores else 0.0
        level = "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"
        
        return {
            "side": side,
            "extracted": extracted,
            "confidence": {
                "overall": round(avg_conf, 3),
                "level": level,
                "per_field": confidence_scores,
            },
            "processing_ms": int((time.time() - start_time) * 1000),
        }
    
    def _preprocess_card(self, image: np.ndarray) -> np.ndarray:
        """Preprocess card image for field detection and OCR."""
        # Resize to standard width
        normalized = resize_to_standard(image, target_width=settings.TARGET_IMAGE_WIDTH)
        
        # Denoise
        normalized = remove_noise(normalized)
        
        # Enhance contrast
        normalized = enhance_contrast(normalized)
        
        # Perspective correction
        normalized = auto_detect_and_warp(normalized)
        
        # Remove background lines
        normalized = remove_background_lines(normalized)
        
        return normalized
    
    def _merge_results(self, front_result: Dict[str, Any], 
                       back_result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Merge results from front and back sides.
        
        Priority rules:
        - Front side: firstName, lastName, nid, address, serial
        - Back side: add_line_1, add_line_2, nid (back), issue_date, expiry_date
        - Overlapping fields (nid): cross-validate and use higher confidence
        
        Args:
            front_result: Result from front side processing
            back_result: Result from back side processing
            
        Returns:
            Tuple of (merged_result, cross_validation_info)
        """
        merged = {
            "extracted": {},
            "confidence": {
                "overall": 0.0,
                "level": "unknown",
                "per_field": {},
            }
        }
        
        cross_validation = {
            "nid_match": None,
            "nid_front": None,
            "nid_back": None,
            "fields_merged": []
        }
        
        front_data = front_result.get("extracted", {})
        back_data = back_result.get("extracted", {})
        front_conf = front_result.get("confidence", {}).get("per_field", {})
        back_conf = back_result.get("confidence", {}).get("per_field", {})
        
        all_confidences = []
        
        # Process front-priority fields
        for field_name in self.FRONT_PRIORITY_FIELDS:
            if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                continue  # Handle NID separately
            
            front_val = front_data.get(field_name, "")
            front_c = front_conf.get(field_name, 0.0)
            
            if front_val:
                merged["extracted"][field_name] = front_val
                merged["confidence"]["per_field"][field_name] = front_c
                all_confidences.append(front_c)
                cross_validation["fields_merged"].append(f"{field_name}:front")
        
        # Process back-priority fields
        for field_name in self.BACK_PRIORITY_FIELDS:
            if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                continue  # Handle NID separately
            
            back_val = back_data.get(field_name, "")
            back_c = back_conf.get(field_name, 0.0)
            
            if back_val:
                merged["extracted"][field_name] = back_val
                merged["confidence"]["per_field"][field_name] = back_c
                all_confidences.append(back_c)
                cross_validation["fields_merged"].append(f"{field_name}:back")
        
        # Handle NID with cross-validation
        nid_result = self._cross_validate_nid(front_data, back_data, front_conf, back_conf)
        
        if nid_result["nid"]:
            merged["extracted"]["nid"] = nid_result["nid"]
            merged["confidence"]["per_field"]["nid"] = nid_result["confidence"]
            all_confidences.append(nid_result["confidence"])
        
        cross_validation["nid_match"] = nid_result["match"]
        cross_validation["nid_front"] = nid_result["front_nid"]
        cross_validation["nid_back"] = nid_result["back_nid"]
        
        # Handle address merging (combine front address with back address lines)
        address_result = self._merge_address(front_data, back_data)
        if address_result["address"]:
            merged["extracted"]["address"] = address_result["address"]
            merged["confidence"]["per_field"]["address"] = address_result["confidence"]
            all_confidences.append(address_result["confidence"])
        
        # Calculate overall confidence
        if all_confidences:
            avg_conf = float(np.mean(all_confidences))
            merged["confidence"]["overall"] = round(avg_conf, 3)
            merged["confidence"]["level"] = (
                "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"
            )
        
        return merged, cross_validation
    
    def _cross_validate_nid(self, front_data: Dict[str, str], back_data: Dict[str, str],
                            front_conf: Dict[str, float], back_conf: Dict[str, float]) -> Dict[str, Any]:
        """
        Cross-validate NID between front and back sides.
        
        Priority:
        1. front_nid if available and valid
        2. back_nid if front not available
        3. If both available, compare and use higher confidence
        
        Args:
            front_data: Extracted data from front side
            back_data: Extracted data from back side
            front_conf: Confidence scores from front side
            back_conf: Confidence scores from back side
            
        Returns:
            Dictionary with nid, confidence, match status, and source NIDs
        """
        result = {
            "nid": "",
            "confidence": 0.0,
            "match": None,
            "front_nid": None,
            "back_nid": None,
            "source": ""
        }
        
        # Get NID from both sides
        front_nid = front_data.get("front_nid", "") or front_data.get("nid", "")
        back_nid = back_data.get("back_nid", "") or back_data.get("nid", "")
        
        # Normalize NIDs
        front_nid = _normalize_digits(front_nid) if front_nid else ""
        back_nid = _normalize_digits(back_nid) if back_nid else ""
        
        result["front_nid"] = front_nid
        result["back_nid"] = back_nid
        
        front_c = front_conf.get("front_nid", front_conf.get("nid", 0.0))
        back_c = back_conf.get("back_nid", back_conf.get("nid", 0.0))
        
        # Case 1: Only front NID available
        if front_nid and not back_nid:
            result["nid"] = front_nid
            result["confidence"] = front_c
            result["source"] = "front"
            result["match"] = "front_only"
            logger.info(f"NID from front only: {front_nid}")
            return result
        
        # Case 2: Only back NID available
        if back_nid and not front_nid:
            result["nid"] = back_nid
            result["confidence"] = back_c
            result["source"] = "back"
            result["match"] = "back_only"
            logger.info(f"NID from back only: {back_nid}")
            return result
        
        # Case 3: Both NIDs available - compare
        if front_nid and back_nid:
            if front_nid == back_nid:
                # NIDs match - use higher confidence or average
                result["nid"] = front_nid
                result["confidence"] = max(front_c, back_c, (front_c + back_c) / 2)
                result["match"] = "match"
                result["source"] = "both_matched"
                
                # Boost confidence for matching NIDs
                if _is_valid_nid_format(front_nid):
                    result["confidence"] = max(result["confidence"], 0.95)
                
                logger.info(
                    f"NID match confirmed: {front_nid} (front_conf={front_c:.2f}, "
                    f"back_conf={back_c:.2f})"
                )
                return result
            else:
                # NIDs don't match - use front with warning
                result["nid"] = front_nid
                result["confidence"] = front_c * 0.8  # Reduce confidence for mismatch
                result["match"] = "mismatch"
                result["source"] = "front_mismatch"
                
                logger.warning(
                    f"NID mismatch: front={front_nid}, back={back_nid}. Using front."
                )
                return result
        
        # Case 4: No NID available
        result["match"] = "none"
        return result
    
    def _merge_address(self, front_data: Dict[str, str], 
                       back_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Merge address information from both sides.
        
        Front side typically has a summary address.
        Back side has detailed address lines.
        
        Args:
            front_data: Extracted data from front side
            back_data: Extracted data from back side
            
        Returns:
            Dictionary with merged address and confidence
        """
        result = {
            "address": "",
            "confidence": 0.0
        }
        
        front_address = front_data.get("address", "")
        back_line1 = back_data.get("add_line_1", "")
        back_line2 = back_data.get("add_line_2", "")
        
        front_conf = 0.5  # Default confidence
        back_conf = 0.5
        
        # Build complete address
        address_parts = []
        
        # Add back address lines first (more detailed)
        if back_line1:
            address_parts.append(back_line1)
        if back_line2:
            address_parts.append(back_line2)
        
        # Add front address if we have it (might have additional info)
        if front_address:
            # Check if front address is different from back
            if front_address not in ' '.join(address_parts):
                address_parts.append(front_address)
        
        if address_parts:
            result["address"] = ' | '.join(address_parts)
            result["confidence"] = max(front_conf, back_conf)
        
        return result


# Singleton instance
_processor: Optional[DualSideProcessor] = None


def get_dual_side_processor(detector: Optional[EgyptianIDDetector] = None,
                            ocr_engine: Optional[OCREngine] = None) -> DualSideProcessor:
    """Get or create dual-side processor singleton."""
    global _processor
    if _processor is None:
        _processor = DualSideProcessor(detector=detector, ocr_engine=ocr_engine)
    return _processor
