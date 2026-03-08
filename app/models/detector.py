"""
ONNX/YOLO-based Detector for Egyptian ID Card
Uses Ultralytics YOLO for .pt models and ONNX Runtime for .onnx models.

Supports side-aware detection for front/back field filtering.
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
import time

from app.core.config import settings
from app.core.logger import logger
from app.services.field_router import (
    FieldRouter,
    FieldSide,
    get_field_router,
    get_fields_for_side,
    is_field_expected_for_side,
)


@dataclass
class Detection:
    """Represents a single detection result."""
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    side: str = "unknown"  # "front", "back", or "both"


class ONNXFieldDetector:
    """
    ONNX-based field detector using the field_detector.onnx model.
    This is the PRIMARY detector for field detection.
    
    Supports side-aware detection for front/back field filtering.
    """

    def __init__(self, model_path: str):
        """Initialize ONNX detector."""
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_size = (640, 640)
        
        # Field router for side classification
        self.field_router: Optional[FieldRouter] = None

        import os
        if not os.path.exists(model_path):
            logger.warning(f"ONNX model not found: {model_path}")
            return

        try:
            import onnxruntime as ort
            # Use CPU execution provider
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.field_router = get_field_router()
            logger.info(f"Loaded ONNX field detector: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[Detection]:
        """
        Run detection on image.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence threshold for detections
            
        Returns:
            List of Detection objects with side information
        """
        if self.session is None:
            return []

        try:
            # Preprocess
            h, w = image.shape[:2]
            resized = cv2.resize(image, self.input_size)
            blob = cv2.dnn.blobFromImage(resized, 1/255.0, self.input_size, swapRB=True, crop=False)

            # Inference
            start = time.time()
            outputs = self.session.run(None, {self.input_name: blob})
            logger.debug(f"ONNX inference time: {(time.time()-start)*1000:.1f}ms")

            # Parse outputs (YOLOv8 format)
            detections = self._parse_yolo_outputs(outputs[0], w, h, conf_threshold)
            return detections

        except Exception as e:
            logger.error(f"ONNX detection error: {e}")
            return []

    def detect_with_side_filtering(
        self,
        image: np.ndarray,
        expected_side: str,
        conf_threshold: float = 0.4,
        strict_filter: bool = False
    ) -> List[Detection]:
        """
        Run detection with side-aware filtering.
        
        Args:
            image: Input image (BGR format)
            expected_side: Expected card side ("front" or "back")
            conf_threshold: Minimum confidence threshold
            strict_filter: If True, only return fields expected for the side
            
        Returns:
            List of Detection objects filtered by side
        """
        detections = self.detect(image, conf_threshold)
        
        if not detections:
            return detections
        
        # Classify detections by side
        front_detections = []
        back_detections = []
        both_detections = []
        
        for det in detections:
            side = self.get_field_side(det.class_id)
            if side == FieldSide.FRONT:
                front_detections.append(det)
            elif side == FieldSide.BACK:
                back_detections.append(det)
            else:
                both_detections.append(det)
        
        logger.debug(
            f"Side classification: front={len(front_detections)}, "
            f"back={len(back_detections)}, both={len(both_detections)}"
        )
        
        # Check for dual-side indicators
        if front_detections and back_detections:
            logger.warning(
                f"Detected fields from both sides in single-side image. "
                f"Front fields: {[d.class_name for d in front_detections]}, "
                f"Back fields: {[d.class_name for d in back_detections]}"
            )
        
        # Filter based on expected side
        if strict_filter:
            if expected_side == "front":
                filtered = front_detections + both_detections
            elif expected_side == "back":
                filtered = back_detections + both_detections
            else:
                filtered = detections
        else:
            # Soft filtering - keep all but mark unexpected ones
            filtered = detections
            for det in filtered:
                side = self.get_field_side(det.class_id)
                if side.value != expected_side and side != FieldSide.BOTH:
                    det.confidence *= 0.7  # Reduce confidence for unexpected fields
        
        return filtered

    def _parse_yolo_outputs(
        self,
        outputs: np.ndarray,
        img_w: int,
        img_h: int,
        conf_threshold: float
    ) -> List[Detection]:
        """Parse YOLO outputs to detections with side information."""
        detections = []

        # Output shape: [batch, 84, 8400] for YOLOv8
        # 84 = 4 (bbox) + 80 (classes) or 4 + num_classes
        outputs = outputs[0]  # Remove batch dimension

        num_classes = outputs.shape[0] - 4

        for i in range(outputs.shape[1]):
            scores = outputs[4:, i]
            conf = np.max(scores)

            if conf > conf_threshold:
                class_id = int(np.argmax(scores))
                bbox_xywh = outputs[:4, i]

                # Convert to xyxy
                x_center, y_center, bw, bh = bbox_xywh
                x1 = int((x_center - bw/2) * img_w / self.input_size[0])
                y1 = int((y_center - bh/2) * img_h / self.input_size[1])
                x2 = int((x_center + bw/2) * img_w / self.input_size[0])
                y2 = int((y_center + bh/2) * img_h / self.input_size[1])

                # Get class name from ONNX classes
                onnx_class_name = settings.ONNX_FIELD_DETECTOR_CLASSES.get(
                    class_id, f"class_{class_id}"
                )

                # Map to internal name
                class_name = settings.FIELD_NAME_MAP.get(onnx_class_name, onnx_class_name)
                
                # Get side classification
                side = self._get_side_for_class_id(class_id)

                detections.append(Detection(
                    bbox=[max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)],
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(conf),
                    side=side
                ))

        return detections

    def _get_side_for_class_id(self, class_id: int) -> str:
        """Get side classification for an ONNX class ID."""
        if self.field_router:
            field_side = self.field_router.get_field_side(class_id)
            return field_side.value
        # Fallback: use hardcoded mapping
        if class_id in {0, 1, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16}:
            return "front"
        elif class_id in {2, 3, 5, 7, 8}:
            return "back"
        return "unknown"

    def classify_fields_by_side(
        self,
        detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """
        Classify detected fields by side.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary with 'front', 'back', 'both' keys containing respective detections
        """
        result = {"front": [], "back": [], "both": []}
        
        for det in detections:
            if self.field_router:
                side = self.field_router.get_field_side(det.class_id)
                if side == FieldSide.FRONT:
                    result["front"].append(det)
                elif side == FieldSide.BACK:
                    result["back"].append(det)
                else:
                    result["both"].append(det)
            else:
                # Fallback
                side = self._get_side_for_class_id(det.class_id)
                result.get(side, result["both"]).append(det)
        
        return result

    def filter_fields_by_side(
        self,
        detections: List[Detection],
        side: str,
        include_both: bool = True
    ) -> List[Detection]:
        """
        Filter detections to only include fields expected for a given side.
        
        Args:
            detections: List of Detection objects
            side: Target side ("front" or "back")
            include_both: Include fields that appear on both sides
            
        Returns:
            Filtered list of detections
        """
        if not self.field_router:
            return detections
        
        expected_ids = self.field_router.get_fields_for_side(side)
        if include_both:
            both_ids = self.field_router.get_fields_for_side("both")
            expected_ids = expected_ids | both_ids
        
        return [det for det in detections if det.class_id in expected_ids]

    def boost_confidence_for_expected_fields(
        self,
        detections: List[Detection],
        expected_side: str,
        boost_factor: float = 1.15
    ) -> List[Detection]:
        """
        Boost confidence for fields expected on the given side.
        
        Args:
            detections: List of Detection objects
            expected_side: Expected card side
            boost_factor: Multiplier for confidence boost
            
        Returns:
            List of detections with adjusted confidence scores
        """
        if not self.field_router:
            return detections
        
        for det in detections:
            if self.field_router.is_field_expected_for_side(det.class_id, expected_side):
                det.confidence = min(1.0, det.confidence * boost_factor)
                logger.debug(
                    f"Boosted confidence for {det.class_name}: "
                    f"{det.confidence/boost_factor:.2f} -> {det.confidence:.2f}"
                )
        
        return detections

    def detect_dual_side_indicators(
        self,
        detections: List[Detection]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if the detected fields suggest a dual-side image.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Tuple of (is_dual_side, details)
        """
        details = {
            "front_fields": [],
            "back_fields": [],
            "front_count": 0,
            "back_count": 0,
            "is_dual_side": False
        }
        
        classified = self.classify_fields_by_side(detections)
        
        details["front_fields"] = [d.class_name for d in classified["front"]]
        details["back_fields"] = [d.class_name for d in classified["back"]]
        details["front_count"] = len(classified["front"])
        details["back_count"] = len(classified["back"])
        
        # Dual-side if we have significant fields from both sides
        # Require at least 2 front fields and 1 back field
        if details["front_count"] >= 2 and details["back_count"] >= 1:
            details["is_dual_side"] = True
            logger.info(
                f"Dual-side detected from fields: "
                f"front={details['front_fields']}, back={details['back_fields']}"
            )
        
        return details["is_dual_side"], details

    def get_field_side(self, class_id: int) -> FieldSide:
        """Get side classification for a field class ID."""
        if self.field_router:
            return self.field_router.get_field_side(class_id)
        # Fallback
        side_str = self._get_side_for_class_id(class_id)
        if side_str == "front":
            return FieldSide.FRONT
        elif side_str == "back":
            return FieldSide.BACK
        return FieldSide.BOTH


class YOLODetector:
    """
    YOLO detector using Ultralytics for .pt models.
    """

    def __init__(self, model_path: str):
        """Initialize detector with the given model path."""
        self.model_path = model_path
        self.model = None

        import os
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on an image."""
        if self.model is None:
            logger.warning("No model loaded, returning empty detections")
            return []

        try:
            # Run inference
            results = self.model(image, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < settings.YOLO_CONF_THRESHOLD:
                        continue

                    class_id = int(box.cls[0])
                    xyxy = box.xyxy[0].tolist()

                    class_name = settings.CLASS_NAMES.get(class_id, f"class_{class_id}")
                    
                    # Map to internal name
                    class_name = settings.FIELD_NAME_MAP.get(class_name, class_name)

                    detections.append(
                        Detection(
                            bbox=[int(x) for x in xyxy],
                            class_id=class_id,
                            class_name=class_name,
                            confidence=conf,
                        )
                    )

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class EgyptianIDDetector:
    """
    High-level detector for Egyptian ID cards.
    Uses ONNX model for field detection (primary) and YOLO for card detection.
    """

    def __init__(self):
        """Initialize the detector with card and field detection models."""
        logger.info("Initializing Egyptian ID Detector...")

        # Card detection (YOLO .pt model)
        self.card_detector = YOLODetector(settings.YOLO_CARD_MODEL)
        
        # Field detection (ONNX model - PRIMARY)
        self.field_detector = ONNXFieldDetector("weights/field_detector.onnx")
        
        # Fallback field detector (YOLO .pt model)
        self.field_detector_fallback = YOLODetector(settings.YOLO_FIELDS_MODEL)

        logger.info("Egyptian ID Detector initialized (ONNX primary, YOLO fallback)")

    def crop_card(self, image: np.ndarray) -> np.ndarray:
        """Detect and crop the ID card from the full image."""
        detections = self.card_detector.detect(image)

        if not detections:
            logger.warning(f"No card detected (confidence threshold: {settings.YOLO_CONF_THRESHOLD}). All detections: {[(d.class_name, d.confidence) for d in detections]}")
            logger.info("Using full image - will try template-based ROI extraction")
            return image

        # The card model detects corners/edges (front-up, front-bottom, etc.)
        # Group all detections to find card boundaries
        valid_classes = ['front-up', 'front-bottom', 'front-left', 'front-right',
                        'back-up', 'back-bottom', 'back-left', 'back-right',
                        'id_card', 'card']
        
        card_dets = [d for d in detections if d.class_name in valid_classes or d.class_id in [0, 1, 2, 3, 4, 5, 6, 7]]

        if not card_dets:
            logger.warning(f"No valid card detections found. All: {[(d.class_name, d.confidence) for d in detections]}")
            logger.info("Using full image - will try template-based ROI extraction")
            return image

        # Calculate bounding box from all corner/edge detections
        x1 = min(d.bbox[0] for d in card_dets)
        y1 = min(d.bbox[1] for d in card_dets)
        x2 = max(d.bbox[2] for d in card_dets)
        y2 = max(d.bbox[3] for d in card_dets)

        # Ensure valid crop
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        avg_conf = sum(d.confidence for d in card_dets) / len(card_dets)
        logger.info(f"Card detected from {len(card_dets)} parts at [{x1},{y1},{x2},{y2}] (avg conf: {avg_conf:.2f})")
        return image[y1:y2, x1:x2]

    def crop_fields(
        self,
        card_image: np.ndarray,
        expected_side: Optional[str] = None,
        strict_side_filter: bool = False
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Detect and crop individual fields from the card image.
        
        Args:
            card_image: Card image to process
            expected_side: Expected card side ("front" or "back"). If None, no filtering.
            strict_side_filter: If True, only return fields expected for the side
            
        Returns:
            Dictionary mapping field names to (cropped_image, confidence) tuples
        """
        # Try ONNX detector first (PRIMARY)
        if expected_side and self.field_detector.session is not None:
            detections = self.field_detector.detect_with_side_filtering(
                card_image,
                expected_side=expected_side,
                conf_threshold=0.4,
                strict_filter=strict_side_filter
            )
        else:
            detections = self.field_detector.detect(card_image, conf_threshold=0.4)

        # Fallback to YOLO if ONNX returns nothing
        if not detections and self.field_detector_fallback.model is not None:
            logger.debug("ONNX detector returned no results, trying YOLO fallback...")
            fallback_detections = self.field_detector_fallback.detect(card_image)
            if not detections:
                detections = fallback_detections

        # Boost confidence for expected fields
        if expected_side and detections:
            detections = self.field_detector.boost_confidence_for_expected_fields(
                detections, expected_side
            )

        fields = {}
        for det in detections:
            # Filter for relevant fields
            relevant_fields = {
                'firstName', 'lastName', 'nid', 'serial', 'address',
                'add_line_1', 'add_line_2', 'front_nid', 'back_nid',
                'serial_num', 'id_number', 'issue_date', 'expiry_date',
                'dob', 'gender', 'job_title'
            }

            if det.class_name not in relevant_fields:
                continue

            x1, y1, x2, y2 = det.bbox

            # Ensure valid crop
            h, w = card_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = card_image[y1:y2, x1:x2]
            if crop.size > 0:
                fields[det.class_name] = (crop, det.confidence)
                logger.debug(
                    f"Field detected: {det.class_name} at [{x1},{y1},{x2},{y2}] "
                    f"(conf: {det.confidence:.2f}, side: {det.side})"
                )

        if not fields:
            logger.warning("No fields detected on the card")

        return fields

    def detect_fields_with_metadata(
        self,
        card_image: np.ndarray,
        expected_side: Optional[str] = None
    ) -> Tuple[List[Detection], Dict[str, Any]]:
        """
        Detect fields and return with comprehensive metadata.
        
        Args:
            card_image: Card image to process
            expected_side: Expected card side for filtering
            
        Returns:
            Tuple of (detections, metadata)
        """
        detections = self.field_detector.detect(card_image, conf_threshold=0.4)
        
        metadata = {
            "total_detections": len(detections),
            "front_count": 0,
            "back_count": 0,
            "is_dual_side_indicator": False,
            "side_classification": {}
        }
        
        if detections:
            # Classify by side
            classified = self.field_detector.classify_fields_by_side(detections)
            metadata["front_count"] = len(classified["front"])
            metadata["back_count"] = len(classified["back"])
            metadata["side_classification"] = {
                "front": [d.class_name for d in classified["front"]],
                "back": [d.class_name for d in classified["back"]],
                "both": [d.class_name for d in classified["both"]]
            }
            
            # Check for dual-side indicators
            is_dual, dual_details = self.field_detector.detect_dual_side_indicators(detections)
            metadata["is_dual_side_indicator"] = is_dual
            
            # Apply side filtering if expected side is provided
            if expected_side:
                detections = self.field_detector.filter_fields_by_side(
                    detections, expected_side
                )
                detections = self.field_detector.boost_confidence_for_expected_fields(
                    detections, expected_side
                )
        
        return detections, metadata
