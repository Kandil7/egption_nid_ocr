"""
ONNX/YOLO-based Detector for Egyptian ID Card
Uses Ultralytics YOLO for .pt models and ONNX Runtime for .onnx models.
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time

from app.core.config import settings
from app.core.logger import logger


@dataclass
class Detection:
    """Represents a single detection result."""

    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


class ONNXFieldDetector:
    """
    Optimized ONNX-based field detector for Egyptian ID cards.

    Uses ONNX Runtime for faster CPU inference compared to PyTorch.
    Model: YOLOv8 export format (Ultralytics)
    Output shape: [batch, 21, anchors] where 21 = 4 (bbox) + 17 (classes)

    YOLOv8 format differs from YOLOv5:
    - YOLOv8: [batch, 4+num_classes, anchors] - NO objectness score
    - YOLOv5: [batch, 4+1+num_classes, anchors] - WITH objectness score

    The class scores in YOLOv8 already incorporate objectness, so we filter
    directly on class confidence.

    Model trained on: /content/drive/MyDrive/unified_egypt_id/data.yaml
    Training date: 2026-02-15, Ultralytics v8.4.14

    Class mapping (17 classes from field_detector.onnx metadata):
    """

    # Field class mapping - EXACT names from model's data.yaml
    # These are the ACTUAL class names the model was trained on
    FIELD_CLASSES = {
        0: "first_name",       # First name field (Arabic)
        1: "last_name",        # Last name field (Arabic)
        2: "add_line_1",       # Address line 1
        3: "add_line_2",       # Address line 2
        4: "front_nid",        # Front NID card (full card region)
        5: "back_nid",         # Back NID card (full card region)
        6: "serial_num",       # Serial number
        7: "issue_code",       # Issue code/region
        8: "expiry_date",      # Expiry date
        9: "job_title",        # Job title/profession
        10: "gender",          # Gender field
        11: "religion",        # Religion field
        12: "marital_status",  # Marital status
        13: "face",            # Face photo region
        14: "front_logo",      # Front logo/emblem
        15: "address",         # Full address block
        16: "dob",             # Date of birth
    }

    # Valid fields we want to use for OCR extraction
    # Filter out full card detections (front_nid, back_nid) used for card alignment
    VALID_FIELDS = {
        "first_name", "last_name", "add_line_1", "add_line_2",
        "serial_num", "issue_code", "expiry_date", "job_title",
        "gender", "religion", "marital_status", "face",
        "front_logo", "address", "dob"
    }
    
    def __init__(self, model_path: str = "weights/field_detector.onnx"):
        """Initialize ONNX field detector."""
        self.model_path = model_path
        self.session = None
        self.input_height = 640
        self.input_width = 640
        
        import os
        if not os.path.exists(model_path):
            logger.warning(f"ONNX model not found: {model_path}")
            return
        
        try:
            import onnxruntime as ort
            
            # Configure for optimal CPU performance
            providers = ['CPUExecutionProvider']
            
            # Try to use OpenVINO for Intel CPUs (faster)
            try:
                providers.insert(0, 'OpenVINOExecutionProvider')
            except:
                pass
            
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers,
                providers_options=[{}]
            )
            
            # Get input/output info
            inputs = self.session.get_inputs()
            self.input_name = inputs[0].name
            
            logger.info(f"Loaded ONNX field detector: {model_path}")
            logger.info(f"  Input: {inputs[0].shape}, Output: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Preprocess image for ONNX model inference.
        
        - Resize to model input size (640x640)
        - Convert BGR to RGB
        - Normalize to [0, 1]
        - Transpose to CHW format
        
        Returns:
            Preprocessed image, scale factors (sx, sy), padding (pad_w, pad_h)
        """
        h, w = image.shape[:2]
        
        # Resize with letterboxing (maintain aspect ratio)
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with padding
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        pad_w = (self.input_width - new_w) // 2
        pad_h = (self.input_height - new_h) // 2
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Transpose to CHW format
        chw = np.transpose(rgb, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)
        
        # Calculate scale factors for coordinate transformation
        sx, sy = w / new_w, h / new_h
        
        return batch, (sx, sy), (pad_w, pad_h)
    
    def _postprocess(self, output: np.ndarray, scale: Tuple[float, float],
                     padding: Tuple[int, int], conf_threshold: float = 0.25) -> List[Detection]:
        """
        Post-process ONNX model output.

        Model: YOLOv8 export format
        Output shape: [batch, 21, anchors] where 21 = 4 (bbox) + 17 (classes)

        YOLOv8 format:
        - Indices 0-3: box coordinates [cx, cy, w, h]
        - Indices 4-20: 17 class scores (NO separate objectness)

        In YOLOv8, class scores already incorporate objectness, so we filter
        directly on class confidence.

        Args:
            output: Model output [1, 21, anchors]
            scale: Scale factors (sx, sy)
            padding: Padding (pad_w, pad_h)
            conf_threshold: Confidence threshold (default 0.25 for YOLOv8)

        Returns:
            List of Detection objects
        """
        # Transpose to [anchors, 21]
        output = np.transpose(output[0], (1, 0))  # [anchors, 21]

        # YOLOv8 format: 4 bbox + 17 classes (no objectness)
        boxes = output[:, 0:4]  # [cx, cy, w, h]
        class_scores = output[:, 4:21]  # 17 classes

        # Log detailed output statistics
        max_class_score = float(np.max(class_scores))
        avg_class_score = float(np.mean(class_scores))

        # Count high-confidence anchors
        max_scores_per_anchor = np.max(class_scores, axis=1)
        high_conf_anchors = int(np.sum(max_scores_per_anchor > 0.3))

        logger.info(f"ONNX stats: cls_max={max_class_score:.3f}, cls_avg={avg_class_score:.4f}, high_conf={high_conf_anchors}")

        detections = []
        debug_dets = []

        for i, anchor in enumerate(output):
            # YOLOv8 format: 4 bbox + 17 classes (no objectness)
            # Extract box coordinates [cx, cy, w, h]
            cx, cy, w, h = anchor[0:4]

            # Extract class scores (17 classes at indices 4-20)
            scores = anchor[4:21]

            # Get best class
            class_id = int(np.argmax(scores))
            class_conf = float(scores[class_id])

            # Use class confidence directly (YOLOv8 doesn't have separate objectness)
            conf = class_conf

            # Log promising detections for debugging
            if class_conf > 0.3:
                debug_dets.append((i, class_id, class_conf))

            # Filter on class confidence only (YOLOv8 format)
            if conf < conf_threshold:
                continue

            # Convert to [x1, y1, x2, y2]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Remove padding and scale back to original image
            x1 = (x1 - padding[0]) * scale[0]
            y1 = (y1 - padding[1]) * scale[1]
            x2 = (x2 - padding[0]) * scale[0]
            y2 = (y2 - padding[1]) * scale[1]

            # Clip to image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(10000, int(x2)), min(10000, int(y2))

            class_name = self.FIELD_CLASSES.get(class_id, f"class_{class_id}")

            # Skip invalid_* detections
            if class_name.startswith("invalid_"):
                continue

            detections.append(
                Detection(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(round(conf, 3)),
                )
            )

        # Log debug info
        if debug_dets:
            logger.info(f"ONNX promising ({len(debug_dets)}): {[(self.FIELD_CLASSES.get(d[1], f'c{d[1]}'), round(d[2], 3)) for d in debug_dets[:15]]}")

        logger.info(f"ONNX pre-NMS: {len(detections)} detections")
        
        # Apply Non-Maximum Suppression (NMS)
        if detections:
            detections = self._apply_nms(detections)
        
        logger.info(f"ONNX post-NMS: {len(detections)} detections")
        
        return detections
    
    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Uses per-class NMS to avoid suppressing detections of different classes
        that may overlap (e.g., face photo overlapping with personal info fields).
        """
        if not detections:
            return []

        # Group detections by class for per-class NMS
        by_class: Dict[int, List[Detection]] = {}
        for d in detections:
            if d.class_id not in by_class:
                by_class[d.class_id] = []
            by_class[d.class_id].append(d)

        logger.info(f"NMS input: {len(detections)} detections across {len(by_class)} classes")

        result = []

        for class_id, class_dets in by_class.items():
            if len(class_dets) == 1:
                # Only one detection for this class, keep it
                result.extend(class_dets)
                continue

            # Multiple detections for same class - apply NMS
            boxes = [d.bbox for d in class_dets]
            scores = [d.confidence for d in class_dets]

            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=0.1,  # Minimum score to consider
                nms_threshold=iou_threshold  # IoU threshold for suppression
            )

            # Handle different OpenCV return formats
            if indices is not None and len(indices) > 0:
                if hasattr(indices, 'flatten'):
                    indices = indices.flatten()
                result.extend([class_dets[int(i)] for i in indices])
            else:
                # NMS returned empty - keep all (shouldn't happen)
                logger.warning(f"NMS returned empty for class {class_id} - keeping all {len(class_dets)} detections")
                result.extend(class_dets)

        logger.info(f"NMS output: {len(result)} detections retained")
        return result
    
    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to field names."""
        return self.FIELD_CLASSES
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.35) -> List[Detection]:
        """
        Run field detection on an image.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold (lowered to 0.35 for better recall)
            
        Returns:
            List of Detection objects
        """
        if self.session is None:
            return []
        
        try:
            # Preprocess
            input_tensor, scale, padding = self._preprocess(image)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Postprocess with lower threshold for better recall
            detections = self._postprocess(outputs[0], scale, padding, conf_threshold)
            
            logger.debug(f"ONNX detector found {len(detections)} fields")
            
            return detections
            
        except Exception as e:
            logger.error(f"ONNX detection error: {e}")
            return []


class YOLODetector:
    """
    YOLO detector using Ultralytics for .pt models.
    Supports both PyTorch and ONNX formats.
    """

    def __init__(self, model_path: str):
        """Initialize detector with the given model path."""
        self.model_path = model_path
        self.model = None

        # Check if file exists
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
    Uses two-stage detection: first find the card, then find the fields.
    
    Prioritizes ONNX field detector for better performance and accuracy.
    """

    def __init__(self):
        """Initialize the detector with card and field detection models."""
        logger.info("Initializing Egyptian ID Detector...")

        self.card_detector = YOLODetector(settings.YOLO_CARD_MODEL)
        
        # Try ONNX field detector first (faster, more accurate)
        self.field_detector_onnx = ONNXFieldDetector("weights/field_detector.onnx")
        self.field_detector_yolo = YOLODetector(settings.YOLO_FIELDS_MODEL)
        
        self.nid_detector = YOLODetector(settings.YOLO_NID_MODEL)

        logger.info("Egyptian ID Detector initialized")
    
    def get_field_class_names(self) -> Dict[int, str]:
        """Get field class names from ONNX or YOLO detector."""
        if self.field_detector_onnx.session:
            return self.field_detector_onnx.get_class_names()
        elif self.field_detector_yolo.model:
            return self.field_detector_yolo.model.names
        return {}

    def crop_card(self, image: np.ndarray) -> np.ndarray:
        """Detect and crop the ID card from the full image."""
        detections = self.card_detector.detect(image)

        # Find card detections (typically class 0 = card)
        card_dets = [d for d in detections if d.class_name == "id_card" or d.class_id == 0]

        if not card_dets:
            logger.warning("No card detected, using full image")
            return image

        # Use the detection with highest confidence
        best = max(card_dets, key=lambda d: d.confidence)
        x1, y1, x2, y2 = best.bbox

        # Ensure valid crop
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        logger.info(f"Card detected: {best.class_name} (conf: {best.confidence:.2f})")
        return image[y1:y2, x1:x2]

    def crop_fields(self, card_image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Detect and crop individual fields from the card image.
        
        Uses ONNX field detector if available (faster, more accurate),
        otherwise falls back to YOLO detector.
        """
        # Try ONNX detector first (lower threshold for better recall)
        if self.field_detector_onnx.session is not None:
            try:
                start_time = time.time()
                detections = self.field_detector_onnx.detect(card_image, conf_threshold=0.30)
                onnx_time = (time.time() - start_time) * 1000
                detected_classes = [d.class_name for d in detections]
                logger.info(f"ONNX field detection: {len(detections)} fields in {onnx_time:.1f}ms - {detected_classes}")
                
                if detections:
                    fields = {}
                    for det in detections:
                        # Skip back_nid (full back card) but keep front_nid for NID number OCR
                        if det.class_name == "back_nid":
                            continue

                        x1, y1, x2, y2 = det.bbox

                        # Ensure valid crop
                        h, w = card_image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        crop = card_image[y1:y2, x1:x2]
                        if crop.size > 0:
                            fields[det.class_name] = (crop, det.confidence)
                            logger.debug(f"Field detected: {det.class_name} (conf: {det.confidence:.2f})")

                    if fields:
                        return fields
                        
            except Exception as e:
                logger.warning(f"ONNX field detection failed: {e}, falling back to YOLO")
        
        # Fallback to YOLO detector
        detections = self.field_detector_yolo.detect(card_image)

        fields = {}
        for det in detections:
            if det.class_name in settings.CLASS_NAMES.values():
                x1, y1, x2, y2 = det.bbox

                # Ensure valid crop
                h, w = card_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = card_image[y1:y2, x1:x2]
                if crop.size > 0:
                    fields[det.class_name] = (crop, det.confidence)
                    logger.debug(f"Field detected: {det.class_name} (conf: {det.confidence:.2f})")

        if not fields:
            logger.warning("No fields detected on the card")

        return fields
