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
    Model output: [batch, 21, anchors] where 21 = 4 (bbox) + 17 (classes)
    
    This is the YOLOv8 format WITHOUT separate objectness score.
    Class scores are direct sigmoid outputs - use them directly as confidence.

    Advantages:
    - Faster inference on CPU (no GPU required)
    - Consistent performance across platforms
    - Lower memory footprint
    - Better for production deployment

    Class mapping based on Ultralytics field_detector model metadata:
    Names extracted from model.metadata_props['names']
    """

    # Field class mapping based on actual ONNX model metadata (17 classes, indices 0-16)
    # These are the exact class names from the model's training configuration
    # Note: Model output has 17 class scores matching the 17 metadata classes
    FIELD_CLASSES = {
        0: "first_name",       # first name field
        1: "last_name",        # last name field
        2: "add_line_1",       # address line 1
        3: "add_line_2",       # address line 2
        4: "front_nid",        # front NID number
        5: "back_nid",         # back NID number
        6: "serial_num",       # serial number
        7: "issue_code",       # issue code
        8: "expiry_date",      # expiry date
        9: "job_title",        # job title
        10: "gender",          # gender field
        11: "religion",        # religion field
        12: "marital_status",  # marital status field
        13: "face",            # face photo region
        14: "front_logo",      # front logo
        15: "address",         # general address field
        16: "dob",             # date of birth
    }
    
    def __init__(self, model_path: str = "weights/field_detector.onnx"):
        """Initialize ONNX field detector."""
        self.model_path = model_path
        self.session = None
        self.input_height = 640
        self.input_width = 640
        self.num_classes = 17  # Default: YOLOv8 format (4 box + 17 class = 21)

        import os
        if not os.path.exists(model_path):
            logger.warning(f"ONNX model not found: {model_path}")
            return

        try:
            import onnxruntime as ort
            import onnx

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
            
            # Determine num_classes from actual output shape
            # YOLOv8 format: [batch, 4+num_classes, anchors] - NO objectness
            outputs = self.session.get_outputs()
            output_shape = outputs[0].shape
            self.num_classes = output_shape[1] - 4  # 21 - 4 = 17 classes

            # Try to load class names from model metadata
            try:
                onnx_model = onnx.load(model_path)
                for meta in onnx_model.metadata_props:
                    if meta.key == 'names':
                        # Parse the names dictionary from metadata
                        # Format: {0: 'first_name', 1: 'last_name', ...}
                        import ast
                        metadata_classes = ast.literal_eval(meta.value)
                        
                        # Only use classes that are actually in the output
                        # (metadata may have more classes than the output tensor)
                        self.FIELD_CLASSES = {
                            k: v for k, v in metadata_classes.items() 
                            if k < self.num_classes
                        }
                        logger.info(f"Loaded {len(self.FIELD_CLASSES)} class names from ONNX metadata (output has {self.num_classes} classes)")
                        break
            except Exception as e:
                logger.warning(f"Could not load class names from metadata: {e}, using defaults")

            logger.info(f"Loaded ONNX field detector: {model_path}")
            logger.info(f"  Input: {inputs[0].shape}, Output: {outputs[0].shape}, Classes: {self.num_classes}")

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
                     padding: Tuple[int, int], conf_threshold: float = 0.15) -> List[Detection]:
        """
        Post-process ONNX model output.

        Model output format: [batch, num_values, anchors]
        where num_values = 4 (box) + num_classes (YOLOv8 format - NO objectness)
        
        Class scores are direct sigmoid outputs - use them directly as confidence.

        Args:
            output: Model output [1, num_values, anchors]
            scale: Scale factors (sx, sy)
            padding: Padding (pad_w, pad_h)
            conf_threshold: Confidence threshold

        Returns:
            List of Detection objects
        """
        # Transpose to [anchors, num_values]
        output = np.transpose(output[0], (1, 0))  # [anchors, num_values]

        # Dynamically determine number of classes from output shape
        # YOLOv8 format: [anchors, 4 (box) + num_classes] - NO objectness
        num_classes = output.shape[1] - 4

        # Log detailed output statistics
        max_class_score = float(np.max(output[:, 4:4+num_classes]))
        avg_class_score = float(np.mean(output[:, 4:4+num_classes]))

        # Count high-confidence anchors
        high_conf_anchors = np.sum(np.max(output[:, 4:4+num_classes], axis=1) > conf_threshold)

        logger.info(f"ONNX stats: cls_max={max_class_score:.3f}, cls_avg={avg_class_score:.3f}, high_conf={high_conf_anchors}, num_classes={num_classes}")

        detections = []
        debug_dets = []

        for i, anchor in enumerate(output):
            # Extract box coordinates [cx, cy, w, h]
            cx, cy, w, h = anchor[0:4]

            # Extract class scores (starts at index 4 in YOLOv8 format)
            class_scores = anchor[4:4+num_classes]

            # Get best class
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])

            # Use class confidence directly (YOLOv8 format - no objectness multiplication)
            conf = class_conf

            # Log promising detections for debugging
            if conf > conf_threshold:
                debug_dets.append((i, class_id, conf))

            # Filter by confidence threshold
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
            logger.info(f"ONNX promising: {[(self.FIELD_CLASSES.get(d[1], f'c{d[1]}'), round(d[2],3)) for d in debug_dets[:10]]}")

        logger.info(f"ONNX pre-NMS: {len(detections)} detections")

        # Apply Non-Maximum Suppression (NMS)
        if detections:
            detections = self._apply_nms(detections)

        logger.info(f"ONNX post-NMS: {len(detections)} detections")

        return detections
    
    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.7) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not detections:
            return []

        # Log detections before NMS
        logger.info(f"NMS input: {len(detections)} detections: {[f'{d.class_name}({d.confidence:.2f})' for d in detections]}")
        
        # Convert to format for NMS
        boxes = [d.bbox for d in detections]
        scores = [d.confidence for d in detections]

        # OpenCV NMS with very low score threshold
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.05, nms_threshold=iou_threshold)

        # Handle different OpenCV return formats
        if indices is not None and len(indices) > 0:
            # Convert indices to list (handle both numpy array and list formats)
            if hasattr(indices, 'flatten'):
                indices = indices.flatten()
            result = [detections[int(i)] for i in indices]
            logger.info(f"NMS output: {len(result)} detections retained")
            return result
        else:
            # NMS filtered everything - return all detections (shouldn't happen)
            logger.warning(f"NMS returned empty - keeping all {len(detections)} detections")
            return detections
    
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
        self.class_names = {}

        # Check if file exists
        import os

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            # Get class names from the model itself
            # Ultralytics stores class names in model.names dict
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                logger.info(f"Loaded YOLO model: {model_path} with {len(self.class_names)} classes")
                # Log first few class names for debugging
                sample_classes = {k: v for k, v in list(self.class_names.items())[:5]}
                logger.info(f"Sample class names: {sample_classes}")
            else:
                logger.warning(f"Model has no class names, using empty dict")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Detection]:
        """Run detection on an image."""
        if self.model is None:
            logger.warning("No model loaded, returning empty detections")
            return []

        # Use provided threshold or default from settings
        if conf_threshold is None:
            conf_threshold = settings.YOLO_CONF_THRESHOLD

        try:
            # Run inference
            results = self.model(image, verbose=False, conf=conf_threshold)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue

                    class_id = int(box.cls[0])
                    xyxy = box.xyxy[0].tolist()

                    # Use class names from the model itself, not from settings
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

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

        # Find card detections - accept multiple possible class names
        # The detect_id_card.pt model may have classes like: id_card, card, id, national_id, 
        # front, back, front_side, back_side, back-up, etc.
        CARD_CLASS_NAMES = {
            "id_card", "card", "id", "national_id", 
            "front", "back", "front_side", "back_side", 
            "back-up", "front_up", "back_up", "front_down", "back_down"
        }
        
        # First try to find front-side card detections (prefer front over back)
        card_dets = [
            d for d in detections
            if d.class_name in ("id_card", "card", "id", "national_id", "front", "front_side", "front_up", "front_down")
        ]
        
        # If no front detection, accept any card-related class
        if not card_dets:
            card_dets = [
                d for d in detections
                if d.class_name in CARD_CLASS_NAMES or d.class_id == 0
            ]

        # If no card found but we have other detections, use the highest confidence detection
        # This handles cases where the model uses different class naming
        if not card_dets and detections:
            logger.warning(f"No card detection found, using best available: {detections[0].class_name}")
            best = max(detections, key=lambda d: d.confidence)
            card_dets = [best]

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

        Uses ONNX field detector first (faster, more accurate),
        falls back to YOLO detector if ONNX returns 0 detections.
        
        Maps between ONNX class names (snake_case) and YOLO class names (camelCase).
        """
        # Class name mapping: ONNX (snake_case) -> canonical names
        # Based on actual ONNX model metadata from weights/field_detector.onnx
        # Model class names: first_name, last_name, add_line_1, add_line_2, front_nid,
        #                    back_nid, serial_num, issue_code, expiry_date, job_title,
        #                    gender, religion, marital_status, face, front_logo, address, dob
        ONNX_TO_CANONICAL = {
            'first_name': 'firstName',
            'last_name': 'lastName',
            'add_line_1': 'addressLine1',
            'add_line_2': 'addressLine2',
            'front_nid': 'nid',
            'back_nid': 'nid_back',
            'serial_num': 'serial',
            'issue_code': 'serial',  # Issue code is part of serial
            'expiry_date': 'expiryDate',
            'dob': 'dateOfBirth',
            'job_title': 'jobTitle',
            'gender': 'gender',
            'religion': 'religion',
            'marital_status': 'maritalStatus',
            'face': 'photo',
            'front_logo': 'frontLogo',
            'address': 'address',
        }

        # YOLO (detect_odjects.pt) to canonical mapping
        # Based on NASO7Y model classes from config.py:
        # 0: "address", 1: "demo", 2: "dob", 3: "expiry", 4: "firstName",
        # 5: "front_logo", 24: "lastName", 25: "nid", 26: "nid_back",
        # 27: "photo", 28: "poe", 29: "serial", 30: "watermark_tut"
        YOLO_TO_CANONICAL = {
            'address': 'address',
            'firstName': 'firstName',
            'lastName': 'lastName',
            'nid': 'nid',
            'nid_back': 'nid_back',
            'serial': 'serial',
            'expiry': 'expiryDate',
            'dob': 'dateOfBirth',
            'photo': 'photo',
            'front_logo': 'frontLogo',
            'poe': 'address',  # Place of employment -> address
            'demo': 'demo',
            'watermark_tut': 'watermark_tut',
        }
        
        # Valid fields we want to extract
        VALID_FIELDS = {
            'firstName', 'lastName', 'nid', 'serial', 'address',
            'addressLine1', 'addressLine2',  # Address line fields
            'expiryDate', 'dateOfBirth', 'jobTitle', 'gender',
            'religion', 'maritalStatus', 'photo', 'frontLogo',
            # Additional NASO7Y model classes
            'nid_back', 'demo', 'poe',
        }

        # Try ONNX detector first with lower threshold for better recall
        if self.field_detector_onnx.session is not None:
            try:
                start_time = time.time()
                detections = self.field_detector_onnx.detect(card_image, conf_threshold=0.18)
                onnx_time = (time.time() - start_time) * 1000
                
                # Map ONNX class names to canonical names
                detected_fields = {}
                for det in detections:
                    canonical_name = ONNX_TO_CANONICAL.get(det.class_name, det.class_name)

                    # Skip back_nid and other unwanted fields
                    if det.class_name in ('back_nid',):
                        continue

                    # Skip if not a valid field
                    if canonical_name not in VALID_FIELDS:
                        logger.debug(f"Skipping non-valid field: {det.class_name} -> {canonical_name}")
                        continue
                    
                    x1, y1, x2, y2 = det.bbox
                    h, w = card_image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = card_image[y1:y2, x1:x2]
                    if crop.size > 0:
                        detected_fields[canonical_name] = (crop, det.confidence)
                
                detected_classes = list(detected_fields.keys())
                logger.info(f"ONNX field detection: {len(detected_fields)} fields in {onnx_time:.1f}ms - {detected_classes}")

                if detected_fields:
                    logger.info(f"ONNX successfully extracted {len(detected_fields)} fields")
                    return detected_fields

                logger.warning(f"ONNX returned {len(detections)} detections, falling back to YOLO")

            except Exception as e:
                logger.warning(f"ONNX field detection failed: {e}, falling back to YOLO")

        # Fallback to YOLO detector when ONNX fails or returns 0 detections
        logger.info("Using YOLO field detector as fallback")
        if self.field_detector_yolo.model is None:
            logger.warning("YOLO model not loaded - cannot extract fields")
            return {}
            
        detections = self.field_detector_yolo.detect(card_image, conf_threshold=0.25)

        if not detections:
            logger.warning("YOLO also returned 0 detections - no fields extracted")
            return {}

        fields = {}
        for det in detections:
            canonical_name = YOLO_TO_CANONICAL.get(det.class_name, det.class_name)
            
            # Skip invalid_* detections
            if det.class_name.startswith('invalid_'):
                continue
            
            if canonical_name in VALID_FIELDS:
                x1, y1, x2, y2 = det.bbox
                h, w = card_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = card_image[y1:y2, x1:x2]
                if crop.size > 0:
                    fields[canonical_name] = (crop, det.confidence)
                    logger.info(f"YOLO field detected: {det.class_name} -> {canonical_name} (conf: {det.confidence:.2f})")

        logger.info(f"YOLO extracted {len(fields)} fields")
        return fields
