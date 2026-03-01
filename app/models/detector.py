"""
ONNX/YOLO-based Detector for Egyptian ID Card
Uses Ultralytics YOLO for .pt models.
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from app.core.config import settings
from app.core.logger import logger


@dataclass
class Detection:
    """Represents a single detection result."""

    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


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
    """

    def __init__(self):
        """Initialize the detector with card and field detection models."""
        logger.info("Initializing Egyptian ID Detector...")

        self.card_detector = YOLODetector(settings.YOLO_CARD_MODEL)
        self.field_detector = YOLODetector(settings.YOLO_FIELDS_MODEL)
        self.nid_detector = YOLODetector(settings.YOLO_NID_MODEL)

        logger.info("Egyptian ID Detector initialized")

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
        """Detect and crop individual fields from the card image."""
        detections = self.field_detector.detect(card_image)

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
