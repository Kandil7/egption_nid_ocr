"""
OCR Visualization Service
Generates detailed overlay images showing each step of the OCR pipeline.
"""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

from app.core.logger import logger


@dataclass
class StepInfo:
    """Information about a visualization step."""
    name: str
    description: str
    image_base64: str
    width: int
    height: int
    metadata: Dict[str, Any]


@dataclass
class VisualizationResult:
    """Complete visualization result for all OCR steps."""
    steps: List[StepInfo]
    total_steps: int
    processing_time_ms: int
    field_name: str
    extracted_text: str
    confidence: float


class OCRVisualizer:
    """
    Generates detailed visualization overlays for OCR pipeline steps.
    """

    def __init__(self):
        """Initialize visualizer."""
        self.debug_dir = Path("debug/ocr_steps")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def draw_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2, 
                  label: str = "") -> np.ndarray:
        """Draw bounding box with label on image."""
        x1, y1, x2, y2 = bbox
        img = image.copy()
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label if provided
        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x1, y1 - 5), font, font_scale, 
                       (255, 255, 255), text_thickness)
        
        return img

    def add_text_overlay(self, image: np.ndarray, text: str, 
                         position: str = "bottom") -> np.ndarray:
        """Add text overlay to image."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create background for text
        text_bg_height = 40
        if position == "top":
            cv2.rectangle(img, (0, 0), (w, text_bg_height), (0, 0, 0), -1)
            y_pos = text_bg_height - 10
        else:  # bottom
            cv2.rectangle(img, (0, h - text_bg_height), (w, h), (0, 0, 0), -1)
            y_pos = h - 10
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        cv2.putText(img, text, (10, y_pos), font, font_scale, 
                   (255, 255, 255), 2)
        
        return img

    def create_side_by_side(self, img1: np.ndarray, img2: np.ndarray, 
                           label1: str = "", label2: str = "") -> np.ndarray:
        """Create side-by-side comparison of two images."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize to same height
        target_h = max(h1, h2)
        if h1 != target_h:
            scale = target_h / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_h))
        
        # Add labels
        if label1:
            img1 = self.add_text_overlay(img1, label1, "top")
        if label2:
            img2 = self.add_text_overlay(img2, label2, "top")
        
        # Combine horizontally
        combined = np.hstack([img1, img2])
        return combined

    def visualize_card_detection(self, image: np.ndarray, 
                                 card_bbox: Optional[Tuple[int, int, int, int]] = None,
                                 card_detected: bool = False) -> StepInfo:
        """Visualize card detection step."""
        img = image.copy()
        
        if card_detected and card_bbox:
            img = self.draw_bbox(img, card_bbox, (0, 255, 0), 3, "ID Card")
            status = "Card Detected"
        else:
            status = "No Card Detected - Using Full Image"
        
        img = self.add_text_overlay(img, status, "bottom")
        
        return StepInfo(
            name="Card Detection",
            description="YOLO-based ID card detection",
            image_base64=self.encode_image(img),
            width=img.shape[1],
            height=img.shape[0],
            metadata={
                "detected": card_detected,
                "bbox": card_bbox if card_bbox else None
            }
        )

    def visualize_field_detection(self, image: np.ndarray,
                                  fields: Dict[str, Tuple[np.ndarray, float]],
                                  field_order: List[str] = None) -> List[StepInfo]:
        """Visualize field detection for each field."""
        steps = []
        img = image.copy()
        
        # Color map for different fields
        colors = {
            "nid": (255, 0, 0),      # Red
            "id_number": (255, 0, 0),
            "front_nid": (255, 0, 0),
            "firstName": (0, 255, 0),  # Green
            "lastName": (0, 255, 0),
            "address": (0, 0, 255),    # Blue
            "serial": (255, 255, 0),   # Cyan
            "add_line_1": (0, 255, 255),  # Yellow
            "add_line_2": (0, 255, 255),
        }
        
        if field_order is None:
            field_order = list(fields.keys())
        
        # Draw all fields on main image
        for field_name, (field_img, conf) in fields.items():
            # We need to get bbox from somewhere - for now skip main visualization
            color = colors.get(field_name, (255, 255, 255))
        
        # Overall field detection
        overall_img = img.copy()
        overall_img = self.add_text_overlay(overall_img, f"Fields Detected: {len(fields)}", "bottom")
        
        steps.append(StepInfo(
            name="Field Detection Overview",
            description=f"Detected {len(fields)} fields using YOLO",
            image_base64=self.encode_image(overall_img),
            width=overall_img.shape[1],
            height=overall_img.shape[0],
            metadata={"field_count": len(fields)}
        ))
        
        # Individual field visualizations
        for field_name in field_order:
            if field_name not in fields:
                continue
            
            field_img, conf = fields[field_name]
            field_vis = field_img.copy()
            
            # Add field name and confidence
            label = f"{field_name} (conf: {conf:.2f})"
            field_vis = self.add_text_overlay(field_vis, label, "top")
            
            # Resize for display if too large
            h, w = field_vis.shape[:2]
            if w > 400:
                scale = 400 / w
                field_vis = cv2.resize(field_vis, (int(w * scale), int(h * scale)))
            
            steps.append(StepInfo(
                name=f"Field: {field_name}",
                description=f"Cropped field region for OCR processing",
                image_base64=self.encode_image(field_vis),
                width=field_vis.shape[1],
                height=field_vis.shape[0],
                metadata={
                    "field_name": field_name,
                    "confidence": round(conf, 3),
                    "original_size": (w, h)
                }
            ))
        
        return steps

    def visualize_preprocessing_variations(self, original: np.ndarray,
                                          variations: Dict[str, np.ndarray],
                                          field_name: str = "") -> List[StepInfo]:
        """Visualize preprocessing variations applied to field image."""
        steps = []
        
        # Original image
        orig_vis = original.copy()
        if field_name:
            orig_vis = self.add_text_overlay(orig_vis, f"Original: {field_name}", "top")
        
        steps.append(StepInfo(
            name="Original Field Image",
            description="Raw cropped field before preprocessing",
            image_base64=self.encode_image(orig_vis),
            width=orig_vis.shape[1],
            height=orig_vis.shape[0],
            metadata={"type": "original"}
        ))
        
        # Each variation
        for var_name, var_img in variations.items():
            var_vis = var_img.copy()
            label = f"Preprocessing: {var_name}"
            var_vis = self.add_text_overlay(var_vis, label, "top")
            
            # Convert grayscale to BGR if needed
            if len(var_vis.shape) == 2:
                var_vis = cv2.cvtColor(var_vis, cv2.COLOR_GRAY2BGR)
            
            steps.append(StepInfo(
                name=f"Preprocessing: {var_name}",
                description=f"Preprocessing variation applied for better OCR",
                image_base64=self.encode_image(var_vis),
                width=var_vis.shape[1],
                height=var_vis.shape[0],
                metadata={"variation_name": var_name}
            ))
        
        return steps

    def visualize_ocr_result(self, image: np.ndarray,
                            ocr_results: List[Dict],
                            extracted_text: str,
                            confidence: float,
                            engine: str = "Unknown") -> StepInfo:
        """Visualize OCR recognition result."""
        img = image.copy()
        
        # Draw OCR result boxes if available
        for result in ocr_results:
            if 'bbox' in result:
                bbox = result['bbox']
                text = result.get('text', '')
                conf = result.get('confidence', 0)
                img = self.draw_bbox(img, bbox, (0, 255, 0), 2, f"{text} ({conf:.2f})")
        
        # Add result overlay
        result_text = f"OCR: {extracted_text} | Conf: {confidence:.2f} | Engine: {engine}"
        img = self.add_text_overlay(img, result_text, "bottom")
        
        return StepInfo(
            name="OCR Result",
            description=f"Text extracted using {engine}",
            image_base64=self.encode_image(img),
            width=img.shape[1],
            height=img.shape[0],
            metadata={
                "extracted_text": extracted_text,
                "confidence": round(confidence, 3),
                "engine": engine
            }
        )

    def visualize_nid_ensemble(self, image: np.ndarray,
                              tesseract_result: str,
                              easyocr_result: str,
                              voted_result: Optional[str],
                              tesseract_conf: float,
                              easyocr_conf: float) -> StepInfo:
        """Visualize NID ensemble voting result."""
        # Create comparison visualization
        h, w = image.shape[:2]
        
        # Create text lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        line_height = 40
        
        img = image.copy()
        
        # Add result lines
        y_offset = 10
        results = [
            f"Tesseract: {tesseract_result} (conf: {tesseract_conf:.2f})",
            f"EasyOCR: {easyocr_result} (conf: {easyocr_conf:.2f})",
        ]
        
        if voted_result:
            results.append(f"ENSEMBLE VOTE: {voted_result} (conf: 0.95)")
        
        for text in results:
            # Text background
            cv2.rectangle(img, (10, y_offset), (w - 10, y_offset + line_height), 
                         (0, 0, 0), -1)
            # Text
            cv2.putText(img, text, (15, y_offset + line_height - 10), 
                       font, font_scale, (255, 255, 255), 2)
            y_offset += line_height + 10
        
        return StepInfo(
            name="NID Ensemble Voting",
            description="Combined results from Tesseract + EasyOCR",
            image_base64=self.encode_image(img),
            width=img.shape[1],
            height=img.shape[0],
            metadata={
                "tesseract_result": tesseract_result,
                "easyocr_result": easyocr_result,
                "voted_result": voted_result,
                "tesseract_confidence": round(tesseract_conf, 3),
                "easyocr_confidence": round(easyocr_conf, 3)
            }
        )

    def generate_full_pipeline_visualization(self, 
                                            original_image: np.ndarray,
                                            card_bbox: Optional[Tuple],
                                            card_detected: bool,
                                            fields: Dict[str, Tuple],
                                            ocr_results: Dict[str, Dict],
                                            final_extracted: Dict[str, str],
                                            final_confidence: Dict[str, float]) -> VisualizationResult:
        """Generate complete visualization of entire OCR pipeline."""
        start_time = time.time()
        steps = []
        
        # Step 1: Card Detection
        steps.append(self.visualize_card_detection(
            original_image, card_bbox, card_detected
        ))
        
        # Step 2: Field Detection
        if fields:
            field_steps = self.visualize_field_detection(original_image, fields)
            steps.extend(field_steps)
        
        # Step 3: OCR Results for each field
        for field_name, result_data in ocr_results.items():
            if field_name in fields:
                field_img, _ = fields[field_name]
                
                # OCR result step
                ocr_step = self.visualize_ocr_result(
                    field_img,
                    result_data.get('boxes', []),
                    final_extracted.get(field_name, ""),
                    final_confidence.get(field_name, 0),
                    result_data.get('engine', 'Unknown')
                )
                steps.append(ocr_step)
        
        processing_ms = int((time.time() - start_time) * 1000)
        
        return VisualizationResult(
            steps=steps,
            total_steps=len(steps),
            processing_time_ms=processing_ms,
            field_name="all",
            extracted_text=str(final_extracted),
            confidence=float(np.mean(list(final_confidence.values()))) if final_confidence else 0
        )

    def save_step_images(self, steps: List[StepInfo], prefix: str = "step"):
        """Save step images to debug directory."""
        for i, step in enumerate(steps):
            img_data = base64.b64decode(step.image_base64)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            filename = self.debug_dir / f"{prefix}_step_{i:03d}_{step.name.replace(' ', '_')}.jpg"
            cv2.imwrite(str(filename), img)
            logger.debug(f"Saved step image: {filename}")

    def _vote_nid_results(self, results: list) -> str:
        """
        Vote on multiple NID results to get the most likely correct one.

        Args:
            results: List of 14-digit NID strings

        Returns:
            Voted NID string
        """
        if not results:
            return ""

        # Position-wise voting
        voted_digits = []
        for pos in range(14):
            digit_votes = [r[pos] for r in results if len(r) == 14 and pos < len(r)]
            if digit_votes:
                # Most common digit at this position
                voted = max(set(digit_votes), key=digit_votes.count)
                voted_digits.append(voted)
            else:
                voted_digits.append('0')

        return ''.join(voted_digits)
