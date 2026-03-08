"""
Egyptian ID Card Side Classifier
Detects if an image contains: front side, back side, or both sides together.

Uses visual features:
- Photo presence detection (face region on front side)
- Text pattern analysis (Arabic text density)
- Layout feature extraction (field positions)
- Color/texture analysis
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum

from app.core.logger import logger
from app.core.config import settings


class CardSide(str, Enum):
    """Enum for card side classification."""
    FRONT = "front"
    BACK = "back"
    BOTH = "both"
    UNKNOWN = "unknown"


@dataclass
class SideClassification:
    """Result of side classification."""
    side: CardSide
    confidence: float
    details: dict


class SideClassifier:
    """
    Classifier for Egyptian ID card sides.
    
    Front side characteristics:
    - Photo region on the left (circular/oval face photo)
    - Name fields (firstName, lastName) in upper right
    - National ID number in middle
    - Address at bottom
    
    Back side characteristics:
    - Address lines at top
    - National ID number (may be present)
    - Issue date, expiry date at bottom
    - No photo region
    
    Both sides (dual-side image):
    - Two distinct card regions (horizontal or vertical split)
    - Combined features of front and back
    """
    
    def __init__(self):
        """Initialize the side classifier."""
        # Egyptian ID card aspect ratio (approximately 2.35:1)
        self.ID_CARD_ASPECT_RATIO = 2.35
        self.ASPECT_RATIO_TOLERANCE = 0.3
        
        # Minimum card dimensions after detection
        self.MIN_CARD_WIDTH = 300
        self.MIN_CARD_HEIGHT = 150
        
        # Photo detection parameters
        self.PHOTO_REGION_X_RATIO = 0.30  # Left 30% of card
        self.PHOTO_MIN_RADIUS = 30
        self.PHOTO_MAX_RADIUS = 150
        
        # Text density thresholds
        self.FRONT_TEXT_DENSITY_THRESHOLD = 0.15
        self.BACK_TEXT_DENSITY_THRESHOLD = 0.20
        
    def classify(self, image: np.ndarray) -> SideClassification:
        """
        Classify an image as front, back, or both sides.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            SideClassification with side, confidence, and details
        """
        if image is None or image.size == 0:
            return SideClassification(
                side=CardSide.UNKNOWN,
                confidence=0.0,
                details={"error": "Empty or invalid image"}
            )
        
        # Step 1: Check if image contains two cards (dual-side)
        dual_result = self._detect_dual_side(image)
        if dual_result.side == CardSide.BOTH and dual_result.confidence > 0.7:
            logger.info(f"Dual-side detected with confidence {dual_result.confidence:.2f}")
            return dual_result
        
        # Step 2: Preprocess for single-card analysis
        processed = self._preprocess_for_classification(image)
        
        # Step 3: Extract features
        features = self._extract_features(processed, image)
        
        # Step 4: Classify based on features
        side, confidence, details = self._classify_from_features(features)
        
        return SideClassification(
            side=side,
            confidence=confidence,
            details=details
        )
    
    def _detect_dual_side(self, image: np.ndarray) -> SideClassification:
        """
        Detect if image contains both front and back sides.
        
        Strategies:
        1. Aspect ratio analysis (dual-side images are wider or taller)
        2. Contour detection for two separate card regions
        3. Split analysis (check if image can be divided into two card-like regions)
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h if h > 0 else 0
        
        details = {
            "image_size": {"width": w, "height": h},
            "aspect_ratio": aspect_ratio,
            "methods_used": []
        }
        
        # Method 1: Aspect ratio check
        # Dual-side horizontal: ~4.7:1 (two cards side by side)
        # Dual-side vertical: ~1.17:1 (two cards stacked)
        # Single card: ~2.35:1
        
        # More conservative thresholds to avoid false positives
        is_horizontal_dual = aspect_ratio > 4.0  # Very wide images only
        is_vertical_dual = aspect_ratio < 0.8    # Very tall/square images only
        
        if is_horizontal_dual:
            details["methods_used"].append("aspect_ratio_horizontal")
            details["split_orientation"] = "horizontal"
            return SideClassification(
                side=CardSide.BOTH,
                confidence=0.85,
                details=details
            )
        
        if is_vertical_dual:
            details["methods_used"].append("aspect_ratio_vertical")
            details["split_orientation"] = "vertical"
            return SideClassification(
                side=CardSide.BOTH,
                confidence=0.85,
                details=details
            )
        
        # Method 2: Contour-based detection for two cards
        two_cards_detected, orientation = self._detect_two_card_contours(image)
        if two_cards_detected:
            details["methods_used"].append("contour_detection")
            details["split_orientation"] = orientation
            return SideClassification(
                side=CardSide.BOTH,
                confidence=0.90,
                details=details
            )
        
        # Method 3: Projection profile analysis
        # Check for gaps in horizontal/vertical projection
        split_detected, orientation = self._analyze_projection_profile(image)
        if split_detected:
            details["methods_used"].append("projection_profile")
            details["split_orientation"] = orientation
            return SideClassification(
                side=CardSide.BOTH,
                confidence=0.75,
                details=details
            )
        
        return SideClassification(
            side=CardSide.UNKNOWN,
            confidence=0.0,
            details=details
        )
    
    def _detect_two_card_contours(self, image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Detect if there are two separate card contours in the image.
        
        Returns:
            Tuple of (detected, orientation) where orientation is "horizontal" or "vertical"
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to connect card regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for card-sized contours
        img_area = image.shape[0] * image.shape[1]
        card_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Card should be at least 20% of image (for dual-side)
            if area > img_area * 0.15:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 0
                # Check if contour is card-like
                if 1.5 < aspect < 3.0:
                    card_contours.append((x, y, cw, ch))
        
        # Check if we found exactly 2 card-like contours
        if len(card_contours) == 2:
            c1, c2 = card_contours
            # Determine orientation
            # Horizontal: cards are side by side (x-coordinates differ significantly)
            # Vertical: cards are stacked (y-coordinates differ significantly)
            
            x_overlap = not (c1[0] + c1[2] < c2[0] or c2[0] + c2[2] < c1[0])
            y_overlap = not (c1[1] + c1[3] < c2[1] or c2[1] + c2[3] < c1[1])
            
            if not x_overlap:
                return True, "horizontal"
            elif not y_overlap:
                return True, "vertical"
        
        return False, None
    
    def _analyze_projection_profile(self, image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Analyze horizontal and vertical projection profiles to detect card splits.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Horizontal projection (sum of each row)
        h_proj = np.sum(thresh > 0, axis=1)
        # Vertical projection (sum of each column)
        v_proj = np.sum(thresh > 0, axis=0)
        
        # Normalize
        h_proj = h_proj / np.max(h_proj) if np.max(h_proj) > 0 else h_proj
        v_proj = v_proj / np.max(v_proj) if np.max(v_proj) > 0 else v_proj
        
        # Find gaps (regions with low projection values)
        h_gap = self._find_projection_gap(h_proj, threshold=0.3, min_width=20)
        v_gap = self._find_projection_gap(v_proj, threshold=0.3, min_width=20)
        
        if h_gap is not None:
            return True, "horizontal"
        elif v_gap is not None:
            return True, "vertical"
        
        return False, None
    
    def _find_projection_gap(self, projection: np.ndarray, threshold: float, 
                             min_width: int) -> Optional[int]:
        """Find a gap in projection profile."""
        below_threshold = projection < threshold
        
        # Find continuous regions below threshold
        gap_start = None
        for i in range(len(below_threshold) - 1):
            if below_threshold[i] and below_threshold[i + 1]:
                if gap_start is None:
                    gap_start = i
            elif gap_start is not None:
                gap_width = i - gap_start
                if gap_width >= min_width:
                    return gap_start
                gap_start = None
        
        return None
    
    def _preprocess_for_classification(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for classification."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size for consistent analysis
        target_width = 800
        h, w = gray.shape
        if w != target_width:
            scale = target_width / w
            new_h = int(h * scale)
            gray = cv2.resize(gray, (target_width, new_h), interpolation=cv2.INTER_AREA)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _extract_features(self, processed: np.ndarray, 
                          original: np.ndarray) -> dict:
        """Extract visual features for classification."""
        h, w = processed.shape
        
        features = {
            "photo_score": 0.0,
            "text_density": 0.0,
            "layout_score": 0.0,
            "field_positions": {}
        }
        
        # Feature 1: Photo detection (left side of card)
        photo_region_x = int(w * self.PHOTO_REGION_X_RATIO)
        photo_region = processed[:, :photo_region_x]
        features["photo_score"] = self._detect_photo_region(photo_region)
        
        # Feature 2: Text density analysis
        features["text_density"] = self._analyze_text_density(processed)
        
        # Feature 3: Layout analysis (field position detection)
        features["layout_score"], features["field_positions"] = self._analyze_layout(
            processed, original
        )
        
        # Feature 4: Edge density (front side has more structured edges due to photo)
        features["edge_density"] = self._analyze_edge_density(processed)
        
        return features
    
    def _detect_photo_region(self, region: np.ndarray) -> float:
        """
        Detect if region contains a photo (face area).
        
        Strategies:
        1. Circular/oval shape detection
        2. Texture analysis (face has specific texture pattern)
        3. Intensity distribution (face has characteristic histogram)
        """
        if region.size == 0:
            return 0.0
        
        h, w = region.shape
        
        # Method 1: Hough Circle detection
        circles = cv2.HoughCircles(
            region, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30,
            minRadius=self.PHOTO_MIN_RADIUS,
            maxRadius=self.PHOTO_MAX_RADIUS
        )
        
        if circles is not None and len(circles[0]) > 0:
            return 0.9
        
        # Method 2: Contour-based oval detection
        _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            
            # Fit ellipse
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (center, axes, orientation) = ellipse
                    aspect_ratio = axes[0] / axes[1] if axes[1] > 0 else 1
                    
                    # Photo region is roughly circular/oval (aspect ratio close to 1)
                    if 0.7 < aspect_ratio < 1.4:
                        # Check if it's in the expected position (left side, vertically centered)
                        cy = center[1]
                        if 0.3 * h < cy < 0.7 * h:
                            return 0.8
                except:
                    continue
        
        # Method 3: Texture analysis (variance in photo region)
        # Face photos typically have higher variance than background
        variance = np.var(region)
        if variance > 1500:  # Threshold determined empirically
            return 0.6
        
        return 0.2
    
    def _analyze_text_density(self, image: np.ndarray) -> float:
        """
        Analyze text density in the image.
        Back side typically has more text (address lines).
        """
        # Threshold to get text regions
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate text density
        total_pixels = image.shape[0] * image.shape[1]
        text_pixels = np.sum(thresh > 0)
        
        density = text_pixels / total_pixels if total_pixels > 0 else 0
        
        return min(density * 5, 1.0)  # Normalize to 0-1
    
    def _analyze_layout(self, processed: np.ndarray, 
                        original: np.ndarray) -> Tuple[float, dict]:
        """
        Analyze layout to determine side.
        
        Front side layout:
        - Photo on left
        - Names in upper right
        - NID in middle
        - Address at bottom
        
        Back side layout:
        - Address lines at top
        - NID (optional) in middle
        - Dates at bottom
        """
        h, w = processed.shape
        positions = {}
        
        # Divide card into regions
        regions = {
            "upper_left": processed[:h//3, :w//3],
            "upper_right": processed[:h//3, w//3:],
            "middle": processed[h//3:2*h//3, :],
            "lower": processed[2*h//3:, :],
        }
        
        # Analyze each region
        for name, region in regions.items():
            # Text density in region
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            density = np.sum(thresh > 0) / region.size if region.size > 0 else 0
            positions[name] = density
        
        # Scoring based on expected patterns
        front_score = 0.0
        back_score = 0.0
        
        # Front side indicators
        if positions.get("upper_left", 0) < 0.1:  # Photo region (less text)
            front_score += 0.3
        if positions.get("upper_right", 0) > 0.15:  # Names region
            front_score += 0.2
        if positions.get("middle", 0) > 0.1:  # NID region
            front_score += 0.2
        
        # Back side indicators
        if positions.get("upper_left", 0) > 0.15:  # Address line 1
            back_score += 0.3
        if positions.get("upper_right", 0) > 0.15:  # Address line 2
            back_score += 0.2
        if positions.get("lower", 0) > 0.1:  # Dates
            back_score += 0.2
        
        # Determine which score is higher
        if front_score > back_score:
            return front_score, positions
        else:
            return back_score, positions
    
    def _analyze_edge_density(self, image: np.ndarray) -> float:
        """Analyze edge density using Canny edge detection."""
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return min(edge_density * 10, 1.0)
    
    def _classify_from_features(self, features: dict) -> Tuple[CardSide, float, dict]:
        """
        Classify side based on extracted features.
        
        Uses a simple rule-based classifier with confidence scoring.
        """
        photo_score = features.get("photo_score", 0.0)
        text_density = features.get("text_density", 0.0)
        layout_score = features.get("layout_score", 0.0)
        edge_density = features.get("edge_density", 0.0)
        
        details = {
            "features": features,
            "decision_factors": []
        }
        
        # Decision logic
        front_confidence = 0.0
        back_confidence = 0.0
        
        # High photo score strongly indicates front side
        if photo_score > 0.7:
            front_confidence += 0.5
            details["decision_factors"].append("photo_detected")
        elif photo_score > 0.4:
            front_confidence += 0.3
            details["decision_factors"].append("possible_photo")
        else:
            back_confidence += 0.3
            details["decision_factors"].append("no_photo")
        
        # Text density helps distinguish
        if text_density > 0.4:
            back_confidence += 0.3
            details["decision_factors"].append("high_text_density")
        elif text_density > 0.2:
            front_confidence += 0.2
            details["decision_factors"].append("moderate_text_density")
        
        # Layout analysis
        field_positions = features.get("field_positions", {})
        if field_positions:
            # Front: upper_left should have less text (photo area)
            if field_positions.get("upper_left", 0) < 0.15:
                front_confidence += 0.2
                details["decision_factors"].append("layout_matches_front")
            else:
                back_confidence += 0.2
                details["decision_factors"].append("layout_matches_back")
        
        # Edge density (front has more structured edges)
        if edge_density > 0.3:
            front_confidence += 0.1
            details["decision_factors"].append("high_edge_density")
        
        # Normalize confidences
        total = front_confidence + back_confidence
        if total > 0:
            front_confidence = front_confidence / total
            back_confidence = back_confidence / total
        
        # Final decision
        if front_confidence > back_confidence:
            if front_confidence > 0.7:
                confidence = front_confidence
            else:
                confidence = 0.5 + front_confidence * 0.3
            return CardSide.FRONT, confidence, details
        elif back_confidence > front_confidence:
            if back_confidence > 0.7:
                confidence = back_confidence
            else:
                confidence = 0.5 + back_confidence * 0.3
            return CardSide.BACK, confidence, details
        else:
            return CardSide.UNKNOWN, 0.5, details
    
    def split_dual_side_image(self, image: np.ndarray, 
                              orientation: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dual-side image into two halves.
        
        Args:
            image: Input image containing both sides
            orientation: "horizontal" or "vertical" (auto-detected if None)
            
        Returns:
            Tuple of (first_half, second_half)
        """
        h, w = image.shape[:2]
        
        # Auto-detect orientation if not provided
        if orientation is None:
            aspect_ratio = w / h if h > 0 else 0
            orientation = "horizontal" if aspect_ratio > 2.0 else "vertical"
        
        if orientation == "horizontal":
            # Split vertically (left/right)
            mid_x = w // 2
            
            # Add small overlap for context
            overlap = int(w * 0.02)  # 2% overlap
            
            first_half = image[:, :mid_x + overlap]
            second_half = image[:, mid_x - overlap:]
        else:
            # Split horizontally (top/bottom)
            mid_y = h // 2
            
            overlap = int(h * 0.02)
            
            first_half = image[:mid_y + overlap, :]
            second_half = image[mid_y - overlap:, :]
        
        return first_half, second_half


# Singleton instance
_classifier: Optional[SideClassifier] = None


def get_side_classifier() -> SideClassifier:
    """Get or create side classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = SideClassifier()
    return _classifier


def classify_card_side(image: np.ndarray) -> SideClassification:
    """
    Convenience function to classify card side.
    
    Args:
        image: Input image
        
    Returns:
        SideClassification result
    """
    classifier = get_side_classifier()
    return classifier.classify(image)
