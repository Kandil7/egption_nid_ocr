"""
Specialized NID Extraction Module
Handles Egyptian National ID number extraction with multiple strategies.

Priority:
1. front_nid field (highest priority)
2. back_nid field (fallback)
3. Cross-validation between both sides when available
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from app.core.logger import logger
from app.utils.text_utils import _normalize_digits, _fix_common_digit_ocr_errors, _fix_nid_century_digit, _is_valid_nid_format


@dataclass
class NIDCandidate:
    """Candidate NID result with metadata."""
    text: str
    confidence: float
    source: str
    digit_count: int
    is_valid_format: bool


@dataclass
class NIDExtractionResult:
    """Complete NID extraction result with cross-validation info."""
    nid: str
    confidence: float
    source: str  # "front", "back", "both_matched", "front_mismatch"
    front_nid: Optional[str] = None
    back_nid: Optional[str] = None
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class NIDExtractor:
    """
    Specialized extractor for Egyptian National ID numbers.
    Uses multiple strategies to maximize extraction accuracy.
    """

    def __init__(self):
        """Initialize NID extractor."""
        self.min_digit_count = 10  # Minimum digits to consider

    def extract(self, card_image: np.ndarray, nid_crop: Optional[np.ndarray] = None, 
                recognize_func: Optional[callable] = None) -> Tuple[str, float]:
        """
        Extract NID from card image using multiple strategies.

        Args:
            card_image: Full card image
            nid_crop: Pre-cropped NID field (if available)
            recognize_func: Function to recognize text in a crop (img -> text)

        Returns:
            Tuple of (extracted_nid, confidence)
        """
        self.recognize_func = recognize_func
        candidates = []

        # Strategy 1: Use provided NID crop
        if nid_crop is not None:
            logger.debug("NID extractor: Using provided crop")
            crop_candidates = self._extract_from_crop(nid_crop, "nid_crop")
            candidates.extend(crop_candidates)

        # Strategy 2: Scan multiple NID regions on the card
        logger.debug("NID extractor: Scanning card regions")
        region_candidates = self._extract_from_card_regions(card_image)
        candidates.extend(region_candidates)

        # Strategy 3: Full card scan with NID pattern
        logger.debug("NID extractor: Full card scan")
        full_candidates = self._extract_full_card_scan(card_image)
        candidates.extend(full_candidates)

        # Select best candidate
        return self._select_best_candidate(candidates)

    def _extract_from_crop(self, crop: np.ndarray, source: str) -> List[NIDCandidate]:
        """Extract NID from a cropped region."""
        candidates = []

        # Try multiple preprocessing approaches
        preprocessings = self._get_preprocessing_variants(crop)

        for i, (prep_name, prep_img) in enumerate(preprocessings):
            # Extract digits using contour analysis
            digit_regions = self._find_digit_regions(prep_img)
            
            if len(digit_regions) >= self.min_digit_count:
                # Sort regions left to right and extract digits
                digit_regions = sorted(digit_regions, key=lambda r: r[0])
                digits = self._extract_digits_from_regions(prep_img, digit_regions)
                
                if len(digits) >= self.min_digit_count:
                    candidates.append(NIDCandidate(
                        text=digits[:14],
                        confidence=0.7,
                        source=f"{source}_{prep_name}",
                        digit_count=len(digits[:14]),
                        is_valid_format=_is_valid_nid_format(digits[:14]) if len(digits) >= 14 else False
                    ))

        return candidates

    def _extract_from_card_regions(self, card_image: np.ndarray) -> List[NIDCandidate]:
        """Extract NID by scanning typical NID locations on the card."""
        candidates = []
        h, w = card_image.shape[:2]

        # Define typical NID regions (normalized coordinates)
        # Front side: Usually in middle-right area
        # Back side: Usually at top or bottom
        regions = [
            # Front side regions
            (int(w * 0.3), int(h * 0.4), int(w * 0.95), int(h * 0.65), "front_middle"),
            (int(w * 0.3), int(h * 0.3), int(w * 0.95), int(h * 0.55), "front_upper_middle"),
            (int(w * 0.3), int(h * 0.5), int(w * 0.95), int(h * 0.75), "front_lower_middle"),
            # Back side regions
            (int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.25), "back_top"),
            (int(w * 0.1), int(h * 0.75), int(w * 0.9), int(h * 0.95), "back_bottom"),
            # Full width scans
            (int(w * 0.2), int(h * 0.35), int(w * 0.95), int(h * 0.65), "center_right"),
        ]

        for x1, y1, x2, y2, region_name in regions:
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            region = card_image[y1:y2, x1:x2]
            
            # Skip too small regions
            if region.shape[0] < 30 or region.shape[1] < 100:
                continue

            # Extract from this region
            region_candidates = self._extract_from_crop(region, region_name)
            candidates.extend(region_candidates)

        return candidates

    def _extract_full_card_scan(self, card_image: np.ndarray) -> List[NIDCandidate]:
        """Scan entire card for 14-digit sequences."""
        candidates = []

        # Convert to grayscale and enhance
        if len(card_image.shape) == 3:
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_image.copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect potential digit contours
        digit_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter for digit-like shapes
            if 10 <= h <= 100 and 5 <= w <= 50 and h > w:
                digit_contours.append((x, y, w, h))

        # If we found enough potential digits, try to extract
        if len(digit_contours) >= self.min_digit_count:
            # Sort by y-coordinate to group into lines
            digit_contours.sort(key=lambda c: c[1])
            
            # Group into horizontal lines
            lines = self._group_into_lines(digit_contours)
            
            for line in lines:
                # Sort line left to right
                line.sort(key=lambda c: c[0])
                
                if len(line) < self.min_digit_count:
                    continue
                
                # OCR the entire line as one crop instead of per-digit
                if self.recognize_func:
                    x_min = min(c[0] for c in line)
                    x_max = max(c[0] + c[2] for c in line)
                    y_min = min(c[1] for c in line)
                    y_max = max(c[1] + c[3] for c in line)
                    pad = 4
                    y1 = max(0, y_min - pad)
                    y2 = min(binary.shape[0], y_max + pad)
                    x1 = max(0, x_min - pad)
                    x2 = min(binary.shape[1], x_max + pad)
                    line_crop = binary[y1:y2, x1:x2]
                    
                    if line_crop.size > 0:
                        # Convert to BGR for OCR
                        if len(line_crop.shape) == 2:
                            line_crop = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2BGR)
                        text = self.recognize_func(line_crop)
                        digits_text = re.sub(r'\D', '', text)
                        digits_text = _normalize_digits(digits_text)
                        digits_text = _fix_common_digit_ocr_errors(digits_text)
                        
                        if len(digits_text) >= self.min_digit_count:
                            candidates.append(NIDCandidate(
                                text=digits_text[:14],
                                confidence=0.6,
                                source="contour_scan",
                                digit_count=len(digits_text[:14]),
                                is_valid_format=_is_valid_nid_format(digits_text[:14]) if len(digits_text) >= 14 else False
                            ))

        return candidates

    def _get_preprocessing_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Get different preprocessing variants of the image."""
        variants = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Upscale
        h, w = gray.shape
        target_h = max(80, h)
        scale = target_h / h
        upscaled = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

        # 1. Standard Otsu
        _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", otsu))

        # 2. Inverted Otsu
        _, otsu_inv = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(("otsu_inv", otsu_inv))

        # 3. CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(upscaled)
        _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("clahe_otsu", clahe_otsu))

        # 4. Adaptive Gaussian
        adaptive = cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        variants.append(("adaptive", adaptive))

        return variants

    def _find_digit_regions(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find potential digit regions in binary image."""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_regions = []
        h, w = binary_image.shape
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Filter for digit-like proportions
            aspect_ratio = cw / ch if ch > 0 else 0
            area = cv2.contourArea(cnt)
            
            # Typical digit: tall and narrow, reasonable area
            # Widened range: digit '1' has aspect ratio ~0.15-0.25
            if 0.15 <= aspect_ratio <= 1.0 and 30 <= area <= 3000:
                digit_regions.append((x, y, cw, ch))

        return digit_regions

    def _extract_digits_from_regions(self, image: np.ndarray, 
                                     regions: List[Tuple[int, int, int, int]]) -> str:
        """Extract digits from identified regions using OCR on the full line."""
        if not self.recognize_func or not regions:
            return ""
        
        # Crop the ENTIRE line (leftmost to rightmost digit) and OCR once
        # This is ~14x faster than per-digit OCR
        x_min = min(r[0] for r in regions)
        x_max = max(r[0] + r[2] for r in regions)
        y_min = min(r[1] for r in regions)
        y_max = max(r[1] + r[3] for r in regions)
        
        pad = 4
        y1 = max(0, y_min - pad)
        y2 = min(image.shape[0], y_max + pad)
        x1 = max(0, x_min - pad)
        x2 = min(image.shape[1], x_max + pad)
        
        line_crop = image[y1:y2, x1:x2]
        if line_crop.size == 0:
            return ""
        
        # Convert to BGR if grayscale (for OCR compatibility)
        if len(line_crop.shape) == 2:
            line_crop = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2BGR)
        
        text = self.recognize_func(line_crop)
        digits = re.sub(r'\D', '', text)
        digits = _normalize_digits(digits)
        digits = _fix_common_digit_ocr_errors(digits)
        
        return digits

    def _group_into_lines(self, contours: List[Tuple[int, int, int, int]], 
                          y_threshold: int = 10) -> List[List[Tuple[int, int, int, int]]]:
        """Group contours into horizontal lines based on y-coordinate."""
        if not contours:
            return []

        lines = []
        current_line = [contours[0]]
        current_y = contours[0][1]

        for contour in contours[1:]:
            x, y, w, h = contour
            
            if abs(y - current_y) <= y_threshold:
                # Same line
                current_line.append(contour)
            else:
                # New line
                if current_line:
                    lines.append(current_line)
                current_line = [contour]
                current_y = y

        if current_line:
            lines.append(current_line)

        return lines

    def extract_from_multiple_sources(
        self,
        front_image: Optional[np.ndarray] = None,
        back_image: Optional[np.ndarray] = None,
        front_nid_crop: Optional[np.ndarray] = None,
        back_nid_crop: Optional[np.ndarray] = None,
        recognize_func: Optional[callable] = None
    ) -> NIDExtractionResult:
        """
        Extract NID from multiple sources (front and/or back images) with cross-validation.
        
        Priority:
        1. front_nid field (highest priority)
        2. back_nid field (fallback)
        3. Cross-validation when both available
        
        Args:
            front_image: Front side image
            back_image: Back side image
            front_nid_crop: Pre-cropped NID field from front
            back_nid_crop: Pre-cropped NID field from back
            recognize_func: Function to recognize text in a crop
            
        Returns:
            NIDExtractionResult with cross-validation info
        """
        self.recognize_func = recognize_func
        result = NIDExtractionResult(nid="", confidence=0.0, source="none")
        
        front_nid = ""
        front_conf = 0.0
        back_nid = ""
        back_conf = 0.0
        
        # Extract from front side
        if front_image is not None:
            logger.debug("NID extractor: Processing front side")
            front_nid, front_conf = self.extract(
                front_image, 
                nid_crop=front_nid_crop,
                recognize_func=recognize_func
            )
            result.front_nid = front_nid
        
        # Extract from back side
        if back_image is not None:
            logger.debug("NID extractor: Processing back side")
            back_nid, back_conf = self.extract(
                back_image,
                nid_crop=back_nid_crop,
                recognize_func=recognize_func
            )
            result.back_nid = back_nid
        
        # Cross-validation and selection
        result.nid, result.confidence, result.source, result.cross_validation = \
            self._merge_and_validate_nid(front_nid, front_conf, back_nid, back_conf)
        
        # Add warnings for mismatches
        if result.source == "front_mismatch":
            result.warnings.append(
                f"NID mismatch between front ({front_nid}) and back ({back_nid}). Using front."
            )
        
        logger.info(
            f"NID multi-source extraction: nid={result.nid}, source={result.source}, "
            f"confidence={result.confidence:.2f}"
        )
        
        return result
    
    def _merge_and_validate_nid(
        self,
        front_nid: str,
        front_conf: float,
        back_nid: str,
        back_conf: float
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Merge and validate NID from front and back sources.
        
        Args:
            front_nid: NID extracted from front side
            front_conf: Confidence of front NID
            back_nid: NID extracted from back side
            back_conf: Confidence of back NID
            
        Returns:
            Tuple of (final_nid, confidence, source, cross_validation_info)
        """
        cross_val = {
            "front_nid": front_nid,
            "back_nid": back_nid,
            "front_confidence": front_conf,
            "back_confidence": back_conf,
            "match_status": "none"
        }
        
        # Case 1: Only front NID available
        if front_nid and not back_nid:
            conf = front_conf
            if _is_valid_nid_format(front_nid):
                conf = max(conf, 0.85)
            cross_val["match_status"] = "front_only"
            logger.info(f"NID from front only: {front_nid} (conf={conf:.2f})")
            return front_nid, conf, "front", cross_val
        
        # Case 2: Only back NID available
        if back_nid and not front_nid:
            conf = back_conf
            if _is_valid_nid_format(back_nid):
                conf = max(conf, 0.85)
            cross_val["match_status"] = "back_only"
            logger.info(f"NID from back only: {back_nid} (conf={conf:.2f})")
            return back_nid, conf, "back", cross_val
        
        # Case 3: Both NIDs available
        if front_nid and back_nid:
            if front_nid == back_nid:
                # NIDs match - use higher confidence or average, boost for match
                avg_conf = (front_conf + back_conf) / 2
                max_conf = max(front_conf, back_conf)
                conf = max(avg_conf, max_conf)
                
                if _is_valid_nid_format(front_nid):
                    conf = max(conf, 0.95)  # High confidence for matched valid NID
                
                cross_val["match_status"] = "match"
                logger.info(
                    f"NID match confirmed: {front_nid} (front_conf={front_conf:.2f}, "
                    f"back_conf={back_conf:.2f}, final_conf={conf:.2f})"
                )
                return front_nid, conf, "both_matched", cross_val
            else:
                # NIDs don't match - prefer front with reduced confidence
                conf = front_conf * 0.8  # Reduce confidence for mismatch
                if _is_valid_nid_format(front_nid):
                    conf = max(conf, 0.7)
                
                cross_val["match_status"] = "mismatch"
                logger.warning(
                    f"NID mismatch: front={front_nid} (conf={front_conf:.2f}), "
                    f"back={back_nid} (conf={back_conf:.2f}). Using front."
                )
                return front_nid, conf, "front_mismatch", cross_val
        
        # Case 4: No NID available
        cross_val["match_status"] = "none"
        return "", 0.0, "none", cross_val
    
    def _select_best_candidate(self, candidates: List[NIDCandidate]) -> Tuple[str, float]:
        """Select the best NID candidate."""
        if not candidates:
            return "", 0.0

        # Sort by priority:
        # 1. Valid format with most digits
        # 2. Most digits
        # 3. Highest confidence

        def score_candidate(c: NIDCandidate) -> Tuple:
            return (
                c.is_valid_format,  # Valid format first
                c.digit_count,       # More digits better
                c.confidence         # Higher confidence better
            )

        candidates.sort(key=score_candidate, reverse=True)
        best = candidates[0]

        # Apply century digit fix
        fixed_text = _fix_nid_century_digit(best.text)

        # Calculate final confidence
        final_conf = best.confidence
        if best.is_valid_format:
            final_conf = max(final_conf, 0.85)
        elif best.digit_count == 14:
            final_conf = max(final_conf, 0.7)
        elif best.digit_count >= 10:
            final_conf = max(final_conf, 0.5)

        logger.info(f"NID extractor: Selected '{fixed_text}' from {best.source} "
                   f"(digits={best.digit_count}, valid={best.is_valid_format}, conf={final_conf:.2f})")

        return fixed_text, final_conf


# Singleton instance
_extractor = None


def get_nid_extractor() -> NIDExtractor:
    """Get or create NID extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = NIDExtractor()
    return _extractor
