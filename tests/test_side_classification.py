"""
Tests for Egyptian ID Card Side Classification and Dual-Side Processing

Tests cover:
- Side classifier (front/back/both detection)
- Dual-side processor (image splitting and merging)
- NID cross-validation between sides
- Multi-image pipeline
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.services.side_classifier import (
    SideClassifier, 
    CardSide, 
    SideClassification,
    get_side_classifier,
    classify_card_side
)
from app.services.dual_side_processor import (
    DualSideProcessor,
    DualSideResult,
    get_dual_side_processor
)
from app.models.nid_extractor import (
    NIDExtractor,
    NIDExtractionResult,
    get_nid_extractor
)


# ─────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_front_image():
    """Create a mock front side image (with photo region)."""
    # Create a synthetic image resembling front side
    img = np.ones((400, 800, 3), dtype=np.uint8) * 200
    
    # Add photo region (left side, circular)
    cv2.circle(img, (120, 200), 80, (150, 150, 150), -1)
    
    # Add text-like regions (right side)
    cv2.rectangle(img, (300, 50), (750, 150), (100, 100, 100), -1)
    cv2.rectangle(img, (300, 200), (750, 280), (100, 100, 100), -1)
    
    return img


@pytest.fixture
def sample_back_image():
    """Create a mock back side image (no photo, more text)."""
    # Create a synthetic image resembling back side
    img = np.ones((400, 800, 3), dtype=np.uint8) * 200
    
    # Add text-like regions (more dense than front)
    cv2.rectangle(img, (50, 30), (750, 100), (100, 100, 100), -1)
    cv2.rectangle(img, (50, 120), (750, 190), (100, 100, 100), -1)
    cv2.rectangle(img, (50, 250), (750, 320), (100, 100, 100), -1)
    
    return img


@pytest.fixture
def sample_dual_side_image():
    """Create a mock dual-side image (both sides horizontally)."""
    # Create a wide image with two card-like regions
    img = np.ones((400, 1600, 3), dtype=np.uint8) * 200
    
    # Left side (front)
    cv2.circle(img, (120, 200), 80, (150, 150, 150), -1)
    cv2.rectangle(img, (300, 50), (750, 150), (100, 100, 100), -1)
    
    # Right side (back)
    cv2.rectangle(img, (850, 30), (1550, 100), (100, 100, 100), -1)
    cv2.rectangle(img, (850, 120), (1550, 190), (100, 100, 100), -1)
    
    return img


@pytest.fixture
def classifier():
    """Get side classifier instance."""
    return get_side_classifier()


@pytest.fixture
def nid_extractor():
    """Get NID extractor instance."""
    return get_nid_extractor()


# ─────────────────────────────────────────────────────────
# Side Classifier Tests
# ─────────────────────────────────────────────────────────


class TestSideClassifier:
    """Tests for SideClassifier."""
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert hasattr(classifier, 'classify')
        assert hasattr(classifier, 'split_dual_side_image')
    
    def test_classify_front_side(self, classifier, sample_front_image):
        """Test classification of front side image."""
        result = classifier.classify(sample_front_image)
        
        assert isinstance(result, SideClassification)
        assert result.side in [CardSide.FRONT, CardSide.UNKNOWN]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.details, dict)
    
    def test_classify_back_side(self, classifier, sample_back_image):
        """Test classification of back side image."""
        result = classifier.classify(sample_back_image)
        
        assert isinstance(result, SideClassification)
        assert result.side in [CardSide.BACK, CardSide.UNKNOWN]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_dual_side(self, classifier, sample_dual_side_image):
        """Test classification of dual-side image."""
        result = classifier.classify(sample_dual_side_image)
        
        assert isinstance(result, SideClassification)
        # Dual-side should be detected due to aspect ratio
        assert result.side == CardSide.BOTH
        assert result.confidence > 0.7
        assert "split_orientation" in result.details
    
    def test_classify_empty_image(self, classifier):
        """Test classification of empty/invalid image."""
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = classifier.classify(empty_img)
        
        assert result.side == CardSide.UNKNOWN
        assert result.confidence == 0.0
    
    def test_classify_none_image(self, classifier):
        """Test classification of None image."""
        result = classifier.classify(None)
        
        assert result.side == CardSide.UNKNOWN
        assert "error" in result.details
    
    def test_split_dual_side_horizontal(self, classifier, sample_dual_side_image):
        """Test splitting dual-side image (horizontal)."""
        first, second = classifier.split_dual_side_image(
            sample_dual_side_image, 
            orientation="horizontal"
        )
        
        # Both halves should have similar height
        assert first.shape[0] == second.shape[0]
        # Each half should be roughly half the width (with overlap)
        assert first.shape[1] > sample_dual_side_image.shape[1] // 2
        assert second.shape[1] > sample_dual_side_image.shape[1] // 2
    
    def test_split_dual_side_vertical(self, classifier, sample_dual_side_image):
        """Test splitting dual-side image (vertical)."""
        first, second = classifier.split_dual_side_image(
            sample_dual_side_image,
            orientation="vertical"
        )
        
        # Both halves should have similar width
        assert first.shape[1] == second.shape[1]
        # Each half should be roughly half the height (with overlap)
        assert first.shape[0] > sample_dual_side_image.shape[0] // 2
    
    def test_detect_two_card_contours(self, classifier, sample_dual_side_image):
        """Test detection of two card contours."""
        detected, orientation = classifier._detect_two_card_contours(sample_dual_side_image)
        
        # Should detect two cards in horizontal arrangement
        assert detected is True
        assert orientation == "horizontal"
    
    def test_photo_detection(self, classifier, sample_front_image):
        """Test photo region detection."""
        # Extract left region where photo should be
        h, w = sample_front_image.shape[:2]
        photo_region = sample_front_image[:, :int(w * 0.30)]
        
        score = classifier._detect_photo_region(photo_region)
        assert score > 0.5  # Should detect the circular photo region
    
    def test_text_density_analysis(self, classifier, sample_front_image, sample_back_image):
        """Test text density analysis."""
        gray_front = cv2.cvtColor(sample_front_image, cv2.COLOR_BGR2GRAY)
        gray_back = cv2.cvtColor(sample_back_image, cv2.COLOR_BGR2GRAY)
        
        front_density = classifier._analyze_text_density(gray_front)
        back_density = classifier._analyze_text_density(gray_back)
        
        # Back side should have higher text density
        assert 0.0 <= front_density <= 1.0
        assert 0.0 <= back_density <= 1.0


# ─────────────────────────────────────────────────────────
# NID Extractor Tests
# ─────────────────────────────────────────────────────────


class TestNIDExtractor:
    """Tests for NIDExtractor with multi-source support."""
    
    def test_extractor_initialization(self, nid_extractor):
        """Test NID extractor initializes correctly."""
        assert nid_extractor is not None
        assert hasattr(nid_extractor, 'extract')
        assert hasattr(nid_extractor, 'extract_from_multiple_sources')
    
    def test_merge_and_validate_nid_both_match(self, nid_extractor):
        """Test NID merging when both sides match."""
        front_nid = "29901011234567"
        back_nid = "29901011234567"
        front_conf = 0.85
        back_conf = 0.80
        
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            front_nid, front_conf, back_nid, back_conf
        )
        
        assert nid == front_nid
        assert conf >= 0.95  # Boosted for match
        assert source == "both_matched"
        assert cross_val["match_status"] == "match"
    
    def test_merge_and_validate_nid_mismatch(self, nid_extractor):
        """Test NID merging when sides don't match."""
        front_nid = "29901011234567"
        back_nid = "29901019876543"
        front_conf = 0.85
        back_conf = 0.80
        
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            front_nid, front_conf, back_nid, back_conf
        )
        
        assert nid == front_nid  # Front takes priority
        assert conf < front_conf  # Reduced for mismatch
        assert source == "front_mismatch"
        assert cross_val["match_status"] == "mismatch"
    
    def test_merge_and_validate_nid_front_only(self, nid_extractor):
        """Test NID merging when only front available."""
        front_nid = "29901011234567"
        back_nid = ""
        front_conf = 0.85
        back_conf = 0.0
        
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            front_nid, front_conf, back_nid, back_conf
        )
        
        assert nid == front_nid
        assert conf >= 0.85
        assert source == "front"
        assert cross_val["match_status"] == "front_only"
    
    def test_merge_and_validate_nid_back_only(self, nid_extractor):
        """Test NID merging when only back available."""
        front_nid = ""
        back_nid = "29901011234567"
        front_conf = 0.0
        back_conf = 0.85
        
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            front_nid, front_conf, back_nid, back_conf
        )
        
        assert nid == back_nid
        assert conf >= 0.85
        assert source == "back"
        assert cross_val["match_status"] == "back_only"
    
    def test_merge_and_validate_nid_none(self, nid_extractor):
        """Test NID merging when neither available."""
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            "", 0.0, "", 0.0
        )
        
        assert nid == ""
        assert conf == 0.0
        assert source == "none"


# ─────────────────────────────────────────────────────────
# Dual-Side Processor Tests
# ─────────────────────────────────────────────────────────


class TestDualSideProcessor:
    """Tests for DualSideProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create dual-side processor with mocked dependencies."""
        mock_detector = Mock()
        mock_ocr = Mock()
        return DualSideProcessor(detector=mock_detector, ocr_engine=mock_ocr)
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor is not None
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'FRONT_PRIORITY_FIELDS')
        assert hasattr(processor, 'BACK_PRIORITY_FIELDS')
    
    def test_process_dual_side_image(self, processor, sample_dual_side_image):
        """Test processing dual-side image."""
        result = processor.process(sample_dual_side_image)
        
        assert isinstance(result, DualSideResult)
        assert hasattr(result, 'extracted')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_ms')
        assert hasattr(result, 'split_info')
        assert hasattr(result, 'cross_validation')
    
    def test_cross_validate_nid(self, processor):
        """Test NID cross-validation in processor."""
        front_data = {"front_nid": "29901011234567", "nid": "29901011234567"}
        back_data = {"back_nid": "29901011234567", "nid": "29901011234567"}
        front_conf = {"front_nid": 0.85, "nid": 0.85}
        back_conf = {"back_nid": 0.80, "nid": 0.80}
        
        result = processor._cross_validate_nid(front_data, back_data, front_conf, back_conf)
        
        assert result["nid"] == "29901011234567"
        assert result["confidence"] >= 0.95
        assert result["match"] == "match"
    
    def test_merge_address(self, processor):
        """Test address merging from both sides."""
        front_data = {"address": "القاهرة، مصر"}
        back_data = {"add_line_1": "شارع التحرير", "add_line_2": "وسط البلد"}
        
        result = processor._merge_address(front_data, back_data)
        
        assert "address" in result
        assert result["address"] != ""
    
    def test_merge_results_priority(self, processor):
        """Test field priority in result merging."""
        front_result = {
            "extracted": {
                "firstName": "محمد",
                "lastName": "عبدالله",
                "nid": "29901011234567"
            },
            "confidence": {"per_field": {"firstName": 0.9, "lastName": 0.85, "nid": 0.95}}
        }
        back_result = {
            "extracted": {
                "add_line_1": "شارع التحرير",
                "issue_date": "01/01/2020",
                "expiry_date": "01/01/2030"
            },
            "confidence": {"per_field": {"add_line_1": 0.8, "issue_date": 0.85, "expiry_date": 0.85}}
        }
        
        merged, cross_val = processor._merge_results(front_result, back_result)
        
        # Front priority fields should be present
        assert "firstName" in merged["extracted"]
        assert "lastName" in merged["extracted"]
        
        # Back priority fields should be present
        assert "add_line_1" in merged["extracted"]
        assert "issue_date" in merged["extracted"]
        assert "expiry_date" in merged["extracted"]
        
        # NID cross-validation info should be present
        assert "nid_match" in cross_val


# ─────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_side_classifier_singleton(self):
        """Test side classifier singleton pattern."""
        c1 = get_side_classifier()
        c2 = get_side_classifier()
        assert c1 is c2
    
    def test_nid_extractor_singleton(self):
        """Test NID extractor singleton pattern."""
        e1 = get_nid_extractor()
        e2 = get_nid_extractor()
        assert e1 is e2
    
    def test_classify_card_side_convenience_function(self, sample_front_image):
        """Test convenience function for side classification."""
        result = classify_card_side(sample_front_image)
        assert isinstance(result, SideClassification)


# ─────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_small_image(self, classifier):
        """Test classification of very small image."""
        small_img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result = classifier.classify(small_img)
        
        assert result.side in [CardSide.UNKNOWN, CardSide.FRONT, CardSide.BACK]
    
    def test_very_large_aspect_ratio(self, classifier):
        """Test classification of image with extreme aspect ratio."""
        # Very wide image
        wide_img = np.ones((200, 2000, 3), dtype=np.uint8) * 200
        result = classifier.classify(wide_img)
        
        assert result.side == CardSide.BOTH
        assert result.details.get("split_orientation") == "horizontal"
    
    def test_square_image(self, classifier):
        """Test classification of square image."""
        square_img = np.ones((500, 500, 3), dtype=np.uint8) * 200
        result = classifier.classify(square_img)
        
        # Square images might be classified as dual-side vertical
        assert result.side in [CardSide.UNKNOWN, CardSide.BOTH]
    
    def test_noisy_image(self, classifier):
        """Test classification of noisy image."""
        noisy_img = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        result = classifier.classify(noisy_img)
        
        # Should still return a valid classification
        assert isinstance(result, SideClassification)
        assert result.side in [CardSide.FRONT, CardSide.BACK, CardSide.BOTH, CardSide.UNKNOWN]
    
    def test_nid_with_invalid_format(self, nid_extractor):
        """Test NID validation with invalid format."""
        front_nid = "12345"  # Too short
        back_nid = "invalid"  # Not digits
        
        nid, conf, source, cross_val = nid_extractor._merge_and_validate_nid(
            front_nid, 0.5, back_nid, 0.3
        )
        
        # Should still return something (front takes priority)
        assert nid == front_nid or nid == ""


# ─────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────


class TestPerformance:
    """Performance tests for timing requirements."""
    
    def test_classification_speed(self, classifier, sample_front_image):
        """Test that classification completes quickly."""
        import time
        
        start = time.time()
        for _ in range(10):
            classifier.classify(sample_front_image)
        elapsed = time.time() - start
        
        # Each classification should take less than 100ms on average
        assert elapsed / 10 < 0.1
    
    def test_nid_merge_speed(self, nid_extractor):
        """Test that NID merging is fast."""
        import time
        
        start = time.time()
        for _ in range(100):
            nid_extractor._merge_and_validate_nid(
                "29901011234567", 0.9,
                "29901011234567", 0.85
            )
        elapsed = time.time() - start
        
        # Each merge should take less than 1ms on average
        assert elapsed / 100 < 0.001
