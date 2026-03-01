"""
Test suite for Egyptian ID OCR
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from app.models.id_parser import (
    parse_national_id,
    validate_national_id,
    format_national_id,
)
from app.utils.image_utils import (
    decode_image,
    resize_to_standard,
    assess_quality,
    extract_roi,
)
from app.utils.text_utils import clean_field


# ─────────────────────────────────────────────────────────────────────────────
# ID Parser Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIDParser:
    """Tests for the National ID parser."""

    def test_parse_valid_id_male(self):
        """Test parsing valid male national ID."""
        # 29901011234567: 2=1900s, 99=1999, 01=Jan, 01=01, 01=Cairo, 0=male
        result = parse_national_id("29901011234567")

        assert result.valid is True
        assert result.birth_date == "01/01/1999"
        assert result.governorate == "القاهرة"
        assert result.gender == "ذكر"
        assert result.age == 27  # Assuming 2026
        assert result.sequence == "2345"

    def test_parse_valid_id_female(self):
        """Test parsing valid female national ID."""
        # 30101021234567: 3=2000s, 01=2001, 01=Jan, 01=01, 02=female
        result = parse_national_id("30101021234567")

        assert result.valid is True
        assert result.birth_date == "01/01/2001"
        assert result.gender == "أنثى"

    def test_parse_id_with_non_digits(self):
        """Test parsing ID with spaces and dashes."""
        result = parse_national_id("299 0101 12345 67")

        assert result.valid is True
        assert result.raw == "29901011234567"

    def test_parse_invalid_length(self):
        """Test parsing ID with invalid length."""
        result = parse_national_id("12345")

        assert result.valid is False
        assert "length" in result.error.lower()

    def test_parse_invalid_century(self):
        """Test parsing ID with invalid century code."""
        result = parse_national_id("19901011234567")  # 1 is invalid

        assert result.valid is False
        assert "century" in result.error.lower()

    def test_parse_invalid_month(self):
        """Test parsing ID with invalid month."""
        result = parse_national_id("29913011234567")  # 13 is invalid

        assert result.valid is False
        assert "month" in result.error.lower()

    def test_validate_national_id(self):
        """Test validation function."""
        assert validate_national_id("29901011234567") is True
        assert validate_national_id("12345") is False

    def test_format_national_id(self):
        """Test ID formatting."""
        formatted = format_national_id("29901011234567")
        assert formatted == "299 0101 12345 67"


# ─────────────────────────────────────────────────────────────────────────────
# Image Utils Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestImageUtils:
    """Tests for image processing utilities."""

    def test_resize_to_standard(self):
        """Test image resizing."""
        # Create test image
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Resize
        resized = resize_to_standard(img, target_width=400)

        # Should maintain aspect ratio
        assert resized.shape[1] == 400  # width
        assert resized.shape[0] == 200  # height = 100 * (400/200)

    def test_assess_quality(self):
        """Test image quality assessment."""
        # Create a clear image
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

        quality = assess_quality(img)

        assert "overall" in quality
        assert "sharpness" in quality
        assert "brightness" in quality
        assert "contrast" in quality
        assert "acceptable" in quality
        assert 0 <= quality["overall"] <= 1

    def test_extract_roi(self):
        """Test ROI extraction."""
        # Create test image
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Extract ROI at 10%, 10%, 50%, 30%
        roi = extract_roi(img, (0.1, 0.1, 0.5, 0.3))

        # Should have correct dimensions
        assert roi.shape[0] == 30  # 100 * 0.3
        assert roi.shape[1] == 100  # 200 * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Text Utils Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTextUtils:
    """Tests for text processing utilities."""

    def test_clean_id_number(self):
        """Test cleaning ID number field."""
        text = "2990101 12345 67"
        cleaned = clean_field(text, "id_number")

        assert cleaned == "29901011234567"

    def test_clean_id_with_letters(self):
        """Test cleaning ID with OCR mistakes."""
        text = "2990I0I1234567"  # I instead of 1
        cleaned = clean_field(text, "id_number")

        # Should fix common OCR mistakes
        assert "I" not in cleaned

    def test_clean_arabic_text(self):
        """Test cleaning Arabic text."""
        text = "محمد@@@كمال  عبدالله"
        cleaned = clean_field(text, "name_ar")

        # Should keep only Arabic characters
        assert "@" not in cleaned
        assert cleaned.strip() == "محمد كمال عبدالله"

    def test_clean_english_text(self):
        """Test cleaning English text."""
        text = "mohamed 123 Kamal"
        cleaned = clean_field(text, "name_en")

        # Should keep only letters and uppercase
        assert cleaned == "MOHAMED KAMAL"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes for testing."""
        # Create a simple test image
        img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", img)
        return buffer.tobytes()

    def test_pipeline_processes_image(self, sample_image_bytes):
        """Test that pipeline can process an image."""
        from app.services.pipeline import IDExtractionPipeline

        pipeline = IDExtractionPipeline()

        # Process should not crash
        result = pipeline.process(sample_image_bytes)

        # Should return a dict with expected keys
        assert isinstance(result, dict)

    def test_invalid_image_error(self):
        """Test handling of invalid image."""
        from app.services.pipeline import IDExtractionPipeline

        pipeline = IDExtractionPipeline()

        # Invalid bytes
        result = pipeline.process(b"not an image")

        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
