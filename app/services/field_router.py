"""
Field Router Service for Egyptian ID Card OCR

Provides:
- Mapping between ONNX field classes and internal field names
- Side classification for each field (front/back)
- Validation rules for each field type
- Field aliases and canonical name resolution
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, Any
from enum import Enum

from app.core.logger import logger
from app.core.config import settings


class FieldSide(str, Enum):
    """Enum for field side classification."""
    FRONT = "front"
    BACK = "back"
    BOTH = "both"  # Fields that can appear on both sides


@dataclass
class FieldDefinition:
    """Definition of a field with metadata."""
    
    # ONNX class ID (0-16)
    onnx_class_id: int
    
    # Internal canonical name
    canonical_name: str
    
    # Which side this field appears on
    side: FieldSide
    
    # Field description
    description: str
    
    # Validation regex pattern (optional)
    validation_pattern: Optional[str] = None
    
    # Expected character type
    char_type: str = "mixed"  # "arabic", "latin", "digits", "mixed"
    
    # Min/max length constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # Is this a critical field (required for valid ID)
    is_critical: bool = False
    
    # Alternative names/aliases
    aliases: Set[str] = field(default_factory=set)
    
    # OCR-specific configuration hints
    ocr_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of field validation."""
    
    is_valid: bool
    field_name: str
    value: str
    error_message: Optional[str] = None
    warnings: list = field(default_factory=list)


class FieldRouter:
    """
    Routes and validates fields detected by the ONNX model.
    
    Responsibilities:
    1. Map ONNX class IDs to internal field names
    2. Classify fields by side (front/back)
    3. Validate field values against rules
    4. Provide field metadata for processing
    """
    
    # ONNX class ID to field definition mapping
    # Based on weights/field_detector.onnx model output
    FIELD_DEFINITIONS: Dict[int, FieldDefinition] = {
        0: FieldDefinition(
            onnx_class_id=0,
            canonical_name="firstName",
            side=FieldSide.FRONT,
            description="First name (given name) in Arabic",
            char_type="arabic",
            min_length=2,
            max_length=50,
            is_critical=True,
            aliases={"first_name", "fname", "given_name"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        1: FieldDefinition(
            onnx_class_id=1,
            canonical_name="lastName",
            side=FieldSide.FRONT,
            description="Last name (family name) in Arabic",
            char_type="arabic",
            min_length=2,
            max_length=50,
            is_critical=True,
            aliases={"last_name", "lname", "family_name", "surname"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        2: FieldDefinition(
            onnx_class_id=2,
            canonical_name="add_line_1",
            side=FieldSide.BACK,
            description="Address line 1 (detailed address)",
            char_type="arabic",
            min_length=5,
            max_length=100,
            aliases={"address_line_1", "address1"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        3: FieldDefinition(
            onnx_class_id=3,
            canonical_name="add_line_2",
            side=FieldSide.BACK,
            description="Address line 2 (additional address details)",
            char_type="arabic",
            min_length=5,
            max_length=100,
            aliases={"address_line_2", "address2"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        4: FieldDefinition(
            onnx_class_id=4,
            canonical_name="nid",
            side=FieldSide.FRONT,
            description="National ID number (14 digits) on front side",
            validation_pattern=r"^\d{14}$",
            char_type="digits",
            min_length=14,
            max_length=14,
            is_critical=True,
            aliases={"front_nid", "id_number", "national_id", "nid_front"},
            ocr_hints={"engine": "easyocr", "digits_only": True}
        ),
        5: FieldDefinition(
            onnx_class_id=5,
            canonical_name="nid",
            side=FieldSide.BACK,
            description="National ID number (14 digits) on back side",
            validation_pattern=r"^\d{14}$",
            char_type="digits",
            min_length=14,
            max_length=14,
            is_critical=True,
            aliases={"back_nid", "id_number", "national_id", "nid_back"},
            ocr_hints={"engine": "easyocr", "digits_only": True}
        ),
        6: FieldDefinition(
            onnx_class_id=6,
            canonical_name="serial",
            side=FieldSide.FRONT,
            description="Serial number of the ID card",
            char_type="latin",
            min_length=4,
            max_length=20,
            aliases={"serial_num", "serial_number", "card_serial"},
            ocr_hints={"engine": "easyocr", "lang": "en"}
        ),
        7: FieldDefinition(
            onnx_class_id=7,
            canonical_name="issue_code",
            side=FieldSide.BACK,
            description="Issue code/date information",
            char_type="mixed",
            min_length=4,
            max_length=20,
            aliases={"issue_date", "issue"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        8: FieldDefinition(
            onnx_class_id=8,
            canonical_name="expiry_date",
            side=FieldSide.BACK,
            description="Expiry date of the ID card",
            char_type="mixed",
            min_length=4,
            max_length=20,
            aliases={"expiry", "expiration_date", "valid_until"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        9: FieldDefinition(
            onnx_class_id=9,
            canonical_name="job_title",
            side=FieldSide.FRONT,
            description="Profession/job title",
            char_type="arabic",
            min_length=2,
            max_length=50,
            aliases={"profession", "occupation", "job"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        10: FieldDefinition(
            onnx_class_id=10,
            canonical_name="gender",
            side=FieldSide.FRONT,
            description="Gender (ذكر/أنثى)",
            char_type="arabic",
            min_length=4,
            max_length=10,
            aliases={"sex"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        11: FieldDefinition(
            onnx_class_id=11,
            canonical_name="religion",
            side=FieldSide.FRONT,
            description="Religion (مسلم/مسيحي)",
            char_type="arabic",
            min_length=4,
            max_length=20,
            aliases={"faith"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        12: FieldDefinition(
            onnx_class_id=12,
            canonical_name="marital_status",
            side=FieldSide.FRONT,
            description="Marital status (أعزب/متزوج/مطلق/أرمل)",
            char_type="arabic",
            min_length=4,
            max_length=20,
            aliases={"marriage_status"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        13: FieldDefinition(
            onnx_class_id=13,
            canonical_name="photo",
            side=FieldSide.FRONT,
            description="Face photo region",
            char_type="image",
            is_critical=False,
            aliases={"face", "face_photo", "portrait"},
            ocr_hints={"skip_ocr": True}
        ),
        14: FieldDefinition(
            onnx_class_id=14,
            canonical_name="front_logo",
            side=FieldSide.FRONT,
            description="Front logo/emblem region",
            char_type="image",
            is_critical=False,
            aliases={"logo", "emblem", "coat_of_arms"},
            ocr_hints={"skip_ocr": True}
        ),
        15: FieldDefinition(
            onnx_class_id=15,
            canonical_name="address",
            side=FieldSide.FRONT,
            description="Short address on front side",
            char_type="arabic",
            min_length=5,
            max_length=100,
            aliases={"short_address"},
            ocr_hints={"engine": "paddle", "lang": "ar"}
        ),
        16: FieldDefinition(
            onnx_class_id=16,
            canonical_name="dob",
            side=FieldSide.FRONT,
            description="Date of birth",
            char_type="digits",
            min_length=8,
            max_length=10,
            is_critical=True,
            aliases={"birth_date", "date_of_birth"},
            ocr_hints={"engine": "easyocr", "digits_only": True}
        ),
    }
    
    # Front side field class IDs
    FRONT_FIELD_IDS: Set[int] = {0, 1, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16}
    
    # Back side field class IDs
    BACK_FIELD_IDS: Set[int] = {2, 3, 5, 7, 8}
    
    # Fields that can appear on both sides (NID)
    BOTH_SIDE_FIELD_IDS: Set[int] = {4, 5}  # front_nid and back_nid both map to "nid"
    
    # Lookup tables for quick access
    _onnx_to_canonical: Dict[int, str] = {}
    _canonical_to_onnx: Dict[str, int] = {}
    _alias_to_onnx: Dict[str, int] = {}
    _side_to_field_ids: Dict[str, Set[int]] = {}
    
    def __init__(self):
        """Initialize the field router with lookup tables."""
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for efficient field resolution."""
        self._onnx_to_canonical = {}
        self._canonical_to_onnx = {}
        self._alias_to_onnx = {}
        self._side_to_field_ids = {
            "front": set(),
            "back": set(),
            "both": set()
        }
        
        for onnx_id, definition in self.FIELD_DEFINITIONS.items():
            # ONNX ID to canonical name
            self._onnx_to_canonical[onnx_id] = definition.canonical_name
            
            # Canonical name to ONNX ID (first occurrence wins for overlapping fields)
            if definition.canonical_name not in self._canonical_to_onnx:
                self._canonical_to_onnx[definition.canonical_name] = onnx_id
            
            # Aliases to ONNX ID
            for alias in definition.aliases:
                self._alias_to_onnx[alias.lower()] = onnx_id
            
            # Side to field IDs
            self._side_to_field_ids[definition.side.value].add(onnx_id)
        
        logger.debug(f"Field router initialized with {len(self.FIELD_DEFINITIONS)} field definitions")
    
    def get_field_definition(self, onnx_class_id: int) -> Optional[FieldDefinition]:
        """
        Get field definition by ONNX class ID.
        
        Args:
            onnx_class_id: The ONNX model class ID (0-16)
            
        Returns:
            FieldDefinition if found, None otherwise
        """
        return self.FIELD_DEFINITIONS.get(onnx_class_id)
    
    def get_canonical_name(self, onnx_class_id: int) -> Optional[str]:
        """
        Get canonical field name from ONNX class ID.
        
        Args:
            onnx_class_id: The ONNX model class ID
            
        Returns:
            Canonical field name or None if not found
        """
        return self._onnx_to_canonical.get(onnx_class_id)
    
    def get_onnx_class_id(self, field_name: str) -> Optional[int]:
        """
        Get ONNX class ID from field name or alias.
        
        Args:
            field_name: Field name or alias
            
        Returns:
            ONNX class ID or None if not found
        """
        # Try direct lookup
        if field_name in self._canonical_to_onnx:
            return self._canonical_to_onnx[field_name]
        
        # Try alias lookup
        return self._alias_to_onnx.get(field_name.lower())
    
    def get_field_side(self, onnx_class_id: int) -> FieldSide:
        """
        Get the side classification for a field.
        
        Args:
            onnx_class_id: The ONNX model class ID
            
        Returns:
            FieldSide enum value
        """
        definition = self.FIELD_DEFINITIONS.get(onnx_class_id)
        if definition:
            return definition.side
        return FieldSide.FRONT  # Default to front for unknown fields
    
    def get_fields_for_side(self, side: str) -> Set[int]:
        """
        Get all ONNX field IDs that belong to a specific side.
        
        Args:
            side: "front", "back", or "both"
            
        Returns:
            Set of ONNX class IDs for that side
        """
        return self._side_to_field_ids.get(side, set())
    
    def is_field_expected_for_side(self, onnx_class_id: int, side: str) -> bool:
        """
        Check if a field is expected to appear on a given side.
        
        Args:
            onnx_class_id: The ONNX model class ID
            side: "front" or "back"
            
        Returns:
            True if the field is expected on that side
        """
        field_side = self.get_field_side(onnx_class_id)
        
        if field_side == FieldSide.BOTH:
            return True
        return field_side.value == side
    
    def get_confidence_boost(self, onnx_class_id: int, detected_side: str) -> float:
        """
        Get confidence boost factor for a field based on detected side.
        
        Args:
            onnx_class_id: The ONNX model class ID
            detected_side: The detected card side ("front" or "back")
            
        Returns:
            Confidence boost factor (1.0 = no boost, >1.0 = boost)
        """
        if self.is_field_expected_for_side(onnx_class_id, detected_side):
            return 1.15  # 15% boost for expected fields
        return 1.0  # No boost for unexpected fields
    
    def validate_field(self, onnx_class_id: int, value: str) -> ValidationResult:
        """
        Validate a field value against its rules.
        
        Args:
            onnx_class_id: The ONNX model class ID
            value: The field value to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        definition = self.FIELD_DEFINITIONS.get(onnx_class_id)
        
        if not definition:
            return ValidationResult(
                is_valid=True,
                field_name=f"class_{onnx_class_id}",
                value=value,
                warnings=[f"Unknown field class ID: {onnx_class_id}"]
            )
        
        warnings = []
        errors = []
        
        # Skip validation for image fields
        if definition.char_type == "image":
            return ValidationResult(
                is_valid=True,
                field_name=definition.canonical_name,
                value=value
            )
        
        # Check for empty values
        if not value or not value.strip():
            if definition.is_critical:
                errors.append(f"Critical field '{definition.canonical_name}' is empty")
            return ValidationResult(
                is_valid=len(errors) == 0,
                field_name=definition.canonical_name,
                value=value,
                error_message=errors[0] if errors else None,
                warnings=warnings
            )
        
        value = value.strip()
        
        # Length validation
        if definition.min_length and len(value) < definition.min_length:
            errors.append(
                f"Value too short: {len(value)} < {definition.min_length} chars"
            )
        elif definition.max_length and len(value) > definition.max_length:
            warnings.append(
                f"Value exceeds max length: {len(value)} > {definition.max_length} chars"
            )
        
        # Pattern validation
        if definition.validation_pattern:
            if not re.match(definition.validation_pattern, value):
                errors.append(
                    f"Value does not match pattern: {definition.validation_pattern}"
                )
        
        # Character type validation
        if definition.char_type == "digits":
            if not value.isdigit():
                # Allow some OCR errors but warn
                digit_count = sum(c.isdigit() for c in value)
                if digit_count < len(value) * 0.8:
                    errors.append("Expected digits but found non-digit characters")
                else:
                    warnings.append("Some non-digit characters detected in digit field")
        
        elif definition.char_type == "arabic":
            # Check for Arabic characters (Unicode range)
            arabic_chars = sum(1 for c in value if '\u0600' <= c <= '\u06FF')
            if arabic_chars < len(value) * 0.5:
                warnings.append("Expected Arabic text but found limited Arabic characters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            field_name=definition.canonical_name,
            value=value,
            error_message=errors[0] if errors else None,
            warnings=warnings
        )
    
    def get_ocr_config(self, onnx_class_id: int) -> Dict[str, Any]:
        """
        Get OCR configuration hints for a field.
        
        Args:
            onnx_class_id: The ONNX model class ID
            
        Returns:
            Dictionary with OCR configuration
        """
        definition = self.FIELD_DEFINITIONS.get(onnx_class_id)
        if definition:
            return definition.ocr_hints
        return {}
    
    def should_skip_ocr(self, onnx_class_id: int) -> bool:
        """
        Check if OCR should be skipped for this field (image-only fields).
        
        Args:
            onnx_class_id: The ONNX model class ID
            
        Returns:
            True if OCR should be skipped
        """
        hints = self.get_ocr_config(onnx_class_id)
        return hints.get("skip_ocr", False)
    
    def get_all_canonical_names(self) -> Set[str]:
        """Get all canonical field names."""
        return set(self._canonical_to_onnx.keys())
    
    def get_all_aliases(self) -> Set[str]:
        """Get all field aliases."""
        return set(self._alias_to_onnx.keys())
    
    def resolve_field_name(self, name: str) -> Optional[str]:
        """
        Resolve any field name/alias to canonical name.
        
        Args:
            name: Field name or alias
            
        Returns:
            Canonical field name or None if not found
        """
        onnx_id = self.get_onnx_class_id(name)
        if onnx_id is not None:
            return self.get_canonical_name(onnx_id)
        return None
    
    def detect_dual_side_from_fields(self, detected_field_ids: Set[int]) -> bool:
        """
        Detect if detected fields suggest a dual-side image.
        
        If fields from both front and back are detected, it's likely a dual-side image.
        
        Args:
            detected_field_ids: Set of detected ONNX field class IDs
            
        Returns:
            True if dual-side image is detected
        """
        front_detected = detected_field_ids & self.FRONT_FIELD_IDS
        back_detected = detected_field_ids & self.BACK_FIELD_IDS
        
        # If we have significant fields from both sides, it's dual-side
        has_front = len(front_detected) >= 2
        has_back = len(back_detected) >= 1
        
        return has_front and has_back


# Singleton instance
_router: Optional[FieldRouter] = None


def get_field_router() -> FieldRouter:
    """Get or create field router singleton."""
    global _router
    if _router is None:
        _router = FieldRouter()
    return _router


# Convenience functions
def get_field_side(onnx_class_id: int) -> FieldSide:
    """Get side classification for a field."""
    return get_field_router().get_field_side(onnx_class_id)


def is_field_expected_for_side(onnx_class_id: int, side: str) -> bool:
    """Check if field is expected on a given side."""
    return get_field_router().is_field_expected_for_side(onnx_class_id, side)


def validate_field(onnx_class_id: int, value: str) -> ValidationResult:
    """Validate a field value."""
    return get_field_router().validate_field(onnx_class_id, value)


def get_fields_for_side(side: str) -> Set[int]:
    """Get all field IDs for a side."""
    return get_field_router().get_fields_for_side(side)
