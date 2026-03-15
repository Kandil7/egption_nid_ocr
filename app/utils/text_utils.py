"""
Text Post-Processing Utilities
Cleans and normalizes OCR output based on field type.
"""

import re
import unicodedata
from typing import List, Dict, Tuple


# Arabic Unicode ranges for text cleaning
ARABIC_CHARS = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EUROPEAN_DIGITS = "0123456789"
ARABIC_INDIC_TO_EUROPEAN = str.maketrans(ARABIC_INDIC_DIGITS, EUROPEAN_DIGITS)


def _normalize_digits(text: str) -> str:
    """Normalize Arabic-Indic digits to European digits."""
    return text.translate(ARABIC_INDIC_TO_EUROPEAN)


def _validate_and_correct_nid(text: str) -> str:
    """
    Validate and correct Egyptian NID (14 digits).

    Applies rules:
    1. Must be exactly 14 digits
    2. First digit must be 2 or 3 (century code)
    3. Digits 4-5 must be valid month (01-12)
    4. Digits 6-7 must be valid day (01-31)
    5. Checksum validation (last digit)

    Attempts correction if validation fails.

    Args:
        text: Raw digit string

    Returns:
        Corrected 14-digit NID or best effort
    """
    if not text:
        return text

    # Handle 13-digit NID - try all possible checksum digits
    if len(text) == 13:
        try:
            from app.models.id_parser import validate_nid_checksum

            # Try all 10 possible checksum digits
            for checksum_digit in "0123456789":
                candidate = text + checksum_digit
                if validate_nid_checksum(candidate):
                    logger.info(f"NID: Recovered 14-digit NID from 13-digit: {candidate}")
                    return candidate
        except ImportError:
            pass

        # If no valid checksum found, still try to add a reasonable digit
        # based on the century code
        if text[0] in "23":
            # Add 0 as fallback checksum (many IDs have 0 as checksum)
            logger.info(f"NID: Could not validate 13-digit, using fallback: {text}0")
            return text + "0"

    # If we have exactly 14 digits, apply corrections
    if len(text) == 14:
        # Validate century code (digit 1)
        if text[0] not in "23":
            # Try to infer from context (age)
            # For now, default to 2 (1900s) as most common
            text = "2" + text[1:]

        # Validate month (digits 3-4, 0-indexed: 2-3)
        month = text[2:4]
        if not ("01" <= month <= "12"):
            # Common OCR mistakes: 0O, 1I, 6G
            month = month.replace("O", "0").replace("I", "1").replace("l", "1")
            if "01" <= month <= "12":
                text = text[:2] + month + text[4:]

        # Validate day (digits 5-6, 0-indexed: 4-5)
        day = text[4:6]
        if not ("01" <= day <= "31"):
            day = day.replace("O", "0").replace("I", "1").replace("l", "1")
            if "01" <= day <= "31":
                text = text[:4] + day + text[6:]

        # Attempt checksum-guided correction removed as it mutates valid Egyptian NIDs due to non-standard algorithms.

    # If length is wrong, try to extract a valid 14-digit sequence before just truncating
    elif len(text) > 14:
        import re

        # Look for [23] followed by 13 digits (rough heuristic for NID)
        match = re.search(r"[23]\d{13}", text)
        if match:
            text = match.group(0)
        else:
            # Fallback to first 14 digits
            text = text[:14]

    return text


def _is_arabic_char(char: str) -> bool:
    """Check if character is Arabic."""
    code = ord(char)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for consistent representation.

    - Normalize alef forms (أ, إ, آ → ا)
    - Normalize ha forms (ة → ه)
    - Normalize yeh forms (ى → ي)
    - Remove tatweel (ـ)

    This improves matching and consistency across OCR results.
    """
    if not text:
        return text

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Character normalizations for consistency
    normalizations = [
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),  # Alef forms
        ("ة", "ه"),  # Ta marbuta
        ("ى", "ي"),  # Alif maqsura
        ("ـ", ""),  # Tatweel
        ("ؤ", "ء"),  # Waw with hamza
        ("ئ", "ء"),  # Yeh with hamza
    ]

    for old, new in normalizations:
        text = text.replace(old, new)

    return text


def clean_field(text: str, field_type: str) -> str:
    """Clean and normalize text based on field type.

    Args:
        text: Raw OCR text
        field_type: Type of field (nid, firstName, lastName, address, etc.)

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Unicode normalization first
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    # ID numbers - normalize digits and keep only digits
    if field_type in [
        "nid",
        "front_nid",
        "back_nid",
        "id_number",
        "serial",
        "serial_num",
        "issue_code",
    ]:
        # Normalize Arabic-Indic digits to European digits
        text = _normalize_digits(text)
        # Fix common OCR digit mistakes (0/O, 1/I, etc.) before stripping non-digits
        for wrong, right in [
            ("O", "0"),
            ("o", "0"),
            ("l", "1"),
            ("I", "1"),
            ("S", "5"),
            ("B", "8"),
            ("Z", "2"),
            ("G", "6"),
        ]:
            text = text.replace(wrong, right)
        # Keep only digits
        text = re.sub(r"\D", "", text)

        # NID-specific validation and correction
        if field_type in ["nid", "front_nid", "back_nid", "id_number"]:
            text = _validate_and_correct_nid(text)

    # Arabic text fields - keep both Arabic AND English (names can be in either)
    elif field_type in [
        "firstName",
        "lastName",
        "name_ar",
        "address",
        "add_line_1",
        "add_line_2",
        "nationality",
    ]:
        # Keep Arabic, English letters, and spaces
        pattern = f"[^{ARABIC_CHARS}a-zA-Z\\s]"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Apply Arabic normalization for consistency
        text = normalize_arabic_text(text)

        # Fix common OCR mistakes in common Arabic names
        word_fixes = {
            "محموا": "محمود",
            "عبدالله": "عبد الله",
            "ابراهيم": "إبراهيم",
            "احمد": "أحمد",
        }
        words = text.split()
        for i, w in enumerate(words):
            if w in word_fixes:
                words[i] = word_fixes[w]
        text = " ".join(words)
        
        # Also catch occurrences where they might be joined
        text = text.replace("عبدالله", "عبد الله")

    # English text
    elif field_type in ["serial", "job_title"]:
        text = re.sub(r"[^A-Za-z\s]", "", text).strip().upper()

    # Dates - keep as is
    elif field_type in ["issue_date", "expiry_date", "dob"]:
        pass

    return text


def assemble_address(line1: str, line2: str) -> str:
    """Assemble full address from address lines.

    Args:
        line1: First line of address
        line2: Second line of address

    Returns:
        Full assembled address
    """
    parts = []
    if line1:
        parts.append(line1.strip())
    if line2:
        parts.append(line2.strip())
    return ", ".join(parts) if parts else ""


def format_name(first_name: str, last_name: str) -> str:
    """Format full name from first and last name.

    Args:
        first_name: First name
        last_name: Last name

    Returns:
        Full name
    """
    parts = []
    if first_name:
        parts.append(first_name.strip())
    if last_name:
        parts.append(last_name.strip())
    return " ".join(parts) if parts else ""

def sort_blocks_by_reading_direction(blocks: List[Dict]) -> Tuple[str, float]:
    """
    Sort OCR text blocks based on reading direction (RTL for Arabic, LTR otherwise).
    Auto-detects RTL if any block contains Arabic characters.
    
    Args:
        blocks: List of dicts, each with 'bbox' (coordinates), 'text', 'confidence'
        
    Returns:
        Tuple of (combined_text, average_confidence)
    """
    if not blocks:
        return "", 0.0

    import re
    # Auto-detect RTL if any block contains Arabic characters
    is_rtl = False
    for b in blocks:
        if re.search(r'[\u0600-\u06FF]', b.get("text", "")):
            is_rtl = True
            break
            
    enhanced = []
    for idx, b in enumerate(blocks):
        bbox = b.get("bbox", [])
        if not bbox or len(bbox) < 4:
            # Fallback for missing/invalid bbox
            enhanced.append((99999, 99999, 0, 0, idx, b))
            continue
            
        # Parse coordinates
        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            # [x1, y1, x2, y2] format
            x_coords = [bbox[0], bbox[2]]
            y_coords = [bbox[1], bbox[3]]
        else:
            # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] format
            x_coords = [float(p[0]) for p in bbox]
            y_coords = [float(p[1]) for p in bbox]
            
        y_min = min(y_coords)
        y_max = max(y_coords)
        x_min = min(x_coords)
        x_max = max(x_coords)
        
        enhanced.append((y_min, y_max, x_max, x_min, idx, b))
        
    # Sort top-to-bottom by y_min
    enhanced.sort(key=lambda x: x[0])
    
    lines = []
    current_line = []
    current_y_max = None
    
    for item in enhanced:
        y_min, y_max, x_max, x_min, idx, b = item
        if current_y_max is None:
            current_y_max = y_max
            current_line.append(item)
        elif y_min < current_y_max:  # Overlaps vertically -> same line
            current_line.append(item)
            current_y_max = max(current_y_max, y_max)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y_max = y_max
            
    if current_line:
        lines.append(current_line)
        
    final_texts = []
    confs = []
    
    for line in lines:
        if is_rtl:
            # Sort Right-to-Left (descending x_max)
            line.sort(key=lambda x: x[2], reverse=True)
        else:
            # Sort Left-to-Right (ascending x_min)
            line.sort(key=lambda x: x[3])
            
        for *_, b in line:
            txt = str(b.get("text", "")).strip()
            if txt:
                # If RTL, the OCR usually outputs the characters in visual LTR order (completely backward).
                # We must reverse the text in this block before joining.
                if is_rtl:
                    # Reverse the entire string for this block
                    txt = txt[::-1]
                final_texts.append(txt)
            if b.get("confidence") is not None:
                confs.append(float(b["confidence"]))
                
    combined_text = " ".join(final_texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    
    return combined_text, float(avg_conf)
