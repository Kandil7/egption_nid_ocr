"""
Text Post-Processing Utilities
Cleans and normalizes OCR output based on field type.
"""

import re
import unicodedata


# Arabic Unicode ranges for text cleaning
ARABIC_CHARS = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EUROPEAN_DIGITS = "0123456789"
ARABIC_INDIC_TO_EUROPEAN = str.maketrans(ARABIC_INDIC_DIGITS, EUROPEAN_DIGITS)


def _normalize_digits(text: str) -> str:
    """Normalize Arabic-Indic digits to European digits."""
    return text.translate(ARABIC_INDIC_TO_EUROPEAN)


def _reorder_arabic_tokens(text: str) -> str:
    """
    Reverse entire Arabic text string to correct character order.
    """
    if not text:
        return text
    return text[::-1]


def _fix_common_digit_ocr_errors(text: str) -> str:
    """Fix common OCR digit recognition errors."""
    replacements = [
        ("O", "0"), ("o", "0"), ("Q", "0"), ("D", "0"),
        ("l", "1"), ("I", "1"), ("|", "1"), ("!", "1"),
        ("Z", "2"), ("S", "5"), ("B", "8"), ("G", "6"),
        ("b", "6"), ("q", "9"), ("g", "9"),
    ]
    for wrong, right in replacements:
        text = text.replace(wrong, right)
    return text


def _fix_nid_century_digit(text: str) -> str:
    """Fix NID century digit based on birth year context."""
    if len(text) < 14 or not text.isdigit():
        return text
    
    century = int(text[0])
    year = int(text[1:3])
    
    if century in [2, 3]:
        return text
    
    # Infer century from year
    if 0 <= year <= 26:
        return "3" + text[1:]
    elif 27 <= year <= 99:
        return "2" + text[1:]
    
    # Handle common visual confusions
    if century == 6:
        return "3" + text[1:] if year <= 30 else "2" + text[1:]
    elif century in [5, 8] and year <= 30:
        return "3" + text[1:]
    
    return text


def _is_valid_nid_format(nid: str) -> bool:
    """Check if NID has valid format and components."""
    if len(nid) != 14 or not nid.isdigit():
        return False
    
    century_char = nid[0]
    if century_char not in ['2', '3']:
        return False
    
    # Birth date components
    yy = nid[1:3]
    mm = nid[3:5]
    dd = nid[5:7]
    
    month = int(mm)
    if month < 1 or month > 12:
        return False
    
    day = int(dd)
    if day < 1 or day > 31:
        return False
    
    # Gov code
    gov_code = int(nid[7:9])
    # Known Egyptian Gov codes: 01 (Cairo), 02 (Alex), 03 (Port Said), etc. up to 88
    if gov_code < 1 or gov_code > 88:
        return False
    
    return True


def _validate_and_correct_nid(text: str) -> tuple:
    """Validate and auto-correct Egyptian National ID number."""
    digits = re.sub(r'\D', '', text)
    
    if len(digits) < 10:
        return digits, False
    
    if len(digits) > 14:
        for i in range(len(digits) - 14 + 1):
            candidate = digits[i:i + 14]
            if _is_valid_nid_format(candidate):
                return candidate, True
        digits = digits[:14]
    
    if len(digits) < 14:
        digits = digits.ljust(14, '0')
    
    # Fix century digit
    digits = _fix_nid_century_digit(digits)
    
    is_valid = _is_valid_nid_format(digits)
    return digits, is_valid


def clean_field(text: str, field_type: str) -> str:
    """Clean and normalize text based on field type."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    if field_type in ["nid", "front_nid", "back_nid", "id_number", "serial", "serial_num", "issue_code"]:
        text = _normalize_digits(text)
        text = _fix_common_digit_ocr_errors(text)
        
        if field_type in ["nid", "front_nid", "back_nid", "id_number"]:
            text, _ = _validate_and_correct_nid(text)
        else:
            text = re.sub(r"\D", "", text)

    elif field_type in ["firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"]:
        pattern = f"[^\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeffa-zA-Z\\s]"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = _reorder_arabic_tokens(text)

    elif field_type in ["serial", "job_title"]:
        text = re.sub(r"[^A-Za-z\s]", "", text).strip().upper()

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
