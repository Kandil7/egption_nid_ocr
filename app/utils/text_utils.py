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
    Heuristic reordering for Arabic text.

    For pure-Arabic tokens, reverse codepoint order to better match
    expected visual order when OCR returns left-to-right sequences.

    Mixed tokens (with Latin or digits) are left unchanged.
    """
    if not text:
        return text

    tokens = text.split()
    # pattern that matches tokens composed only of Arabic characters
    arabic_only = re.compile(f"^[{ARABIC_CHARS}]+$")

    reordered = []
    for tok in tokens:
        if arabic_only.match(tok):
            reordered.append(tok[::-1])
        else:
            reordered.append(tok)
    return " ".join(reordered)


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
    if field_type in ["nid", "front_nid", "back_nid", "id_number", "serial", "serial_num", "issue_code"]:
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

    # Arabic text fields - keep both Arabic AND English (names can be in either)
    elif field_type in ["firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"]:
        # Keep Arabic, English letters, and spaces
        # Arabic: \u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff
        # English: A-Za-z
        pattern = f"[^\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeffa-zA-Z\\s]"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Apply heuristic reordering for Arabic-only tokens
        text = _reorder_arabic_tokens(text)

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
