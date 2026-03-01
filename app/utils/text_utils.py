"""
Text Post-Processing Utilities
Cleans and normalizes OCR output based on field type.
"""

import re


# Arabic Unicode ranges for text cleaning
ARABIC_CHARS = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"


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

    text = text.strip()

    # ID numbers - keep only digits
    if field_type in ["nid", "front_nid", "back_nid", "serial", "serial_num", "issue_code"]:
        text = re.sub(r"\D", "", text)
        # Fix common OCR digit mistakes
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

    # Arabic text fields - keep both Arabic AND English (names can be in either)
    elif field_type in ["firstName", "lastName", "address", "add_line_1", "add_line_2"]:
        # Keep Arabic, English letters, and spaces
        # Arabic: \u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff
        # English: A-Za-z
        pattern = f"[^\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeffa-zA-Z\\s]"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()

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
