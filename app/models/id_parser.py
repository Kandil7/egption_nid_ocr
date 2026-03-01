"""
Egyptian National ID Parser
Parses the 14-digit Egyptian national ID number to extract:
- Birth date
- Gender
- Governorate
- Sequence number
- Age
"""

import re
from typing import Dict, Optional
from dataclasses import dataclass

from app.core.logger import logger


# Egyptian Governorates mapping (based on governorate code)
GOVERNORATES: Dict[int, str] = {
    1: "القاهرة",
    2: "الإسكندرية",
    3: "بورسعيد",
    4: "السويس",
    11: "دمياط",
    12: "الدقهلية",
    13: "الشرقية",
    14: "القليوبية",
    15: "كفر الشيخ",
    16: "الغربية",
    17: "المنوفية",
    18: "البحيرة",
    19: "الإسماعيلية",
    21: "الجيزة",
    22: "بني سويف",
    23: "الفيوم",
    24: "المنيا",
    25: "أسيوط",
    26: "سوهاج",
    27: "قنا",
    28: "أسوان",
    29: "الأقصر",
    31: "البحر الأحمر",
    32: "الوادي الجديد",
    33: "مطروح",
    34: "شمالسيناء",
    35: "جنوبسيناء",
    88: "خارج الجمهورية",
}


# Common OCR digit mistakes
OCR_DIGIT_FIXES = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}


@dataclass
class ParsedNationalID:
    """Parsed national ID information."""

    valid: bool
    birth_date: Optional[str] = None
    governorate: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    sequence: Optional[str] = None
    raw: Optional[str] = None
    error: Optional[str] = None


def parse_national_id(raw_id: str) -> ParsedNationalID:
    """
    Parse Egyptian national ID (14 digits).

    The national ID structure:
    - Digit 1: Century code (2 = 1900s, 3 = 2000s)
    - Digits 2-3: Birth year (last 2 digits)
    - Digits 4-5: Birth month
    - Digits 6-7: Birth day
    - Digits 8-9: Governorate code
    - Digit 10: Gender code (odd = male, even = female)
    - Digits 11-14: Sequence number

    Args:
        raw_id: Raw national ID string (may contain non-digit characters)

    Returns:
        ParsedNationalID object with parsed information
    """
    # Clean the ID - keep only digits
    national_id = re.sub(r"\D", "", raw_id)

    # Apply common OCR fixes
    for wrong, right in OCR_DIGIT_FIXES.items():
        national_id = national_id.replace(wrong, right)

    # Validate length
    if len(national_id) != 14:
        return ParsedNationalID(
            valid=False,
            error=f"Invalid length: expected 14 digits, got {len(national_id)}",
            raw=national_id,
        )

    try:
        # Extract components
        century_code = int(national_id[0])
        year = int(national_id[1:3])
        month = int(national_id[3:5])
        day = int(national_id[5:7])
        gov_code = int(national_id[7:9])
        gender_digit = int(national_id[9])

        # Validate century code
        if century_code not in [2, 3]:
            return ParsedNationalID(
                valid=False,
                error="Invalid century code (must be 2 or 3)",
                raw=national_id,
            )

        # Determine full birth year
        century = 1900 if century_code == 2 else 2000
        full_year = century + year

        # Validate month and day
        if not (1 <= month <= 12):
            return ParsedNationalID(
                valid=False, error=f"Invalid month: {month}", raw=national_id
            )

        if not (1 <= day <= 31):
            return ParsedNationalID(
                valid=False, error=f"Invalid day: {day}", raw=national_id
            )

        # Determine gender
        gender = "ذكر" if gender_digit % 2 != 0 else "أنثى"

        # Get governorate
        governorate = GOVERNORATES.get(gov_code, "غير معروف")

        # Calculate age
        from datetime import datetime

        current_year = datetime.now().year
        age = current_year - full_year

        # Extract sequence number
        sequence = national_id[9:13]

        # Format birth date
        birth_date = f"{day:02d}/{month:02d}/{full_year}"

        return ParsedNationalID(
            valid=True,
            birth_date=birth_date,
            governorate=governorate,
            gender=gender,
            age=age,
            sequence=sequence,
            raw=national_id,
        )

    except (ValueError, IndexError) as e:
        return ParsedNationalID(
            valid=False, error=f"Parse error: {str(e)}", raw=national_id
        )


def validate_national_id(national_id: str) -> bool:
    """
    Validate if a national ID is structurally valid.

    Args:
        national_id: National ID string to validate

    Returns:
        True if valid, False otherwise
    """
    result = parse_national_id(national_id)
    return result.valid


def format_national_id(national_id: str) -> str:
    """
    Format national ID with proper spacing.

    Args:
        national_id: Raw national ID

    Returns:
        Formatted ID (e.g., "299 0101 12345 67")
    """
    cleaned = re.sub(r"\D", "", national_id)
    if len(cleaned) != 14:
        return national_id

    return f"{cleaned[:3]} {cleaned[3:7]} {cleaned[7:11]} {cleaned[11:]}"
