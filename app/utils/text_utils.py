"""
Text Post-Processing Utilities
Cleans and normalizes OCR output based on field type.
"""

import re
import unicodedata
from typing import List


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
    
    # Check if we can recover a 13-digit NID
    if len(text) == 13:
        try:
            from app.models.id_parser import calculate_nid_checksum
            # Pad with 0 as a dummy checksum to calculate the real one
            temp_nid = text + "0"
            expected_checksum = calculate_nid_checksum(temp_nid)
            if expected_checksum != -1:
                text = text + str(expected_checksum)
        except ImportError:
            pass

    # If we have exactly 14 digits, apply corrections
    if len(text) == 14:
        # Validate century code (digit 1)
        if text[0] not in '23':
            # Try to infer from context (age)
            # For now, default to 2 (1900s) as most common
            text = '2' + text[1:]
        
        # Validate month (digits 3-4, 0-indexed: 2-3)
        month = text[2:4]
        if not ('01' <= month <= '12'):
            # Common OCR mistakes: 0O, 1I, 6G
            month = month.replace('O', '0').replace('I', '1').replace('l', '1')
            if '01' <= month <= '12':
                text = text[:2] + month + text[4:]
        
        # Validate day (digits 5-6, 0-indexed: 4-5)
        day = text[4:6]
        if not ('01' <= day <= '31'):
            day = day.replace('O', '0').replace('I', '1').replace('l', '1')
            if '01' <= day <= '31':
                text = text[:4] + day + text[6:]

        # Attempt checksum-guided correction
        try:
            from app.models.id_parser import calculate_nid_checksum, validate_nid_checksum
            if not validate_nid_checksum(text):
                # Try swapping commonly confused digits
                confusable_pairs = [
                    ('0', '8'), ('8', '0'),
                    ('1', '7'), ('7', '1'),
                    ('3', '8'), ('8', '3'),
                    ('5', '6'), ('6', '5'),
                    ('2', '7'), ('7', '2'),
                    ('0', '5'), ('5', '0')
                ]
                found_correction = False
                for i, char in enumerate(text[:-1]): # Don't swap the checksum digit itself
                    if found_correction:
                        break
                    for from_char, to_char in confusable_pairs:
                        if char == from_char:
                            candidate = text[:i] + to_char + text[i+1:]
                            if validate_nid_checksum(candidate):
                                text = candidate
                                found_correction = True
                                break
        except ImportError:
            pass
    
    # If length is wrong, try to extract a valid 14-digit sequence before just truncating
    elif len(text) > 14:
        import re
        # Look for [23] followed by 13 digits (rough heuristic for NID)
        match = re.search(r'[23]\d{13}', text)
        if match:
            text = match.group(0)
        else:
            # Fallback to first 14 digits
            text = text[:14]
    
    return text


def _is_arabic_char(char: str) -> bool:
    """Check if character is Arabic."""
    code = ord(char)
    return (0x0600 <= code <= 0x06FF or 
            0x0750 <= code <= 0x077F or
            0x08A0 <= code <= 0x08FF or
            0xFB50 <= code <= 0xFDFF or
            0xFE70 <= code <= 0xFEFF)


def _reorder_arabic_tokens(text: str) -> str:
    """
    Reorder Arabic text at BOTH character and word level.
    
    OCR engines scan left-to-right, but Arabic is written right-to-left.
    This causes TWO issues:
    1. Characters within each word appear reversed
    2. Word order appears reversed
    
    This function fixes BOTH:
    1. Reverses characters within each Arabic word
    2. Reverses the order of Arabic word sequences
    
    Example:
        OCR returns: "دمحم يوادعس" (characters reversed, words in wrong order)
        Step 1 (char reversal): "محمد سعداوي" 
        Step 2 (word reversal): "سعداوي محمد"
        Output: "سعداوي محمد" (correct)
        
        Complex: "دمحم يوادعس عيفشلا دبع رصان"
        → "محمد سعداوي الشفيع عبد ناصر"
        → "ناصر عبد الشفيع سعداوي محمد"
    """
    if not text:
        return text
    
    # Split into words
    words = text.split(' ')
    
    if len(words) <= 1:
        # Single word - just reverse characters if Arabic
        if words and words[0]:
            word = words[0]
            if any(_is_arabic_char(c) for c in word):
                return word[::-1]  # Reverse characters
        return text
    
    # Step 1: Reverse characters within each Arabic word
    corrected_words = []
    for word in words:
        if not word:
            corrected_words.append(word)
            continue
            
        if any(_is_arabic_char(c) for c in word):
            # Reverse characters in Arabic words
            corrected_words.append(word[::-1])
        else:
            # Keep non-Arabic words as-is
            corrected_words.append(word)
    
    # Step 2: Reverse order of contiguous Arabic word sequences
    result = []
    arabic_sequence = []
    
    for word in corrected_words:
        if not word:
            continue
            
        # Check if word is primarily Arabic
        is_arabic = all(_is_arabic_char(c) or c.isspace() for c in word if c.isalpha())
        
        if is_arabic:
            arabic_sequence.append(word)
        else:
            # Flush Arabic sequence (reversed order)
            if arabic_sequence:
                result.extend(reversed(arabic_sequence))
                arabic_sequence = []
            result.append(word)
    
    # Flush any remaining Arabic sequence
    if arabic_sequence:
        result.extend(reversed(arabic_sequence))
    
    return ' '.join(result)


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
        ('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'),  # Alef forms
        ('ة', 'ه'),  # Ta marbuta
        ('ى', 'ي'),  # Alif maqsura
        ('ـ', ''),   # Tatweel
        ('ؤ', 'ء'),  # Waw with hamza
        ('ئ', 'ء'),  # Yeh with hamza
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
        
        # NID-specific validation and correction
        if field_type in ["nid", "front_nid", "back_nid", "id_number"]:
            text = _validate_and_correct_nid(text)

    # Arabic text fields - keep both Arabic AND English (names can be in either)
    elif field_type in ["firstName", "lastName", "name_ar", "address", "add_line_1", "add_line_2", "nationality"]:
        # Keep Arabic, English letters, and spaces
        pattern = f"[^{ARABIC_CHARS}a-zA-Z\\s]"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Apply proper Arabic reordering (character-level within words)
        text = _reorder_arabic_tokens(text)
        
        # Apply Arabic normalization for consistency
        text = normalize_arabic_text(text)

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
