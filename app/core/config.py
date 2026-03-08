"""
Application Configuration
Loads settings from environment variables with .env file support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


# Field classes based on ACTUAL ONNX model (weights/field_detector.onnx)
# This is the PRIMARY model for field detection
ONNX_FIELD_DETECTOR_CLASSES = {
    0: "first_name",
    1: "last_name",
    2: "add_line_1",
    3: "add_line_2",
    4: "front_nid",
    5: "back_nid",
    6: "serial_num",
    7: "issue_code",
    8: "expiry_date",
    9: "job_title",
    10: "gender",
    11: "religion",
    12: "marital_status",
    13: "face",
    14: "front_logo",
    15: "address",
    16: "dob",
}

# Field classes from .pt model (weights/detect_odjects.pt) - fallback
NASO7Y_CLASSES = {
    0: "address",
    4: "firstName",
    24: "lastName",
    29: "serial",
    25: "nid",
    1: "demo",
    2: "dob",
    3: "expiry",
    5: "front_logo",
    26: "nid_back",
    27: "photo",
    28: "poe",
    30: "watermark_tut",
    6: "invalid_address",
    7: "invalid_barcode",
    8: "invalid_demo",
    9: "invalid_dob",
    10: "invalid_expiry",
    11: "invalid_firstName",
    12: "invalid_issue",
    13: "invalid_job",
    14: "invalid_lastName",
    15: "invalid_logo",
    16: "invalid_nid",
    17: "invalid_nid_back",
    18: "invalid_photo",
    19: "invalid_poe",
    20: "invalid_serial",
    21: "invalid_watermark_tut",
    22: "issue",
    23: "job",
}

# Card detection classes (corner-based detection)
CARD_CLASSES = {
    0: "back-bottom",
    1: "back-left", 
    2: "back-right",
    3: "back-up",
    4: "front-bottom",
    5: "front-left",
    6: "front-right",
    7: "front-up",
}

# Field name mapping (ONNX -> internal names)
FIELD_NAME_MAP = {
    "first_name": "firstName",
    "last_name": "lastName",
    "add_line_1": "add_line_1",
    "add_line_2": "add_line_2",
    "front_nid": "nid",
    "back_nid": "nid",
    "serial_num": "serial",
    "issue_code": "issue_code",
    "expiry_date": "expiry_date",
    "job_title": "job_title",
    "gender": "gender",
    "religion": "religion",
    "marital_status": "marital_status",
    "face": "photo",
    "front_logo": "front_logo",
    "address": "address",
    "dob": "dob",
    # NASO7Y mappings
    "firstName": "firstName",
    "lastName": "lastName",
    "nid": "nid",
    "serial": "serial",
}

# Canonical field mapping (our internal names -> NASO7Y names)
FIELD_ALIASES = {
    "first_name": "firstName",
    "last_name": "lastName",
    "front_nid": "nid",
    "serial_num": "serial",
    "address": "address",
}

# OCR engine routing based on field type
FIELD_OCR_CONFIG = {
    "firstName": {"engine": "paddle", "lang": "ar"},
    "lastName": {"engine": "paddle", "lang": "ar"},
    "address": {"engine": "paddle", "lang": "ar"},
    "nid": {"engine": "easyocr", "digits_only": True},
    "serial": {"engine": "easyocr", "lang": "en"},
}


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App Configuration
    APP_ENV: str = Field(default="development", description="Environment: development | production")
    APP_HOST: str = Field(default="0.0.0.0", description="Host to bind the server")
    APP_PORT: int = Field(default=8000, description="Port to bind the server")
    APP_WORKERS: int = Field(default=2, description="Number of uvicorn workers")
    APP_TITLE: str = Field(default="Egyptian ID OCR API", description="API title")
    APP_VERSION: str = Field(default="1.0.0", description="API version")

    # Model Paths
    YOLO_CARD_MODEL: str = Field(
        default="weights/detect_id_card.pt", description="Path to card detection model"
    )
    YOLO_FIELDS_MODEL: str = Field(
        default="weights/detect_odjects.pt", description="Path to fields detection model"
    )
    YOLO_NID_MODEL: str = Field(
        default="weights/detect_id.pt", description="Path to NID digit detection model"
    )
    MODELS_CACHE_DIR: str = Field(
        default="./models_cache", description="Directory for OCR model caches"
    )

    # YOLO Inference Settings
    YOLO_CONF_THRESHOLD: float = Field(default=0.50, description="Confidence threshold for YOLO")
    YOLO_IOU_THRESHOLD: float = Field(default=0.45, description="IOU threshold for NMS")
    YOLO_INPUT_SIZE: int = Field(default=640, description="Input size for YOLO models")

    # OCR Settings
    OCR_CPU_THREADS: int = Field(default=4, description="Number of CPU threads for OCR")
    OCR_ENABLE_MKL: bool = Field(default=True, description="Enable Intel MKL-DNN acceleration")
    PADDLE_USE_GPU: bool = Field(default=False, description="Use GPU for PaddleOCR if available")
    PADDLE_AR_REC_MODEL_DIR: str = Field(
        default="", description="Optional custom directory for Arabic PaddleOCR rec model"
    )
    PADDLE_DIGIT_REC_MODEL_DIR: str = Field(
        default="",
        description="Optional custom directory for digit/Latin PaddleOCR PP-OCRv4 rec model",
    )
    TESSDATA_DIR: str = Field(
        default="./weights",
        description="Directory containing Tesseract trained data files",
    )

    # Card Field Class IDs (NASO7Y schema)
    CLASS_ID_CARD: int = Field(default=0, description="Class ID for card detection")
    CLASS_NAMES: dict = Field(
        default_factory=lambda: NASO7Y_CLASSES, description="Mapping of class IDs to field names (NASO7Y schema)"
    )

    # ONNX Field Detector Classes
    ONNX_FIELD_DETECTOR_CLASSES: dict = Field(
        default_factory=lambda: ONNX_FIELD_DETECTOR_CLASSES,
        description="Mapping of class IDs to field names (ONNX model)"
    )

    # Field Name Mapping (ONNX -> internal)
    FIELD_NAME_MAP: dict = Field(
        default_factory=lambda: FIELD_NAME_MAP,
        description="Mapping from ONNX/YOLO names to internal field names"
    )

    # Image Settings
    RECTIFIED_SIZE: tuple = Field(
        default=(1024, 640), description="Rectified card size (width, height)"
    )
    TARGET_IMAGE_WIDTH: int = Field(
        default=900, description="Target width for image resizing and normalization"
    )

    # Image Validation Settings
    MAX_IMAGE_SIZE_MB: int = Field(default=10, description="Maximum allowed image size in MB")
    MIN_QUALITY_SCORE: float = Field(
        default=0.35, description="Minimum acceptable image quality score"
    )

    # Logging Settings
    LOG_LEVEL: str = Field(
        default="INFO", description="Logging level: DEBUG | INFO | WARNING | ERROR"
    )
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    LOG_ROTATION: str = Field(default="10 MB", description="Log rotation size")
    LOG_RETENTION: str = Field(default="7 days", description="Log retention period")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance."""
    return settings
