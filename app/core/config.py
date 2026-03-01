"""
Application Configuration
Loads settings from environment variables with .env file support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


# Field classes based on NASO7Y project
# Source: https://github.com/NASO7Y/OCR_Egyptian_ID
NASO7Y_CLASSES = {
    0: "firstName",  # First name (Arabic)
    1: "lastName",  # Last name (Arabic)
    2: "serial",  # Serial number
    3: "address",  # Address
    4: "nid",  # National ID number
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

    # Card Field Class IDs (NASO7Y schema)
    CLASS_ID_CARD: int = Field(default=0, description="Class ID for card detection")
    CLASS_NAMES: dict = Field(
        default=NASO7Y_CLASSES, description="Mapping of class IDs to field names (NASO7Y schema)"
    )

    # Image Settings
    RECTIFIED_SIZE: tuple = Field(
        default=(1024, 640), description="Rectified card size (width, height)"
    )

    # Image Validation Settings
    MAX_IMAGE_SIZE_MB: int = Field(default=10, description="Maximum allowed image size in MB")
    MIN_QUALITY_SCORE: float = Field(
        default=0.35, description="Minimum acceptable image quality score"
    )
    TARGET_IMAGE_WIDTH: int = Field(default=1200, description="Target width for image resizing")

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
