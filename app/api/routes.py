"""
API Routes for Egyptian ID OCR Service
Provides endpoints for ID card extraction and visualization.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import io

from app.services.pipeline import IDExtractionPipeline
from app.core.config import settings
from app.core.logger import logger

router = APIRouter(prefix="/api/v1", tags=["OCR"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 10


@router.post("/extract")
async def extract_id(file: UploadFile = File(...)):
    """Extract information from an Egyptian ID card image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file.content_type}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large")
    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="File too small")

    logger.info(f"Processing file: {file.filename}")

    pipeline = IDExtractionPipeline()
    result = pipeline.process(contents)

    return JSONResponse(content=result)


@router.post("/visualize")
async def visualize_fields(file: UploadFile = File(...)):
    """
    Visualize detected fields on the ID card.
    Shows bounding boxes around detected areas.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported format")

    contents = await file.read()

    # Decode image
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result_image = image.copy()

    pipeline = IDExtractionPipeline()
    detector = pipeline._detector

    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not loaded")

    try:
        # Detect card
        card_img = detector.crop_card(image.copy())

        # Get card detection
        card_dets = detector.card_detector.detect(image)
        for det in card_dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                result_image, "Card", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

        # Detect fields (use crop_fields method which handles ONNX/YOLO logic)
        fields_dict = detector.crop_fields(card_img)
        
        # Convert to detection list for visualization
        from app.models.detector import Detection
        field_dets = []
        for class_name, (crop_img, conf) in fields_dict.items():
            # Find this crop in the original image (approximate)
            # For visualization, we'll just use the full card image
            h, w = card_img.shape[:2]
            # Assume field takes up portion of image
            field_dets.append(
                Detection(
                    bbox=[0, 0, w, h],  # Approximate - actual bbox from ONNX/YOLO
                    class_id=0,
                    class_name=class_name,
                    confidence=float(conf),  # Convert numpy float to Python float
                )
            )

        # Color for each class
        colors = {
            "firstName": (0, 255, 0),
            "lastName": (0, 255, 128),
            "nid": (0, 128, 255),
            "serial": (255, 255, 0),
            "address": (255, 0, 255),
        }

        for det in field_dets:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (0, 255, 0))
            cv2.rectangle(card_img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} ({det.confidence:.2f})"
            cv2.putText(card_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Combine
        h, w = result_image.shape[:2]
        card_img = cv2.resize(card_img, (w, int(card_img.shape[0] * w / card_img.shape[1])))
        combined = np.hstack([result_image, card_img])

        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(
            combined,
            "Detected Fields",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        result_image = combined

    except Exception as e:
        logger.error(f"Visualize error: {e}")

    # Encode
    _, buffer = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")


@router.post("/debug")
async def debug_detection(file: UploadFile = File(...)):
    """
    Debug endpoint - shows raw YOLO detections.
    Returns JSON with all detected classes and their IDs.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported format")

    contents = await file.read()

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    pipeline = IDExtractionPipeline()
    detector = pipeline._detector

    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not loaded")

    try:
        # Get all detections
        card_dets = detector.card_detector.detect(image)
        
        # Use crop_fields to get field detections (works with ONNX or YOLO)
        fields_dict = detector.crop_fields(image)
        
        # Convert to detection list
        from app.models.detector import Detection
        field_dets = []
        for class_name, (crop_img, conf) in fields_dict.items():
            h, w = crop_img.shape[:2]
            field_dets.append(
                Detection(
                    bbox=[0, 0, w, h],
                    class_id=0,
                    class_name=class_name,
                    confidence=float(conf),  # Convert numpy float to Python float
                )
            )

        # Get model class names from detector
        class_names = detector.get_field_class_names()

        return JSONResponse(
            content={
                "model_class_names": class_names,
                "card_detections": [
                    {"class_id": d.class_id, "class_name": d.class_name, "confidence": d.confidence}
                    for d in card_dets
                ],
                "field_detections": [
                    {"class_id": d.class_id, "class_name": d.class_name, "confidence": d.confidence}
                    for d in field_dets
                ],
                "config_class_names": settings.CLASS_NAMES,
            }
        )

    except Exception as e:
        logger.error(f"Debug error: {e}")
        return JSONResponse(content={"error": str(e)})


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": settings.APP_VERSION}


@router.get("/models")
async def model_status():
    """Get status of loaded models."""
    from app.models.ocr_engine import OCREngine

    ocr = OCREngine()
    return {"ocr": ocr.get_available_engines()}
