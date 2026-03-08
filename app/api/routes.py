"""
API Routes for Egyptian ID OCR Service
Provides endpoints for ID card extraction and visualization.

Supports:
- Single image upload (front OR back)
- Multi-image upload (front AND back as separate files)
- Automatic side detection and routing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse
from dataclasses import asdict
import base64
import numpy as np
import cv2
import io
import time
from pathlib import Path
from typing import List, Optional

from app.services.pipeline import IDExtractionPipeline
from app.services.ocr_visualizer import OCRVisualizer, VisualizationResult
from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.utils.image_utils import preprocess_text_field
from app.utils.text_utils import clean_field, _is_valid_nid_format
from app.core.config import settings
from app.core.logger import logger

router = APIRouter(prefix="/api/v1", tags=["OCR"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 10


@router.post("/extract")
async def extract_id(
    file: UploadFile = File(..., description="ID card image (JPEG, PNG, WebP, BMP)"),
    side: Optional[str] = Form(None, description="Optional hint: 'front' or 'back'")
):
    """
    Extract information from an Egyptian ID card image.
    
    Automatically detects if the image contains:
    - Front side only
    - Back side only
    - Both sides (dual-side image)
    
    Args:
        file: ID card image file
        side: Optional hint for side detection ('front' or 'back')
    
    Returns:
        Extracted data with confidence scores
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file.content_type}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_SIZE_MB}MB)")
    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="File too small")

    logger.info(f"Processing file: {file.filename}, side_hint={side}")

    pipeline = IDExtractionPipeline()
    result = pipeline.process(contents)

    return JSONResponse(content=result)


@router.post("/extract-multi")
async def extract_id_multi(
    front: Optional[UploadFile] = File(None, description="Front side image"),
    back: Optional[UploadFile] = File(None, description="Back side image"),
    files: Optional[List[UploadFile]] = File(None, description="Multiple images (auto-detected)")
):
    """
    Extract information from multiple ID card images.
    
    Supports:
    - Front + Back as separate files
    - Multiple images (auto-ordered)
    
    Args:
        front: Front side image (optional)
        back: Back side image (optional)
        files: List of images (alternative to front/back)
    
    Returns:
        Merged extracted data with cross-validation info
    """
    front_bytes = None
    back_bytes = None
    
    # Handle explicit front/back upload
    if front:
        if front.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Front: Unsupported format")
        front_bytes = await front.read()
        
    if back:
        if back.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Back: Unsupported format")
        back_bytes = await back.read()
    
    # Handle multiple files upload
    if files and len(files) > 0:
        if len(files) > 2:
            raise HTTPException(status_code=400, detail="Maximum 2 images supported")
        
        # Sort files by name to determine order (front should come first alphabetically)
        sorted_files = sorted(files, key=lambda f: f.filename or "")
        
        if len(sorted_files) >= 1 and not front_bytes:
            if sorted_files[0].content_type not in ALLOWED_TYPES:
                raise HTTPException(status_code=400, detail=f"File 1: Unsupported format")
            front_bytes = await sorted_files[0].read()
        
        if len(sorted_files) >= 2 and not back_bytes:
            if sorted_files[1].content_type not in ALLOWED_TYPES:
                raise HTTPException(status_code=400, detail=f"File 2: Unsupported format")
            back_bytes = await sorted_files[1].read()
    
    # Validate at least one image provided
    if not front_bytes and not back_bytes:
        raise HTTPException(
            status_code=400, 
            detail="At least one image required (front, back, or files)"
        )
    
    # Validate file sizes
    for name, data in [("front", front_bytes), ("back", back_bytes)]:
        if data:
            size_mb = len(data) / (1024 * 1024)
            if size_mb > MAX_SIZE_MB:
                raise HTTPException(status_code=413, detail=f"{name} file too large")
            if len(data) < 1000:
                raise HTTPException(status_code=400, detail=f"{name} file too small")
    
    logger.info(f"Processing multi-image: front={bool(front_bytes)}, back={bool(back_bytes)}")
    
    pipeline = IDExtractionPipeline()
    result = pipeline.process_multi_image(front_bytes=front_bytes, back_bytes=back_bytes)
    
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

        # Detect fields (try ONNX first, then YOLO fallback)
        field_dets = []
        if hasattr(detector.field_detector, 'session') and detector.field_detector.session is not None:
            field_dets = detector.field_detector.detect(card_img, conf_threshold=0.3)
        elif hasattr(detector.field_detector_fallback, 'model') and detector.field_detector_fallback.model is not None:
            field_dets = detector.field_detector_fallback.detect(card_img)

        # Color for each class
        colors = {
            "firstName": (0, 255, 0),
            "lastName": (0, 255, 128),
            "nid": (0, 128, 255),
            "serial": (255, 255, 0),
            "address": (255, 0, 255),
            "front_nid": (0, 128, 255),
            "back_nid": (0, 128, 255),
            "serial_num": (255, 255, 0),
        }

        for det in field_dets:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (0, 255, 0))
            cv2.rectangle(card_img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} ({det.confidence:.2f})"
            cv2.putText(card_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Combine - ensure same height
        h, w = result_image.shape[:2]
        card_h, card_w = card_img.shape[:2]

        # Resize card image to match width and maintain aspect ratio
        if card_w > 0:
            scale = w / card_w
            card_img = cv2.resize(card_img, (w, int(card_h * scale)), interpolation=cv2.INTER_AREA)

        # Ensure both images have same height
        result_h = result_image.shape[0]
        card_h = card_img.shape[0]
        if result_h != card_h:
            # Resize the shorter one
            if result_h > card_h:
                card_img = cv2.resize(card_img, (w, result_h))
            else:
                result_image = cv2.resize(result_image, (w, card_h))

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
        # Get all detections from ONNX detector (primary)
        card_dets = detector.card_detector.detect(image)

        # Get field detections from ONNX detector
        field_dets = []
        if hasattr(detector.field_detector, 'session') and detector.field_detector.session is not None:
            # ONNX detector
            field_dets = detector.field_detector.detect(image, conf_threshold=0.3)
        elif hasattr(detector.field_detector_fallback, 'model') and detector.field_detector_fallback.model is not None:
            # YOLO fallback
            field_dets = detector.field_detector_fallback.detect(image)

        # Get model class names
        onnx_classes = settings.ONNX_FIELD_DETECTOR_CLASSES if hasattr(settings, 'ONNX_FIELD_DETECTOR_CLASSES') else {}
        yolo_classes = settings.CLASS_NAMES if hasattr(settings, 'CLASS_NAMES') else {}

        return JSONResponse(
            content={
                "onnx_classes": onnx_classes,
                "yolo_classes": yolo_classes,
                "card_detections": [
                    {"class_id": d.class_id, "class_name": d.class_name, "confidence": round(d.confidence, 3), "bbox": d.bbox}
                    for d in card_dets
                ],
                "field_detections": [
                    {"class_id": d.class_id, "class_name": d.class_name, "confidence": round(d.confidence, 3), "bbox": d.bbox}
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


@router.post("/debug/ocr-steps")
async def debug_ocr_steps(
    file: UploadFile = File(...),
    field_name: str = Query(default="nid", description="Field to visualize (nid, firstName, lastName, address, serial)"),
    save_images: bool = Query(default=False, description="Save step images to debug folder")
):
    """
    Detailed OCR visualization endpoint.
    Shows each step of the OCR process with overlays.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported format")

    contents = await file.read()

    # Decode image
    arr = np.frombuffer(contents, np.uint8)
    original_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if original_image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    start_time = time.time()
    visualizer = OCRVisualizer()
    steps = []
    ocr_results_data = {}

    try:
        # Initialize pipeline
        pipeline = IDExtractionPipeline()

        # Step 1: Card Detection
        card_detected = False
        card_bbox = None
        card_image = original_image

        if pipeline._detector:
            card_dets = pipeline._detector.card_detector.detect(original_image)
            if card_dets:
                card_detected = True
                best_card = max(card_dets, key=lambda d: d.confidence)
                card_bbox = best_card.bbox
                card_image = pipeline._detector.crop_card(original_image)

        steps.append(visualizer.visualize_card_detection(original_image, card_bbox, card_detected))

        # Step 2: Field Detection
        fields = {}
        if pipeline._detector:
            fields = pipeline._detector.crop_fields(card_image)

        if fields:
            field_order = [field_name] + [f for f in fields.keys() if f != field_name]
            field_steps = visualizer.visualize_field_detection(card_image, fields, field_order)
            steps.extend(field_steps)
        else:
            # Fallback to template ROIs
            from app.utils.image_utils import (
                resize_to_standard, remove_noise, enhance_contrast,
                auto_detect_and_warp, remove_background_lines,
                detect_card_side, extract_all_rois
            )
            normalized = resize_to_standard(card_image, target_width=settings.TARGET_IMAGE_WIDTH)
            normalized = remove_noise(normalized)
            normalized = enhance_contrast(normalized)
            normalized = auto_detect_and_warp(normalized)
            normalized = remove_background_lines(normalized)
            side = detect_card_side(normalized)
            template_rois = extract_all_rois(normalized, side=side)
            fields = {k: (v, 0.5) for k, v in template_rois.items()}

            if field_name in fields:
                field_steps = visualizer.visualize_field_detection(normalized, fields, [field_name])
                steps.extend(field_steps)

        # Step 3: Preprocessing and OCR for specific field
        if field_name in fields:
            field_img, det_conf = fields[field_name]

            # Generate preprocessing variations for NID
            if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                from app.utils.ocr_preprocess import preprocess_nid_variations
                variations = preprocess_nid_variations(field_img)
                var_dict = {
                    "binary": variations[0] if len(variations) > 0 else field_img,
                    "inverted": variations[1] if len(variations) > 1 else field_img,
                    "clahe": variations[2] if len(variations) > 2 else field_img,
                    "otsu": variations[3] if len(variations) > 3 else field_img,
                }
                prep_steps = visualizer.visualize_preprocessing_variations(field_img, var_dict, field_name)
                steps.extend(prep_steps)

            # Run OCR
            processed = preprocess_text_field(field_img, field_type=field_name)

            if pipeline._ocr:
                # Tesseract NID variations
                if field_name in ["nid", "front_nid", "back_nid", "id_number"] and pipeline._ocr._tesseract and pipeline._ocr._tesseract._client:
                    tesseract_result = pipeline._ocr._tesseract.run_nid_tesseract(field_img)
                    ocr_results_data["tesseract"] = {
                        "text": tesseract_result.text,
                        "confidence": tesseract_result.confidence,
                        "engine": "Tesseract ara_number_id"
                    }
                    steps.append(visualizer.visualize_ocr_result(
                        field_img, [], tesseract_result.text,
                        tesseract_result.confidence, "Tesseract"
                    ))

                # EasyOCR NID
                if pipeline._ocr._easy:
                    easy_result = pipeline._ocr._easy.run_nid(field_img)
                    cleaned = clean_field(easy_result.text, field_name)
                    ocr_results_data["easyocr"] = {
                        "text": cleaned,
                        "confidence": easy_result.confidence,
                        "engine": "EasyOCR"
                    }
                    steps.append(visualizer.visualize_ocr_result(
                        field_img, [], cleaned,
                        easy_result.confidence, "EasyOCR"
                    ))

                # Ensemble voting for NID
                if field_name in ["nid", "front_nid", "back_nid", "id_number"]:
                    tesseract_text = ocr_results_data.get("tesseract", {}).get("text", "")
                    easy_text = ocr_results_data.get("easyocr", {}).get("text", "")

                    if len(tesseract_text) == 14 and len(easy_text) == 14:
                        voted = visualizer._vote_nid_results([tesseract_text, easy_text])
                        steps.append(visualizer.visualize_nid_ensemble(
                            field_img, tesseract_text, easy_text, voted,
                            ocr_results_data.get("tesseract", {}).get("confidence", 0),
                            ocr_results_data.get("easyocr", {}).get("confidence", 0)
                        ))

        # Save images if requested
        if save_images:
            visualizer.save_step_images(steps, f"ocr_{field_name}_{int(time.time())}")

        processing_ms = int((time.time() - start_time) * 1000)

        return JSONResponse(content={
            "field_name": field_name,
            "total_steps": len(steps),
            "processing_time_ms": processing_ms,
            "steps": [asdict(step) for step in steps],
            "ocr_results": ocr_results_data,
            "debug_info": {
                "card_detected": card_detected,
                "fields_detected": len(fields),
                "image_size": {"width": original_image.shape[1], "height": original_image.shape[0]}
            }
        })

    except Exception as e:
        logger.error(f"Debug OCR steps error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug/classify-side")
async def debug_classify_side(file: UploadFile = File(...)):
    """
    Debug endpoint to test side classification.
    Returns classification result with confidence and details.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported format")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    from app.services.side_classifier import get_side_classifier
    
    classifier = get_side_classifier()
    result = classifier.classify(image)
    
    return JSONResponse(content={
        "side": result.side.value,
        "confidence": round(result.confidence, 3),
        "details": result.details,
        "image_size": {"width": image.shape[1], "height": image.shape[0]}
    })


@router.get("/debug/ocr-steps/{step_index}/image")
async def get_step_image(
    step_index: int,
    file: UploadFile = File(...),
    field_name: str = Query(default="nid")
):
    """
    Get a specific step image from OCR visualization.
    """
    raise HTTPException(status_code=400, detail="Use /debug/ocr-steps endpoint to get all steps")


@router.post("/debug/test-card-detection")
async def test_card_detection(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(default=0.3, ge=0.1, le=0.9, description="YOLO confidence threshold")
):
    """
    Test card and field detection with adjustable confidence threshold.
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
        # Temporarily adjust threshold
        original_threshold = settings.YOLO_CONF_THRESHOLD
        settings.YOLO_CONF_THRESHOLD = confidence_threshold

        # Detect card
        card_dets = detector.card_detector.detect(image)
        card_detected = len([d for d in card_dets if d.class_name == "id_card" or d.class_id == 0]) > 0
        card_image = detector.crop_card(image.copy()) if card_detected else image

        # Detect fields on card image
        field_dets = detector.field_detector.detect(card_image)

        # Restore threshold
        settings.YOLO_CONF_THRESHOLD = original_threshold

        # Create visualization
        vis_image = image.copy()

        # Draw card detection
        for det in card_dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            label = f"Card ({det.confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw field detections
        colors = {
            "firstName": (0, 255, 0),
            "lastName": (0, 255, 128),
            "nid": (0, 128, 255),
            "id_number": (0, 128, 255),
            "serial": (255, 255, 0),
            "address": (255, 0, 255),
        }

        for det in field_dets:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (128, 128, 128))
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} ({det.confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Encode and return
        _, buffer = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return JSONResponse(content={
            "image_base64": base64.b64encode(buffer).decode('utf-8'),
            "card_detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 3), "bbox": d.bbox}
                for d in card_dets
            ],
            "field_detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 3), "bbox": d.bbox}
                for d in field_dets
            ],
            "threshold_used": confidence_threshold,
            "card_detected": card_detected,
            "field_count": len(field_dets),
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        })

    except Exception as e:
        logger.error(f"Test card detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
