"""
PaddleOCR-VL-1.5 Engine for Egyptian ID OCR
Vision-Language Model based OCR for enhanced Arabic text recognition.

NOTE: This module is optional. If PaddleOCR-VL-1.5 is not available,
the system will gracefully fall back to other OCR engines.
"""

import re
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from app.core.logger import logger


@dataclass
class PaddleVLResult:
    """Result from PaddleOCR-VL-1.5 operation."""

    text: str
    confidence: float
    blocks: List[Dict]  # Structured blocks with text, bbox, confidence
    latency_ms: int
    engine_used: str = "paddle_vl"


# Try to import PaddleOCR-VL, but make it optional
PADDLE_VL_PIPELINE = None
try:
    from paddleocr import PaddleOCRVL as _PaddleOCRVL

    PADDLE_VL_PIPELINE = _PaddleOCRVL
    logger.info("PaddleOCRVL module loaded successfully")
except ImportError as e:
    logger.warning(f"PaddleOCR-VL not available (optional): {e}")
except Exception as e:
    logger.warning(f"Failed to load PaddleOCR-VL: {e}")


class PaddleVLEngine:
    """
    PaddleOCR-VL-1.5 integration for Egyptian ID OCR.

    This VLM-based OCR provides:
    - Better Arabic text understanding through vision-language context
    - Dynamic resolution handling for small text
    - Robustness to document distortions

    Use cases:
    - Arabic names and addresses (primary use)
    - Complex/mixed language fields
    - When traditional OCR fails as fallback

    NOTE: This is optional. If not available, the system falls back to
    PaddleOCR Arabic or EasyOCR.
    """

    _instance = None
    _initialized = False
    _pipeline = None

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize PaddleOCR-VL-1.5 engine."""
        if self._initialized:
            return

        logger.info("Initializing PaddleOCR-VL-1.5 engine...")

        # Check if the pipeline class is available
        global PADDLE_VL_PIPELINE
        if PADDLE_VL_PIPELINE is None:
            logger.warning("PaddleOCRVL not available - will use fallback OCR engines")
            self._initialized = True
            return

        # Check device availability
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PaddleOCR-VL-1.5 will use device: {self.device}")
        except ImportError:
            self.device = "cpu"
            logger.warning("PyTorch not available, using CPU")

        # Try to initialize the pipeline
        self._pipeline = self._init_pipeline()
        self._initialized = True

    def _init_pipeline(self):
        """Initialize the PaddleOCR-VL-1.5 pipeline."""
        try:
            # Initialize with optimized settings for document OCR
            # Note: Only use arguments supported by the installed PaddleOCR version
            pipeline_kwargs = {}
            
            # Try to use doc orientation detection if available
            try:
                pipeline_kwargs['use_doc_orientation_det'] = False
            except Exception:
                pass
                
            # Try to use doc unwarping if available
            try:
                pipeline_kwargs['use_doc_unwarping'] = False
            except Exception:
                pass

            # Initialize the pipeline with available arguments
            pipeline = PADDLE_VL_PIPELINE(**pipeline_kwargs)

            logger.info("PaddleOCR-VL-1.5 pipeline initialized successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR-VL-1.5: {e}")
            return None

    def available(self) -> bool:
        """Check if PaddleOCR-VL-1.5 is available."""
        return self._pipeline is not None

    def _convert_image(self, image_np: np.ndarray) -> np.ndarray:
        """Convert BGR numpy array to RGB."""
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # BGR to RGB
            return image_np[:, :, ::-1]
        return image_np

    def run_ocr(self, image_np: np.ndarray) -> PaddleVLResult:
        """
        Run OCR on the given image using PaddleOCR-VL-1.5.

        Args:
            image_np: Image as numpy array (BGR or grayscale)

        Returns:
            PaddleVLResult with extracted text and metadata
        """
        t0 = time.time()

        if self._pipeline is None:
            return PaddleVLResult(
                text="",
                confidence=0.0,
                blocks=[],
                latency_ms=0,
            )

        try:
            # Convert to RGB if needed
            image_rgb = self._convert_image(image_np)

            # Run OCR
            result = self._pipeline.predict(image_rgb)

            # Parse result
            texts = []
            blocks = []
            confs = []

            # PaddleOCR-VL returns a list of result items
            if result and isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        score = item.get("score", 0.0)
                        bbox = item.get("bbox", [])

                        if text and isinstance(text, str) and text.strip():
                            texts.append(text.strip())
                            confs.append(float(score))
                            blocks.append(
                                {
                                    "text": text.strip(),
                                    "confidence": float(score),
                                    "bbox": bbox,
                                }
                            )

            full_text = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            return PaddleVLResult(
                text=full_text,
                confidence=avg_conf,
                blocks=blocks,
                latency_ms=int((time.time() - t0) * 1000),
            )

        except Exception as e:
            logger.error(f"PaddleOCR-VL-1.5 OCR error: {e}")
            return PaddleVLResult(
                text="",
                confidence=0.0,
                blocks=[],
                latency_ms=int((time.time() - t0) * 1000),
            )

    def run_arabic(self, image_np: np.ndarray) -> PaddleVLResult:
        """
        Optimized for Arabic text recognition (names, addresses).

        PaddleOCR-VL-1.5 handles Arabic well natively through its
        multilingual VLM capabilities.
        """
        result = self.run_ocr(image_np)
        logger.debug(
            f"PaddleOCR-VL Arabic: '{result.text}' (conf={result.confidence:.2f}, "
            f"latency={result.latency_ms}ms)"
        )
        return result

    def run_digits(self, image_np: np.ndarray) -> PaddleVLResult:
        """
        Extract digits only - useful for NID and serial number fields.

        Returns:
            PaddleVLResult with only digit characters
        """
        result = self.run_ocr(image_np)

        # Filter to digits only (European and Arabic-Indic)
        digit_text = re.sub(r"[^\d٠-٩]", "", result.text)

        # Convert Arabic-Indic digits to European digits
        if digit_text:
            arabic_indic = "٠١٢٣٤٥٦٧٨٩"
            european = "0123456789"
            translation = str.maketrans(arabic_indic, european)
            digit_text = digit_text.translate(translation)

        return PaddleVLResult(
            text=digit_text,
            confidence=result.confidence,
            blocks=result.blocks,
            latency_ms=result.latency_ms,
        )

    def run_mixed(self, image_np: np.ndarray) -> PaddleVLResult:
        """
        Run on mixed content fields (Arabic + English + digits).

        This is the default mode and works well for most ID fields.
        """
        return self.run_ocr(image_np)

    def get_block_text(self, image_np: np.ndarray, block_index: int = 0) -> str:
        """
        Get text from a specific block (useful for structured fields).

        Args:
            image_np: Input image
            block_index: Which block to extract (0 = first)

        Returns:
            Text from the specified block, or empty string
        """
        result = self.run_ocr(image_np)
        if 0 <= block_index < len(result.blocks):
            return result.blocks[block_index].get("text", "")
        return result.text


class PaddleVLRecognitionOnly:
    """
    Alternative: Use transformers library directly for PaddleOCR-VL-1.5.

    This provides more control over the inference process and can be
    faster in some scenarios when using GPU.
    """

    _instance = None
    _initialized = False
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing PaddleOCR-VL-1.5 via transformers...")

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model on {device}...")

            # Load model and processor
            model_path = "PaddlePaddle/PaddleOCR-VL-1.5"

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            # Use bfloat16 for better performance on compatible GPUs
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)

            self._model.eval()
            self._device = device

            logger.info(f"PaddleOCR-VL-1.5 transformers model ready on {device}")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install 'transformers>=5.0.0'")
            self._model = None
            self._processor = None
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL-1.5 model: {e}")
            self._model = None
            self._processor = None

        self._initialized = True

    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    def run_ocr(
        self, image_np: np.ndarray, task: str = "ocr", max_tokens: int = 512
    ) -> PaddleVLResult:
        """
        Run OCR using transformers model.

        Args:
            image_np: Input image (BGR)
            task: Task type - 'ocr', 'table', 'chart', 'formula', 'seal'
            max_tokens: Maximum tokens to generate

        Returns:
            PaddleVLResult
        """
        t0 = time.time()

        if not self.available():
            return PaddleVLResult(
                text="",
                confidence=0.0,
                blocks=[],
                latency_ms=0,
            )

        try:
            from PIL import Image

            # Convert BGR to RGB
            if len(image_np.shape) == 3:
                image_rgb = image_np[:, :, ::-1]
            else:
                image_rgb = image_np

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Prepare prompt
            prompts = {
                "ocr": "OCR:",
                "table": "Table Recognition:",
                "formula": "Formula Recognition:",
                "chart": "Chart Recognition:",
                "seal": "Seal Recognition:",
                "spotting": "Text Spotting:",
            }
            prompt = prompts.get(task, prompts["ocr"])

            # Process inputs
            inputs = self._processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            # Decode
            generated_text = self._processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]

            # Extract text after prompt
            if prompt in generated_text:
                text = generated_text.split(prompt, 1)[1].strip()
            else:
                text = generated_text.strip()

            return PaddleVLResult(
                text=text,
                confidence=0.9,  # VLM doesn't provide per-result confidence
                blocks=[],
                latency_ms=int((time.time() - t0) * 1000),
            )

        except Exception as e:
            logger.error(f"PaddleOCR-VL-1.5 transformers error: {e}")
            return PaddleVLResult(
                text="",
                confidence=0.0,
                blocks=[],
                latency_ms=int((time.time() - t0) * 1000),
            )

    def run_arabic(self, image_np: np.ndarray) -> PaddleVLResult:
        """Run Arabic OCR."""
        return self.run_ocr(image_np, task="ocr")

    def run_digits(self, image_np: np.ndarray) -> PaddleVLResult:
        """Run digit extraction."""
        result = self.run_ocr(image_np, task="ocr")

        # Filter to digits
        digit_text = re.sub(r"[^\d٠-٩]", "", result.text)

        # Convert Arabic-Indic to European
        if digit_text:
            arabic_indic = "٠١٢٣٤٥٦٧٨٩"
            european = "0123456789"
            translation = str.maketrans(arabic_indic, european)
            digit_text = digit_text.translate(translation)

        return PaddleVLResult(
            text=digit_text,
            confidence=result.confidence,
            blocks=result.blocks,
            latency_ms=result.latency_ms,
        )
