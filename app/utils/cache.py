"""
LRU Cache with TTL for OCR results.
Thread-safe caching to avoid redundant OCR processing on identical field crops.
"""

import time
import hashlib
import numpy as np
import cv2
from typing import Optional, Any, Dict
from collections import OrderedDict
from threading import Lock

from app.core.logger import logger


class TTLCache:
    """Thread-safe LRU cache with time-to-live for OCR results."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 30):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries to keep
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _image_hash(self, image: np.ndarray) -> str:
        """
        Fast perceptual hash for image comparison.
        Uses resized grayscale image for quick comparison.
        """
        try:
            # Resize to small fixed size for fast hashing
            resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            gray = (
                cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if len(resized.shape) == 3
                else resized
            )
            return hashlib.md5(gray.tobytes()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to compute image hash: {e}")
            # Fallback to simple hash
            return hashlib.md5(image.tobytes()).hexdigest()[:16]

    def get(self, image: np.ndarray, field_name: str) -> Optional[Any]:
        """
        Get cached OCR result for an image field.

        Args:
            image: Field image numpy array
            field_name: Name of the field (e.g., 'nid', 'firstName')

        Returns:
            Cached OCR result or None if not found/expired
        """
        key = f"{field_name}:{self._image_hash(image)}"

        with self._lock:
            if key in self._cache:
                age = time.time() - self._timestamps[key]
                if age < self._ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug(f"Cache HIT for {field_name} (age: {age:.1f}s)")
                    return self._cache[key]
                else:
                    # Expired - remove
                    del self._cache[key]
                    del self._timestamps[key]
                    logger.debug(f"Cache EXPIRED for {field_name}")

            self._misses += 1
            return None

    def set(self, image: np.ndarray, field_name: str, value: Any):
        """
        Cache an OCR result.

        Args:
            image: Field image numpy array
            field_name: Name of the field
            value: OCR result to cache
        """
        key = f"{field_name}:{self._image_hash(image)}"

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)

            self._cache[key] = value
            self._timestamps[key] = time.time()

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
                logger.debug(f"Cache EVICTED oldest entry")

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl_seconds": self._ttl,
            }


# Global cache instance for OCR results
ocr_cache = TTLCache(max_size=256, ttl_seconds=30)


def get_cache_stats() -> Dict[str, Any]:
    """Get current cache statistics."""
    return ocr_cache.stats()


def clear_cache():
    """Clear the OCR cache."""
    ocr_cache.clear()
