import os
import logging

import cv2
import planetary_computer
import pystac_client
import pystac
import rasterio
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import traceback
import asyncio
import concurrent.futures
from planetary_computer import sign_inplace
from PIL import Image
import tempfile
import time
from typing import List, Dict, Optional, Tuple, Callable, Union, Protocol, Any
import aiohttp
import io
from contextlib import asynccontextmanager, contextmanager
from queue import Queue, Empty  # Add Empty to imports
import threading
import queue  # Add this import for proper queue exception handling
from functools import lru_cache
import hashlib
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Collection-specific band configurations
BAND_CONFIGS = {
    "sentinel-2-l2a": {
        "rgb": ["B04", "B03", "B02"],  # Red, Green, Blue
        "nir": "B08",  # Near-infrared
        "swir": "B12",  # Short-wave infrared
        "resolution": 10,  # meters
    },
    "landsat-8-c2-l2": {
        "rgb": ["SR_B4", "SR_B3", "SR_B2"],
        "nir": "SR_B5",
        "swir": "SR_B7",
        "resolution": 30,
    },
    "landsat-9-c2-l2": {
        "rgb": ["SR_B4", "SR_B3", "SR_B2"],
        "nir": "SR_B5",
        "swir": "SR_B7",
        "resolution": 30,
    },
}

# Add new constants
DOWNLOAD_TIMEOUT = 60  # seconds
PROCESSING_TIMEOUT = 120  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming


# Add new constants for image processing
class ImageQuality(Enum):
    LOW = "low"  # 1000px max dimension
    MEDIUM = "medium"  # 2000px max dimension
    HIGH = "high"  # 4000px max dimension


@dataclass
class ProcessingOptions:
    quality: ImageQuality = ImageQuality.MEDIUM
    chunk_size: int = 1024
    max_dimension: int = 2000
    cache_enabled: bool = True
    timeout: int = 60


# Update existing constants
PROCESSING_TIMEOUTS = {
    ImageQuality.LOW: 30,  # 30 seconds for low quality
    ImageQuality.MEDIUM: 60,  # 1 minute for medium quality
    ImageQuality.HIGH: 120,  # 2 minutes for high quality
}

QUALITY_DIMENSIONS = {
    ImageQuality.LOW: 1000,
    ImageQuality.MEDIUM: 2000,
    ImageQuality.HIGH: 4000,
}


# Add type definitions for progress callbacks
class ProgressCallback(Protocol):
    def __call__(self, progress: float) -> None: ...


class ThreadSafeProgress:
    """Thread-safe progress tracking"""

    def __init__(self, total_steps: int = 100):
        self.progress = 0
        self.total = total_steps
        self._lock = threading.Lock()
        self.queue = Queue()

    def update(self, increment: float):
        """Thread-safe progress update"""
        with self._lock:
            self.progress = min(self.progress + increment, self.total)
            self.queue.put(self.progress)

    def get_progress(self) -> float:
        """Get current progress"""
        with self._lock:
            return (self.progress / self.total) * 100


@contextmanager
def background_processing_context():
    """Context manager for background processing"""
    # Save current Streamlit context state
    has_st_context = (
        hasattr(st, "script_run_ctx") and st.script_run_ctx.get() is not None
    )

    if has_st_context:
        # Clear Streamlit context for background thread
        old_ctx = st.script_run_ctx.get()
        st.script_run_ctx.set(None)

    try:
        yield
    finally:
        # Restore Streamlit context if it existed
        if has_st_context:
            st.script_run_ctx.set(old_ctx)


# Update the function signature for better type hints
async def download_band_with_progress(
    url: str, progress_tracker: ThreadSafeProgress, timeout: int = DOWNLOAD_TIMEOUT
) -> bytes:
    """Download band data with thread-safe progress tracking"""
    with background_processing_context():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download: HTTP {response.status}")

                total_size = int(response.headers.get("content-length", 0))
                data = io.BytesIO()
                downloaded = 0

                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    data.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        progress_tracker.update(progress)

                return data.getvalue()


def init_planetary_computer(retry_attempts: int = 3) -> Optional[pystac_client.Client]:
    """
    Initialize connection to Microsoft Planetary Computer with retry logic

    Args:
        retry_attempts: Number of connection attempts before failing

    Returns:
        STAC API client or None if connection fails
    """
    for attempt in range(retry_attempts):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=sign_inplace,  # Using correct signing method
            )
            # Test connection by making a simple query
            test_search = catalog.search(collections=["sentinel-2-l2a"], limit=1)
            next(test_search.items())  # Using items() instead of get_items()
            logger.info("Successfully connected to Planetary Computer")
            return catalog
        except Exception as e:
            logger.error(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(2**attempt)  # Exponential backoff
                continue
            st.error(f"Failed to connect to Planetary Computer: {e}")
            st.info("Please check your internet connection and try again")
            return None


def validate_search_params(
    lat: float, lon: float, date_start: str, date_end: str, collection: str
) -> Tuple[bool, str]:
    """Validate search parameters"""
    if not -90 <= lat <= 90:
        return False, "Latitude must be between -90 and 90"
    if not -180 <= lon <= 180:
        return False, "Longitude must be between -180 and 180"
    if collection not in BAND_CONFIGS:
        return False, f"Unsupported collection: {collection}"
    try:
        datetime.fromisoformat(date_start)
        datetime.fromisoformat(date_end)
    except ValueError:
        return False, "Invalid date format"
    return True, ""


def search_satellite_imagery(
    catalog: pystac_client.Client,
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
    collection: str = "sentinel-2-l2a",
    max_cloud_cover: int = 20,
    search_radius_km: float = 10.0,
) -> List[Dict]:
    """
    Search for satellite imagery from MPC

    Args:
        catalog: STAC API client
        lat, lon: Coordinates
        date_start, date_end: Date range (ISO format)
        collection: Satellite collection ID
        max_cloud_cover: Maximum cloud cover percentage
        search_radius_km: Search radius in kilometers
    """
    try:
        # Validate parameters
        valid, error_msg = validate_search_params(
            lat, lon, date_start, date_end, collection
        )
        if not valid:
            st.error(error_msg)
            return []

        # Convert radius from km to degrees (approximate)
        degrees_per_km = 1 / 111  # at equator
        search_radius = search_radius_km * degrees_per_km

        # Create search area
        bbox = [
            lon - search_radius,
            lat - search_radius,
            lon + search_radius,
            lat + search_radius,
        ]

        # Configure query based on collection
        cloud_cover_property = (
            "eo:cloud_cover" if "sentinel" in collection else "landsat:cloud_cover"
        )

        # Search parameters with sorting
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{date_start}/{date_end}",
            query={cloud_cover_property: {"lt": max_cloud_cover}},
            sortby=[{"field": "datetime", "direction": "desc"}],
        )

        # Get items with progress indicator using non-deprecated items() method
        with st.spinner("Fetching satellite imagery..."):
            items = list(search.items())  # Changed from get_items() to items()

        if not items:
            st.info("No images found matching your criteria")
            return []

        logger.info(f"Found {len(items)} images matching criteria")
        return items

    except Exception as e:
        logger.error(f"Error searching satellite imagery: {e}")
        st.error(f"Error searching satellite imagery: {str(e)}")
        traceback.print_exc()
        return []


# Add caching utilities
@lru_cache(maxsize=100)
def get_cached_image_path(item_id: str, quality: ImageQuality) -> Optional[str]:
    """Get cached image path if it exists"""
    cache_dir = os.path.join(tempfile.gettempdir(), "satellite_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{item_id}_{quality.value}.jpg")
    return cache_path if os.path.exists(cache_path) else None


def get_item_hash(item: pystac.Item, options: ProcessingOptions) -> str:
    """Generate unique hash for item and processing options"""
    key = f"{item.id}_{options.quality.value}"
    return hashlib.md5(key.encode()).hexdigest()


# Update the main processing function
def process_satellite_image(
    item: pystac.Item,
    bands: Optional[List[str]] = None,
    options: Optional[ProcessingOptions] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Optional[str]:
    """Process satellite imagery with optimizations"""
    try:
        options = options or ProcessingOptions()

        # Check cache first
        if options.cache_enabled:
            item_hash = get_item_hash(item, options)
            cached_path = get_cached_image_path(item_hash, options.quality)
            if cached_path:
                logger.info(f"Using cached image: {cached_path}")
                return cached_path

        collection = item.collection_id
        if collection not in BAND_CONFIGS:
            raise ValueError(f"Unsupported collection: {collection}")

        bands = bands or BAND_CONFIGS[collection]["rgb"]
        total_steps = len(bands) + 1

        # Use smaller chunks for large images
        chunk_size = min(options.chunk_size, QUALITY_DIMENSIONS[options.quality] // 4)

        # Initialize progress tracking
        progress_tracker = ThreadSafeProgress(total_steps)
        progress_queue = Queue()

        # Download and process bands with optimized chunking
        arrays = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            for band in bands:
                href = item.assets[band].href
                signed_url = sign_inplace(href)

                future = executor.submit(
                    download_and_process_band, signed_url, options, progress_tracker
                )
                futures[future] = band

            # Wait with timeout
            timeout = PROCESSING_TIMEOUTS[options.quality]
            done, pending = concurrent.futures.wait(
                futures.keys(),
                timeout=timeout,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )

            # Cancel pending tasks
            for future in pending:
                future.cancel()
                logger.warning(f"Cancelled download for band {futures[future]}")

            if pending:
                raise TimeoutError(
                    f"Processing timeout after {timeout}s. "
                    f"Try using a lower quality setting."
                )

            # Process completed downloads
            for future in done:
                if future.exception():
                    raise future.exception()
                arrays.append(future.result())

        # Stack and process bands
        img_array = process_bands(
            arrays, options.quality, chunk_size, progress_callback
        )

        # Save result
        output_path = save_processed_image(
            img_array, item_hash if options.cache_enabled else None, options
        )

        return output_path

    except Exception as e:
        logger.error(f"Error processing satellite image: {e}")
        raise


async def download_and_process_band(
    url: str, options: ProcessingOptions, progress_tracker: ThreadSafeProgress
) -> np.ndarray:
    """Download and process a single band with optimizations"""
    with background_processing_context():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Download failed: HTTP {response.status}")

                # Stream data in chunks
                data = io.BytesIO()
                total_size = int(response.headers.get("content-length", 0))
                received = 0

                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    data.write(chunk)
                    received += len(chunk)
                    if total_size:
                        progress = (received / total_size) * 100
                        progress_tracker.update(progress)

                # Process band data
                with rasterio.MemoryFile(data.getvalue()) as memfile:
                    with memfile.open() as src:
                        # Read with decimation factor based on quality
                        scale = calculate_scale_factor(
                            src.shape, QUALITY_DIMENSIONS[options.quality]
                        )
                        array = src.read(
                            1,
                            out_shape=(int(src.height * scale), int(src.width * scale)),
                        )
                        return array


def process_bands(
    arrays: List[np.ndarray],
    quality: ImageQuality,
    chunk_size: int,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """Process bands with chunked operations"""
    if not arrays:
        raise ValueError("No arrays to process")

    # Ensure all arrays have the same shape
    target_shape = arrays[0].shape
    arrays = [ensure_shape(arr, target_shape) for arr in arrays]

    # Stack bands
    img_array = np.dstack(arrays)
    img_array = img_array.astype(np.float32)

    # Process in chunks
    total_chunks = (
        (img_array.shape[0] + chunk_size - 1)
        // chunk_size
        * (img_array.shape[1] + chunk_size - 1)
        // chunk_size
    )
    chunk_count = 0

    for i in range(0, img_array.shape[0], chunk_size):
        for j in range(0, img_array.shape[1], chunk_size):
            chunk = img_array[i : i + chunk_size, j : j + chunk_size]

            # Process chunk
            chunk = enhance_contrast_chunk(chunk)
            img_array[i : i + chunk_size, j : j + chunk_size] = chunk

            # Update progress
            chunk_count += 1
            if progress_callback:
                progress_callback(chunk_count / total_chunks * 100)

    return img_array.astype(np.uint8)


# Add helper functions for image processing
def calculate_scale_factor(shape: Tuple[int, int], target_size: int) -> float:
    """Calculate scale factor to resize image to target size"""
    return min(target_size / max(shape), 1.0)


def enhance_contrast_chunk(chunk: np.ndarray) -> np.ndarray:
    """Enhance contrast for image chunk"""
    for i in range(chunk.shape[2]):
        band = chunk[:, :, i]
        valid_pixels = band[band > 0]
        if len(valid_pixels) > 0:
            p2, p98 = np.percentile(valid_pixels, (2, 98))
            chunk[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
    return chunk


def ensure_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Ensure array has target shape"""
    if arr.shape != target_shape:
        return cv2.resize(arr, target_shape[::-1])
    return arr


def save_processed_image(
    img_array: np.ndarray,
    cache_id: Optional[str] = None,
    options: ProcessingOptions = ProcessingOptions(),
) -> str:
    """Save processed image with optional caching"""
    img = Image.fromarray(img_array)

    if cache_id and options.cache_enabled:
        # Save to cache
        cache_dir = os.path.join(tempfile.gettempdir(), "satellite_cache")
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, f"{cache_id}_{options.quality.value}.jpg")
    else:
        # Save to temporary file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        output_path = output_file.name
        output_file.close()

    # Save with appropriate quality
    img.save(output_path, "JPEG", quality=85, optimize=True, progressive=True)

    return output_path


def create_preview_image(
    item: pystac.Item,
    bands: Optional[List[str]] = None,
    size: Tuple[int, int] = (300, 300),
) -> Optional[Image.Image]:
    """Create a preview image from satellite data"""
    try:
        collection = item.collection_id
        if collection not in BAND_CONFIGS:
            raise ValueError(f"Unsupported collection: {collection}")

        if bands is None:
            bands = BAND_CONFIGS[collection]["rgb"]

        # Read and stack bands
        arrays = []
        for band in bands:
            href = item.assets[band].href
            signed_url = sign_inplace(href)
            with rasterio.open(signed_url) as src:
                array = src.read(1)
                arrays.append(array)

        # Stack and normalize
        img_array = np.dstack(arrays)
        img_array = img_array.astype(np.float32)

        # Quick contrast enhancement
        for i in range(img_array.shape[2]):
            band = img_array[:, :, i]
            p2, p98 = np.percentile(band, (2, 98))
            img_array[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)

        img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)

        # Resize for preview
        img.thumbnail(size)
        return img

    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        return None


def get_available_collections() -> Dict[str, str]:
    """Return list of available satellite collections with descriptions"""
    return {
        "sentinel-2-l2a": "Sentinel-2 Level 2A (10m resolution)",
        "landsat-8-c2-l2": "Landsat 8 Collection 2 Level 2 (30m resolution)",
        "landsat-9-c2-l2": "Landsat 9 Collection 2 Level 2 (30m resolution)",
    }
