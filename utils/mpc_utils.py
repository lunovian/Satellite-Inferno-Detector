import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Callable
import requests
from io import BytesIO
import numpy as np
import pystac
import pystac_client
import streamlit as st
from PIL import Image, ImageFile
from planetary_computer import sign_inplace
from pystac.extensions.eo import EOExtension
from dataclasses import dataclass
from enum import Enum
import cv2  # Add cv2 to imports
from functools import lru_cache
import hashlib
from api.utils import TaskManager

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Simplified collection configuration
SUPPORTED_COLLECTIONS = {
    "sentinel-2-l2a": {
        "name": "Sentinel-2 Level 2A",
        "resolution": "10m",
        "preview_assets": [
            ("thumbnail", 100),  # Small JPEG thumbnail (~100KB)
            ("preview", 2048),  # Medium JPEG preview (~2MB)
            ("visual", None),  # Full resolution RGB TIFF
        ],
    },
    "landsat-c2-l2": {
        "name": "Landsat 8/9 Collection 2 Level-2",
        "resolution": "30m",
        "preview_assets": [
            ("rendered_preview", 1024),  # RGB JPEG preview
            ("preview", 2048),  # Alternative preview
            ("thumbnail", 100),  # Thumbnail
        ],
    },
}


# Add new constants and configuration
class ImageQuality(Enum):
    LOW = "low"  # 2000px max dimension
    MEDIUM = "medium"  # 4000px max dimension
    HIGH = "high"  # 8000px max dimension
    FULL = "full"  # No resizing


@dataclass
class ImageProcessingConfig:
    max_dimension: int = 4000  # Default to MEDIUM quality
    quality_preset: ImageQuality = ImageQuality.MEDIUM
    jpeg_quality: int = 85
    allow_huge_images: bool = False
    target_file_size_mb: float = 50.0

    @classmethod
    def from_quality_preset(cls, quality: ImageQuality) -> "ImageProcessingConfig":
        """Create config from quality preset"""
        dimension_map = {
            ImageQuality.LOW: 2000,
            ImageQuality.MEDIUM: 4000,
            ImageQuality.HIGH: 8000,
            ImageQuality.FULL: None,
        }
        return cls(
            max_dimension=dimension_map[quality],
            quality_preset=quality,
            allow_huge_images=(quality == ImageQuality.FULL),
        )


# Update timeout constants
DOWNLOAD_TIMEOUT = 120  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming

# Add caching configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), "satellite_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


@lru_cache(maxsize=100)
def get_cached_preview(item_id: str, asset_key: str) -> Optional[str]:
    """Get cached preview path if it exists"""
    cache_key = hashlib.md5(f"{item_id}_{asset_key}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.jpg")
    return cache_path if os.path.exists(cache_path) else None


def init_planetary_computer() -> Optional[pystac_client.Client]:
    """Initialize connection to Microsoft Planetary Computer"""
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=sign_inplace,
        )
        # Test connection with a simple query
        catalog.search(collections=["sentinel-2-l2a"], limit=1)
        return catalog
    except Exception as e:
        logger.error(f"Failed to connect to Planetary Computer: {e}")
        st.error(
            "Failed to connect to Microsoft Planetary Computer. Please check your internet connection."
        )
        return None


def search_satellite_imagery(
    catalog: pystac_client.Client,
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
    collection: str = "sentinel-2-l2a",
    max_cloud_cover: int = 20,
    expand_aoi_km: float = 5.0,
) -> List[pystac.Item]:
    """Search for satellite imagery using simplified bbox approach"""
    try:
        # Convert km to approximate degrees (at equator)
        km_to_deg = 1 / 111
        expand_deg = expand_aoi_km * km_to_deg

        # Create bounding box around point
        bbox = [
            lon - expand_deg,
            lat - expand_deg,
            lon + expand_deg,
            lat + expand_deg,
        ]

        # Search for items
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )

        items = list(search.items())

        # Sort by cloud cover
        items.sort(key=lambda x: EOExtension.ext(x).cloud_cover or 100)

        return items[:5]  # Return top 5 clearest images

    except Exception as e:
        logger.error(f"Error searching satellite imagery: {e}")
        return []


def safely_open_large_image(
    image_data: BytesIO, config: ImageProcessingConfig
) -> Optional[Image.Image]:
    """Safely open and optionally resize large images"""
    try:
        # If huge images are allowed, disable DecompressionBomb checks
        if config.allow_huge_images:
            Image.MAX_IMAGE_PIXELS = None

        # Open image
        img = Image.open(image_data)

        # Handle resizing if needed
        if config.max_dimension and (
            img.width > config.max_dimension or img.height > config.max_dimension
        ):
            logger.info(
                f"Resizing image from {img.width}x{img.height} to fit within {config.max_dimension}px"
            )

            # Calculate aspect ratio preserving dimensions
            ratio = min(
                config.max_dimension / img.width, config.max_dimension / img.height
            )
            new_size = (int(img.width * ratio), int(img.height * ratio))

            # Use LANCZOS for best quality downsampling
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    except Exception as e:
        logger.error(f"Error processing large image: {str(e)}")
        return None


def find_best_preview_asset(
    item: pystac.Item, max_size_kb: Optional[int] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Find the most appropriate preview asset based on size constraints"""
    collection_info = SUPPORTED_COLLECTIONS.get(item.collection_id)
    if not collection_info:
        return None, None

    for asset_key, target_size in collection_info["preview_assets"]:
        if asset_key in item.assets:
            if max_size_kb is None or (target_size and target_size <= max_size_kb):
                return asset_key, target_size
    return None, None


def create_preview_image(
    item: pystac.Item,
    config: Optional[ImageProcessingConfig] = None,
    force_reload: bool = False,
) -> Optional[Image.Image]:
    """Create an optimized preview image with caching"""
    logger.info(f"Creating preview for item {item.id}")
    config = config or ImageProcessingConfig()

    try:  # Changed { to :
        # Determine appropriate preview size based on config
        max_size_kb = None
        if config.quality_preset != ImageQuality.FULL:
            max_size_kb = {
                ImageQuality.LOW: 1024,  # 1MB
                ImageQuality.MEDIUM: 5120,  # 5MB
                ImageQuality.HIGH: 20480,  # 20MB
            }[config.quality_preset]

        # Find best preview asset
        asset_key, target_size = find_best_preview_asset(item, max_size_kb)
        if not asset_key:
            logger.error("No suitable preview asset found")
            return None

        # Check cache first
        if not force_reload:
            cached_path = get_cached_preview(item.id, asset_key)
            if cached_path:
                logger.info(f"Using cached preview from {cached_path}")
                return Image.open(cached_path)

        # Download preview
        preview_url = item.assets[asset_key].href
        logger.info(f"Downloading {asset_key} preview from {preview_url}")

        # Stream download with progress tracking
        response = requests.get(preview_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        # Get total size for progress calculation
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        image_data = BytesIO()

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            image_data.write(chunk)
            downloaded += len(chunk)
            progress = (downloaded / total_size * 100) if total_size else 0
            logger.debug(f"Download progress: {progress:.1f}%")

        image_data.seek(0)

        # Process image
        preview = safely_open_large_image(image_data, config)
        if not preview:
            return None

        # Cache the preview
        cache_key = hashlib.md5(f"{item.id}_{asset_key}".encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.jpg")
        preview.save(cache_path, "JPEG", quality=85, optimize=True)
        logger.info(f"Cached preview to {cache_path}")

        return preview

    except Exception as e:
        logger.error(f"Error creating preview: {e}")
        logger.exception("Full traceback:")
        return None


def draw_detections(
    image: np.ndarray,
    predictions: List[Dict],
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),  # Green boxes by default
) -> np.ndarray:
    """Draw bounding boxes and labels on satellite imagery

    Args:
        image: Input image as numpy array (BGR format)
        predictions: List of detection dictionaries with 'box' and 'confidence' keys
        thickness: Line thickness for boxes
        color: BGR color tuple for boxes and labels

    Returns:
        Image with drawn detections
    """
    img_height, img_width = image.shape[:2]
    result = image.copy()

    for pred in predictions:
        try:
            # Extract box coordinates and ensure they're integers
            box = pred["box"]
            x1, y1, x2, y2 = map(int, box)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            x2 = max(0, min(x2, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw confidence label
            conf = pred.get("confidence", 0)
            label = f"Fire: {conf:.1%}"

            # Calculate label position
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = y1 - 5 if y1 - 5 > label_size[1] else y1 + label_size[1] + 5

            # Draw label background
            cv2.rectangle(
                result,
                (x1, label_y - label_size[1] - baseline),
                (x1 + label_size[0], label_y + baseline),
                color,
                cv2.FILLED,
            )

            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA,
            )

        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            continue

    return result


def process_satellite_image(
    item: pystac.Item,
    progress_callback: Optional[Callable[[float], None]] = None,
    config: Optional[ImageProcessingConfig] = None,
    predictions: Optional[List[Dict]] = None,  # Add predictions parameter
) -> Optional[str]:
    """Process satellite image with optional detection visualization"""
    try:
        logger.info(f"Processing satellite image {item.id}")
        config = config or ImageProcessingConfig()

        if progress_callback:
            progress_callback(5)

        # Get preview with size control
        preview = create_preview_image(item, config)
        if not preview:
            return None

        if progress_callback:
            progress_callback(50)

        # Convert PIL Image to numpy array for OpenCV
        image_array = np.array(preview)

        # Draw detections if provided
        if predictions:
            logger.info(f"Drawing {len(predictions)} detections")
            image_array = draw_detections(image_array, predictions)

        if progress_callback:
            progress_callback(75)

        # Save result
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            result_image = Image.fromarray(image_array)  # Changed PILImage to Image
            result_image.save(
                tmp.name, "JPEG", quality=config.jpeg_quality, optimize=True
            )
            logger.info(f"Saved result to: {tmp.name}")

            if progress_callback:
                progress_callback(100)

            return tmp.name

    except Exception as e:
        logger.error(f"Error in process_satellite_image: {e}")
        logger.exception("Full traceback:")
        return None


async def process_detection(
    item: pystac.Item, config: ImageProcessingConfig, task_id: str
) -> None:
    """Process detection as a background task"""
    try:
        TaskManager.update_status(task_id, "downloading")

        # Get preview image
        preview = create_preview_image(item, config)
        if not preview:
            TaskManager.update_status(
                task_id, "failed", {"error": "Failed to create preview"}
            )
            return

        TaskManager.update_status(task_id, "detecting")

        # Process detections
        result = process_satellite_image(item, config=config)
        if not result or not isinstance(result, dict):
            TaskManager.update_status(task_id, "failed", {"error": "Detection failed"})
            return

        # Store result with predictions from the result object
        TaskManager.update_status(
            task_id,
            "completed",
            {
                "image_path": result.get("image_path"),
                "detection_count": len(result.get("predictions", [])),
                "predictions": result.get("predictions", []),
            },
        )

    except Exception as e:
        TaskManager.update_status(task_id, "failed", {"error": str(e)})


def get_available_collections() -> Dict[str, str]:
    """Get available satellite collections with descriptions"""
    return {
        id: f"{info['name']} ({info['resolution']})"
        for id, info in SUPPORTED_COLLECTIONS.items()
    }


def setup_mpc() -> Tuple[Optional[pystac_client.Client], Dict[str, str]]:
    """Setup Microsoft Planetary Computer and return both client and collections

    Returns:
        Tuple containing:
        - STAC client (if successful) or None
        - Dictionary of available collections with descriptions
    """
    logger.info("Setting up Microsoft Planetary Computer connection...")

    # Initialize the client
    catalog = init_planetary_computer()
    if not catalog:
        logger.error("Failed to initialize Planetary Computer client")
        return None, {}

    # Get available collections
    collections = get_available_collections()
    logger.info(f"Found {len(collections)} supported collections")

    return catalog, collections
