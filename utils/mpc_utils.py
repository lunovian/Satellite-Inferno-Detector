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
from PIL import Image
from planetary_computer import sign_inplace
from pystac.extensions.eo import EOExtension

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Simplified collection configuration
SUPPORTED_COLLECTIONS = {
    "sentinel-2-l2a": {
        "name": "Sentinel-2 Level 2A",
        "resolution": "10m",
        "preview_asset": "visual",
    },
    "landsat-c2-l2": {
        "name": "Landsat 8/9 Collection 2 Level-2",
        "resolution": "30m",
        "preview_asset": "rendered_preview",
    },
}

# Update timeout constants
DOWNLOAD_TIMEOUT = 120  # seconds
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming


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


def create_preview_image(item: pystac.Item) -> Optional[Image.Image]:
    """Create a preview image using direct download instead of rasterio"""
    logger.info(f"Starting preview creation for item {item.id}")
    try:
        # Get the appropriate preview asset
        preview_asset = SUPPORTED_COLLECTIONS[item.collection_id]["preview_asset"]
        if preview_asset not in item.assets:
            logger.error(f"Preview asset '{preview_asset}' not found in item assets")
            return None

        # Get the signed URL
        preview_url = item.assets[preview_asset].href
        logger.info(f"Downloading preview from: {preview_url}")

        # Download with requests and progress tracking
        response = requests.get(preview_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        # Read image data in chunks
        image_data = BytesIO()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            image_data.write(chunk)

        # Create PIL Image from downloaded data
        logger.info("Creating image from downloaded data")
        return Image.open(image_data)

    except requests.Timeout:
        logger.error(f"Download timed out after {DOWNLOAD_TIMEOUT}s")
        return None
    except requests.RequestException as e:
        logger.error(f"Download failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error creating preview: {str(e)}")
        logger.exception("Full traceback:")
        return None


def process_satellite_image(
    item: pystac.Item, progress_callback: Optional[Callable[[float], None]] = None
) -> Optional[str]:
    """Process satellite image with reliable progress tracking"""
    try:
        logger.info(f"Starting satellite image processing for item {item.id}")

        # Initial progress
        if progress_callback:
            progress_callback(5)
            logger.info("Download preparation started")

        # Get preview image with progress updates
        preview = create_preview_image(item)
        if not preview:
            logger.error("Failed to create preview image")
            return None

        if progress_callback:
            progress_callback(75)
            logger.info("Image downloaded successfully")

        # Save to file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                preview.save(tmp.name, "JPEG", quality=85, optimize=True)
                logger.info(f"Saved preview to: {tmp.name}")

                if progress_callback:
                    progress_callback(100)
                    logger.info("Processing complete")

                return tmp.name

        except Exception as save_error:
            logger.error(f"Error saving image: {save_error}")
            return None

    except Exception as e:
        logger.error(f"Error in satellite image processing: {str(e)}")
        logger.exception("Full traceback:")
        return None


def get_available_collections() -> Dict[str, str]:
    """Get available satellite collections with descriptions"""
    return {
        id: f"{info['name']} ({info['resolution']})"
        for id, info in SUPPORTED_COLLECTIONS.items()
    }
