"""Image processing utilities for tiling and batch processing satellite imagery."""

import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import gc

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TiledImage:
    """Represents a large image split into tiles"""

    original_path: str
    original_name: str
    tiles: List[str]  # Paths to tile images
    tile_coords: List[Tuple[int, int, int, int]]  # (x, y, width, height)
    original_width: int
    original_height: int
    metadata: Optional[Dict] = None


class TileConfig:
    """Configuration for image tiling"""

    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 128,
        batch_size: int = 4,
        max_workers: int = 4,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.max_workers = max_workers


def should_tile_image(file_path: str, max_size_mb: float = 150) -> bool:
    """
    Determine if an image should be tiled based on size or dimensions

    Args:
        file_path: Path to image file
        max_size_mb: Maximum file size before tiling

    Returns:
        Boolean indicating if tiling is needed
    """
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return True

    # Check dimensions
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False
        height, width = img.shape[:2]
        # Tile if dimensions exceed 8000 pixels
        if width > 8000 or height > 8000:
            return True
        del img
        gc.collect()
    except Exception as e:
        logger.error(f"Error checking image dimensions: {e}")
        # Default to file size check if can't read image
        return file_size_mb > max_size_mb

    return False


def create_tiled_image(
    file_path: str,
    file_name: str,
    config: Optional[TileConfig] = None,
    preserve_metadata: bool = True,
) -> Optional[TiledImage]:
    """
    Split a large image into tiles with configurable overlap

    Args:
        file_path: Path to the image file
        file_name: Original filename
        config: TileConfig object with tiling parameters
        preserve_metadata: Whether to preserve geospatial metadata

    Returns:
        TiledImage object containing tile information
    """
    if config is None:
        config = TileConfig()

    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Could not read image: {file_path}")
            return None

        height, width = img.shape[:2]
        tile_size = config.tile_size
        overlap = config.overlap

        # Calculate number of tiles needed
        n_tiles_h = max(1, (height + tile_size - overlap - 1) // (tile_size - overlap))
        n_tiles_w = max(1, (width + tile_size - overlap - 1) // (tile_size - overlap))

        # Create temporary directory for tiles
        temp_dir = tempfile.mkdtemp(prefix="satellite_tiles_")
        tiles = []
        tile_coords = []

        # Process tiles in batches
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    # Calculate tile coordinates with overlap
                    x = min(j * (tile_size - overlap), width - tile_size)
                    y = min(i * (tile_size - overlap), height - tile_size)
                    x = max(0, x)  # Ensure x is not negative
                    y = max(0, y)  # Ensure y is not negative

                    # Handle edge cases for last tiles
                    w = min(tile_size, width - x)
                    h = min(tile_size, height - y)

                    # Extract and save tile
                    tile = img[y : y + h, x : x + w].copy()

                    # Pad if necessary to maintain square aspect ratio
                    if w < tile_size or h < tile_size:
                        padded_tile = np.zeros(
                            (tile_size, tile_size, 3), dtype=np.uint8
                        )
                        padded_tile[0:h, 0:w] = tile
                        tile = padded_tile

                    # Save tile
                    tile_path = os.path.join(temp_dir, f"tile_{i}_{j}.jpg")

                    # Submit tile saving to thread pool
                    futures.append(executor.submit(cv2.imwrite, tile_path, tile))

                    tiles.append(tile_path)
                    tile_coords.append((x, y, w, h))

                    # Clear tile from memory
                    del tile
                    if len(futures) >= config.batch_size:
                        # Wait for batch to complete
                        for future in futures:
                            future.result()
                        futures = []
                        gc.collect()

            # Wait for remaining futures
            for future in futures:
                future.result()

        # Clear original image from memory
        del img
        gc.collect()

        # Extract metadata if needed
        metadata = None
        if preserve_metadata:
            try:
                with Image.open(file_path) as img:
                    metadata = img.info
            except Exception as e:
                logger.warning(f"Could not preserve metadata: {e}")

        return TiledImage(
            original_path=file_path,
            original_name=file_name,
            tiles=tiles,
            tile_coords=tile_coords,
            original_width=width,
            original_height=height,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Error creating tiled image: {e}")
        return None


def merge_tile_results(tiled_image: TiledImage, tile_results: List[Dict]) -> Dict:
    """
    Merge detection results from tiles back into original image

    Args:
        tiled_image: TiledImage object containing tile information
        tile_results: List of detection results for each tile

    Returns:
        Dictionary with merged detection results
    """
    try:
        # Create blank image for visualization
        merged_image = np.zeros(
            (tiled_image.original_height, tiled_image.original_width, 3), dtype=np.uint8
        )

        # Lists to store all predictions
        all_predictions = []

        # Merge tiles
        for i, (tile_result, (x, y, w, h)) in enumerate(
            zip(tile_results, tiled_image.tile_coords)
        ):
            if "error" in tile_result:
                continue

            # Merge visualizations
            if "image" in tile_result:
                tile_img = tile_result["image"]
                # Ensure tile image has correct dimensions
                if tile_img.shape[:2] != (h, w):
                    tile_img = cv2.resize(tile_img, (w, h))
                merged_image[y : y + h, x : x + w] = tile_img

            # Transform predictions to original coordinates
            for pred in tile_result.get("predictions", []):
                # Get bounding box
                box = pred.get("box", [0, 0, 0, 0])

                # Transform coordinates to original image
                transformed_box = [
                    box[0] + x,  # x1
                    box[1] + y,  # y1
                    box[2] + x,  # x2
                    box[3] + y,  # y2
                ]

                # Create new prediction with transformed coordinates
                new_pred = pred.copy()
                new_pred["box"] = transformed_box
                new_pred["tile_idx"] = i

                all_predictions.append(new_pred)

        # Remove duplicate detections at tile boundaries
        final_predictions = remove_duplicate_detections(all_predictions)

        # Create merged result
        merged_result = {
            "image": merged_image,
            "predictions": final_predictions,
            "count": len(final_predictions),
            "filename": tiled_image.original_name,
            "metadata": tiled_image.metadata,
        }

        return merged_result

    except Exception as e:
        logger.error(f"Error merging tile results: {e}")
        return {"error": str(e)}


def remove_duplicate_detections(
    predictions: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Remove duplicate detections at tile boundaries using IoU

    Args:
        predictions: List of detection predictions
        iou_threshold: IoU threshold for considering detections as duplicates

    Returns:
        List of filtered predictions
    """
    if not predictions:
        return []

    # Sort predictions by confidence
    sorted_preds = sorted(
        predictions, key=lambda x: x.get("confidence", 0), reverse=True
    )

    # List to store kept predictions
    kept_preds = []

    for pred in sorted_preds:
        # Keep first prediction
        if not kept_preds:
            kept_preds.append(pred)
            continue

        # Check IoU with all kept predictions
        duplicate = False
        for kept_pred in kept_preds:
            iou = calculate_iou(pred["box"], kept_pred["box"])
            if iou > iou_threshold:
                duplicate = True
                break

        # If not a duplicate, keep it
        if not duplicate:
            kept_preds.append(pred)

    return kept_preds


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes

    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    return iou
