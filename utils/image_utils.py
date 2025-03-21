import os
import cv2
import numpy as np
import tempfile
import math
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
import uuid
import time

logger = logging.getLogger(__name__)


class ImageTile:
    """Represents a single tile from a larger image"""

    def __init__(
        self, path: str, coords: Tuple[int, int, int, int], parent_id: str, index: int
    ):
        self.path = path
        self.coords = coords  # (x_start, y_start, width, height)
        self.parent_id = parent_id
        self.index = index
        self.results = None

    @property
    def x_start(self) -> int:
        return self.coords[0]

    @property
    def y_start(self) -> int:
        return self.coords[1]

    @property
    def width(self) -> int:
        return self.coords[2]

    @property
    def height(self) -> int:
        return self.coords[3]


class TiledImage:
    """Manages a large image split into tiles"""

    def __init__(
        self,
        original_path: str,
        original_name: str,
        tile_size: int = 1024,
        overlap: int = 128,
    ):
        self.original_path = original_path
        self.original_name = original_name
        self.tile_size = tile_size
        self.overlap = overlap
        self.width = 0
        self.height = 0
        self.tiles: List[ImageTile] = []
        self.id = str(uuid.uuid4())
        self.processed = False
        self.combined_result_path = None

    def load_and_split(self) -> bool:
        """Load image and split into tiles"""
        try:
            # Read image dimensions first
            with Image.open(self.original_path) as img:
                self.width, self.height = img.size
                img_format = img.format
                logger.info(
                    f"Image size: {self.width}x{self.height}, format: {img_format}"
                )

            # Load image
            img = cv2.imread(self.original_path)
            if img is None:
                logger.error(f"Failed to load image: {self.original_path}")
                return False

            # Create tiles
            tile_index = 0
            for y in range(0, self.height, self.tile_size - self.overlap):
                for x in range(0, self.width, self.tile_size - self.overlap):
                    # Ensure we don't go beyond image boundaries
                    actual_width = min(self.tile_size, self.width - x)
                    actual_height = min(self.tile_size, self.height - y)

                    # Skip tiny tiles at edges
                    if (
                        actual_width < self.tile_size / 4
                        or actual_height < self.tile_size / 4
                    ):
                        continue

                    # Extract tile
                    tile_img = img[y : y + actual_height, x : x + actual_width]

                    # Create a temporary file for the tile
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp:
                        tile_path = tmp.name

                    # Save tile with high quality
                    cv2.imwrite(tile_path, tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    # Create tile object
                    tile = ImageTile(
                        path=tile_path,
                        coords=(x, y, actual_width, actual_height),
                        parent_id=self.id,
                        index=tile_index,
                    )

                    self.tiles.append(tile)
                    tile_index += 1

            logger.info(f"Image split into {len(self.tiles)} tiles")
            return True

        except Exception as e:
            logger.error(f"Error splitting image: {str(e)}")
            return False

    def combine_results(self, output_dir: Optional[str] = None) -> Optional[str]:
        """Combine detection results from all tiles into a single image"""
        if not self.tiles or not all(tile.results for tile in self.tiles):
            logger.error("Not all tiles have been processed")
            return None

        try:
            # Create blank canvas with original dimensions
            combined_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Create a mask to track where we've placed detections
            detection_mask = np.zeros((self.height, self.width), dtype=np.uint8)

            # Build a results container for all detections
            all_detections = []

            # Process each tile
            for tile in self.tiles:
                # Get tile image with detections
                tile_img = cv2.imread(tile.results["image_path"])
                if tile_img is None:
                    logger.warning(f"Could not read result for tile {tile.index}")
                    continue

                # Get tile coordinates
                x, y, w, h = tile.coords

                # Place the detection image onto the canvas
                combined_img[y : y + h, x : x + w] = tile_img

                # Mark the area as processed in the mask
                detection_mask[y : y + h, x : x + w] = 1

                # Adjust bounding boxes to global coordinates
                for detection in tile.results["predictions"]:
                    # Get local coordinates
                    local_box = detection["box"]

                    # Convert to global coordinates
                    global_box = [
                        local_box[0] + x,  # x1
                        local_box[1] + y,  # y1
                        local_box[2] + x,  # x2
                        local_box[3] + y,  # y2
                    ]

                    # Create a copy of the detection with global coordinates
                    global_detection = detection.copy()
                    global_detection["box"] = global_box
                    global_detection["tile_index"] = tile.index

                    all_detections.append(global_detection)

            # Compute full image with detections
            combined_result = self._apply_detections_to_image(all_detections)

            # Create output directory if needed
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, f"combined_{self.id}.jpg")
            else:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    result_path = tmp.name

            # Save the combined result
            cv2.imwrite(result_path, combined_result)
            self.combined_result_path = result_path
            self.processed = True

            return result_path

        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return None

    def _apply_detections_to_image(self, detections: List[Dict]) -> np.ndarray:
        """Apply detections to the original image"""
        # Load original image
        original = cv2.imread(self.original_path)
        if original is None:
            logger.error(f"Could not read original image: {self.original_path}")
            # Create blank canvas
            original = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # NMS to remove duplicate detections in overlapping areas
        merged_detections = self._perform_nms(detections, iou_threshold=0.5)

        # Draw detections on the image
        for detection in merged_detections:
            box = detection["box"]
            conf = detection["confidence"]
            box = [int(coord) for coord in box]

            # Draw bounding box
            cv2.rectangle(original, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Draw label
            label = f"Fire: {conf:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            y_label = max(box[1], label_size[1])
            cv2.rectangle(
                original,
                (box[0], y_label - label_size[1] - 10),
                (box[0] + label_size[0], y_label),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                original,
                label,
                (box[0], y_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return original

    def _perform_nms(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Perform non-maximum suppression on detections"""
        if not detections:
            return []

        # Extract boxes and scores
        boxes = np.array([d["box"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        # Perform NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.1,  # Low threshold since we already filtered
            nms_threshold=iou_threshold,
        )

        # OpenCV 4.5.4+ returns a flat array
        if isinstance(indices, tuple):
            indices = indices[0]

        # Return detections after NMS
        return [detections[i] for i in indices]

    def cleanup(self):
        """Remove temporary files"""
        for tile in self.tiles:
            try:
                if os.path.exists(tile.path):
                    os.unlink(tile.path)
            except Exception as e:
                logger.warning(f"Error cleaning up tile {tile.index}: {str(e)}")

        # Don't remove the combined result


def check_file_size(file_path: str) -> int:
    """Check file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error checking file size: {str(e)}")
        return 0


def should_tile_image(file_path: str, max_size_mb: int = 150) -> bool:
    """Determine if image should be tiled based on file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = check_file_size(file_path)
    return file_size > max_size_bytes


def get_optimal_tile_size(file_path: str, target_size_mb: int = 50) -> int:
    """Calculate optimal tile size to achieve target file size"""
    try:
        # Get original image dimensions
        img = Image.open(file_path)
        width, height = img.size
        img.close()

        # Get file size
        file_size = check_file_size(file_path)

        # Calculate pixels per MB
        pixels = width * height
        pixels_per_mb = pixels / (file_size / (1024 * 1024))

        # Calculate target pixels for desired tile size
        target_pixels = pixels_per_mb * target_size_mb

        # Calculate tile dimension (assuming square tiles)
        tile_dim = int(math.sqrt(target_pixels))

        # Round to nearest 256
        tile_dim = (tile_dim // 256) * 256

        # Ensure minimum and maximum sizes
        tile_dim = max(512, min(tile_dim, 2048))

        return tile_dim

    except Exception as e:
        logger.error(f"Error calculating tile size: {str(e)}")
        return 1024  # Default tile size


def create_tiled_image(
    file_path: str,
    file_name: str,
    tile_size: Optional[int] = None,
    overlap: int = 128,
    max_size_mb: int = 150,
) -> Optional[TiledImage]:
    """Create a tiled image if the file exceeds size threshold"""
    try:
        # Check if we need to tile this image
        if not should_tile_image(file_path, max_size_mb):
            return None

        # Calculate optimal tile size if not specified
        if tile_size is None:
            tile_size = get_optimal_tile_size(file_path)

        # Create tiled image
        tiled_image = TiledImage(
            original_path=file_path,
            original_name=file_name,
            tile_size=tile_size,
            overlap=overlap,
        )

        # Load and split
        if tiled_image.load_and_split():
            return tiled_image

    except Exception as e:
        logger.error(f"Error creating tiled image: {str(e)}")

    return None
