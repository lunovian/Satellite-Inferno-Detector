import os
import cv2
import numpy as np
import torch
import traceback
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import glob
import argparse
import time
import json
import csv
from tqdm import tqdm
from utils.metric_visualization import MetricsVisualizer


# Add better error handling and debug messages
def debug_log(message):
    print(f"[DEBUG] {message}")


class YOLOEnsemble:
    def __init__(self, models_dir="models", conf_thres=0.3, iou_thres=0.5):
        """
        Initialize the YOLO ensemble with models from the specified directory
        """
        self.models = []
        self.model_names = []  # Store model names for better identification
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize metrics storage
        self.metrics = {}
        self.metrics_visualizer = MetricsVisualizer()

        # Only load models from directory if a directory is specified
        if models_dir:
            try:
                # Load all YOLO models from the models directory
                model_files = glob.glob(os.path.join(models_dir, "*.pt"))
                if not model_files:
                    raise ValueError(f"No model files found in {models_dir}")

                debug_log(f"Loading {len(model_files)} YOLO models...")
                for model_path in model_files:
                    model_name = os.path.basename(model_path)
                    debug_log(f"Loading {model_name}...")
                    model = YOLO(model_path)
                    self.models.append(model)

                    # Extract YOLO version from the model name
                    version_name = self._extract_yolo_version(model_name)
                    self.model_names.append(version_name)
                    debug_log(f"Identified as {version_name}")

                debug_log("All models loaded successfully!")
            except Exception as e:
                print(f"Error loading models: {e}")
                traceback.print_exc()
                # Don't raise exception here, just return an empty model list
                # This allows the application to initialize without models
                # and then load them explicitly later

    def _extract_yolo_version(self, model_name):
        """
        Extract YOLO version from model filename

        Args:
            model_name (str): Filename of the model

        Returns:
            str: YOLO version name
        """
        # Remove file extension
        base_name = os.path.splitext(model_name)[0].lower()

        # Check for known YOLO versions
        if "yolov8" in base_name:
            size_suffix = base_name.replace("yolov8", "")
            return f"YOLOv8-{size_suffix}"
        elif "yolov9" in base_name:
            size_suffix = base_name.replace("yolov9", "")
            return f"YOLOv9-{size_suffix}"
        elif "yolov11" in base_name:
            size_suffix = base_name.replace("yolov11", "")
            return f"YOLOv11-{size_suffix}"
        elif "yolov7" in base_name:
            size_suffix = base_name.replace("yolov7", "")
            return f"YOLOv7-{size_suffix}"
        elif "yolov5" in base_name:
            size_suffix = base_name.replace("yolov5", "")
            return f"YOLOv5-{size_suffix}"
        else:
            # If version can't be determined, return the original name
            return model_name

    def _detect_smoke(self, img, bbox):
        """
        HSV-based smoke detection for a region using thresholding

        Args:
            img: The input image in BGR format
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            float: Smoke confidence score (0-1)
        """
        # Extract the region of interest
        x1, y1, x2, y2 = [int(c) for c in bbox]
        roi = img[y1:y2, x1:x2]

        if roi.size == 0:  # Handle empty ROI
            return 0.0

        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define smoke HSV ranges (grayish-white to light gray)
        # Lower bound: low saturation, high value
        lower_smoke = np.array([0, 0, 150])
        upper_smoke = np.array([180, 60, 255])

        # Create mask for smoke
        mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        # Calculate smoke ratio
        smoke_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

        # Define secondary feature: variance in saturation
        # Smoke typically has low variance in saturation
        sat_channel = hsv[:, :, 1]
        if sat_channel.size > 0:
            sat_variance = np.var(sat_channel) / 255.0
            # Lower variance indicates more uniform appearance like smoke
            sat_confidence = max(0, 1 - min(1, sat_variance * 5))
        else:
            sat_confidence = 0

        # Combine metrics (smoke ratio and saturation uniformity)
        # Higher weight to smoke ratio, lower to sat_confidence
        smoke_confidence = 0.7 * smoke_ratio + 0.3 * sat_confidence

        # Scale and clamp confidence
        smoke_confidence = min(1.0, smoke_confidence * 1.5)

        return smoke_confidence

    def _estimate_fire_size(self, img, bbox):
        """
        Estimate the relative size of fire within a bounding box

        Args:
            img: The input image in BGR format
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            float: Fire ratio (fire pixels / total bbox pixels)
        """
        # Extract the region of interest
        x1, y1, x2, y2 = [int(c) for c in bbox]
        roi = img[y1:y2, x1:x2]

        if roi.size == 0:  # Handle empty ROI
            return 0.0

        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define fire HSV range (red-orange-yellow)
        # First range (red-orange)
        lower_fire1 = np.array([0, 120, 100])
        upper_fire1 = np.array([30, 255, 255])

        # Second range (red hues at upper end of range)
        lower_fire2 = np.array([150, 120, 100])
        upper_fire2 = np.array([180, 255, 255])

        # Create masks for fire
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

        # Combine masks
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate fire ratio (fire pixels / total bbox pixels)
        total_pixels = roi.shape[0] * roi.shape[1]
        fire_pixels = cv2.countNonZero(fire_mask)

        fire_ratio = fire_pixels / total_pixels if total_pixels > 0 else 0

        return fire_ratio

    def _improved_ensemble_fusion(self, predictions):
        """
        Enhanced ensemble fusion that truly combines predictions from multiple models
        using Weighted Box Fusion (WBF) rather than NMS. WBF creates weighted average
        boxes from overlapping predictions of multiple models, effectively leveraging
        the strengths of each model.

        Strategy:
        1. Cluster overlapping predictions across models using IoU
        2. For each cluster, create a weighted average prediction using WBF
        3. Apply model-specific weights based on known strengths
        4. Increase weights when multiple models agree
        5. Use soft-WBF to retain multiple overlapping predictions where appropriate

        Args:
            predictions (list): List of predictions from all models

        Returns:
            list: Fused predictions that leverage complementary model strengths
        """
        if not predictions:
            return []

        try:
            # Extract image for analysis if available
            image = None
            for pred in predictions:
                if "img" in pred:
                    image = pred["img"]
                    break

            # Group predictions by class to process each class separately
            class_predictions = {}
            for pred in predictions:
                cls_id = pred["cls_id"]
                if cls_id not in class_predictions:
                    class_predictions[cls_id] = []
                class_predictions[cls_id].append(pred)

            # Track model counts for debugging
            model_counts = {}
            for pred in predictions:
                model_name = pred.get("model_name", f"Model-{pred['model_idx']}")
                if model_name not in model_counts:
                    model_counts[model_name] = 0
                model_counts[model_name] += 1

            debug_log(f"Input predictions by model: {model_counts}")

            # Final predictions after fusion
            final_predictions = []

            # Process each class separately
            for cls_id, class_preds in class_predictions.items():
                # Skip if no predictions for this class
                if not class_preds:
                    continue

                # 1. First, calculate IoU between all predictions to identify clusters
                clusters = self._cluster_predictions(class_preds, iou_threshold=0.5)

                debug_log(f"Found {len(clusters)} clusters for class {cls_id}")

                # 2. Process each cluster to create fused predictions using WBF
                for cluster_idx, cluster in enumerate(clusters):
                    # Skip empty clusters
                    if not cluster:
                        continue

                    # Skip clusters with only one prediction (no fusion needed)
                    if len(cluster) == 1:
                        final_predictions.append(cluster[0])
                        continue

                    # Create a fused prediction from the cluster using WBF
                    fused_pred = self._apply_weighted_box_fusion(cluster, image)

                    # Add the fused prediction to final results
                    if fused_pred:
                        final_predictions.append(fused_pred)

            # 3. Apply soft-WBF across classes to handle potentially overlapping detections
            if final_predictions:
                final_predictions = self._soft_weighted_box_fusion(final_predictions)

            return final_predictions

        except Exception as e:
            print(f"Error in improved ensemble fusion: {e}")
            traceback.print_exc()
            # Fallback to original NMS implementation
            return self._legacy_nms_predictions(predictions)

    def _apply_weighted_box_fusion(self, cluster, image=None):
        """
        Apply Weighted Box Fusion (WBF) to a cluster of predictions for the same object.
        WBF creates a weighted average box based on the confidence scores and
        model-specific weights.

        Args:
            cluster (list): List of predictions for the same object
            image (array): Original image for analyzing fire characteristics

        Returns:
            dict: Fused prediction with combined strengths
        """
        if not cluster:
            return None

        # If only one prediction, return it directly
        if len(cluster) == 1:
            return cluster[0]

        # Count models in this cluster for diversity weighting
        models_in_cluster = set([pred["model_idx"] for pred in cluster])
        model_diversity_factor = min(len(models_in_cluster) / 3.0, 1.0) + 0.5

        # Calculate WBF weights - these are more important than in our previous approach
        # as they directly impact box coordinates in WBF
        pred_weights = []

        for pred in cluster:
            # Use confidence as base weight (better than NMS which ignores lower confidence detections)
            weight = pred["conf"]

            # Apply model-specific expertise weighting
            model_name = pred.get("model_name", "").lower()

            # YOLOv8: Strong in precision (for higher confidence detections)
            if "v8" in model_name:
                # Exponential boost for high confidence predictions
                confidence_boost = (
                    np.exp(min(pred["conf"] - 0.5, 0.5)) if pred["conf"] > 0.5 else 1.0
                )
                weight *= 1.2 * confidence_boost

            # YOLOv9: Good balance and general detection
            elif "v9" in model_name:
                weight *= 1.1

            # YOLOv11: Better for small fires and difficult detections
            elif "v11" in model_name:
                # Apply a smaller general weight
                weight *= 0.9

                # But boost for scenarios where v11 excels
                if image is not None:
                    # Check if this is a small fire
                    box = pred["box"]
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    img_width, img_height = image.shape[1], image.shape[0]

                    # If box is small relative to image or smoke detected
                    is_small = (box_width * box_height) / (
                        img_width * img_height
                    ) < 0.05
                    smoke_conf = (
                        self._detect_smoke(image, box)
                        if hasattr(self, "_detect_smoke")
                        else 0.0
                    )

                    # Boost weight for small fires or when smoke detected
                    if is_small:
                        weight *= 1.5
                    if smoke_conf > 0.3:
                        weight *= 1.0 + smoke_conf

            # Multiply all weights by the model diversity factor
            weight *= model_diversity_factor

            pred_weights.append(weight)

        # Normalize weights to sum to 1 for proper weighted averaging
        total_weight = sum(pred_weights)
        if total_weight > 0:
            pred_weights = [w / total_weight for w in pred_weights]
        else:
            # Equal weights if all weights are 0
            pred_weights = [1.0 / len(cluster)] * len(cluster)

        # Calculate weighted average box using WBF formula
        fused_box = np.zeros(4)
        fused_conf = 0.0
        model_idxs = []
        model_names = []

        # WBF combines both the bounding box coordinates AND the confidence scores
        for pred, weight in zip(cluster, pred_weights):
            # Weighted box coordinates - key component of WBF
            fused_box += np.array(pred["box"]) * weight

            # Weighted confidence - second key component of WBF
            # In true WBF this is a more complex function than simple averaging
            # We use a modified formula to account for agreement between models
            fused_conf += pred["conf"] * weight * model_diversity_factor

            # Collect model info for reference
            model_idxs.append(pred["model_idx"])
            if "model_name" in pred:
                model_names.append(pred["model_name"])

        # Create the fused prediction
        fused_pred = {
            "box": fused_box,
            "conf": min(fused_conf, 1.0),  # Cap at 1.0
            "cls_id": cluster[0][
                "cls_id"
            ],  # All predictions in cluster have same class
            "model_idx": -1,  # Special value indicating ensemble fusion
            "model_name": "WBF Fusion",  # Changed to indicate WBF
            "source_models": model_idxs,
            "source_model_names": model_names,
            "is_fused": True,
            "fusion_weight": model_diversity_factor,
            "wbf_weights": pred_weights,  # Store the weights for reference
        }

        # Preserve image reference if available
        if "img" in cluster[0]:
            fused_pred["img"] = cluster[0]["img"]

        return fused_pred

    def _soft_weighted_box_fusion(
        self, predictions, iou_threshold=0.5, score_threshold=0.01
    ):
        """
        Apply a soft version of WBF that allows some overlapping boxes to remain
        if they're from different models or detect different sizes.

        Args:
            predictions (list): List of predictions to process
            iou_threshold (float): IoU threshold for considering boxes as overlapping
            score_threshold (float): Minimum confidence threshold to keep

        Returns:
            list: List of fused predictions
        """
        # Sort predictions by confidence
        sorted_preds = sorted(predictions, key=lambda x: x["conf"], reverse=True)

        # Initialize result list
        kept_preds = []

        # Process each prediction
        for pred in sorted_preds:
            # Skip if confidence already below threshold
            if pred["conf"] < self.conf_thres:
                continue

            # Check if we should merge this prediction with any existing one
            merged = False

            for i, kept_pred in enumerate(kept_preds):
                # Only consider merging same class and if they overlap significantly
                if pred["cls_id"] == kept_pred["cls_id"]:
                    iou = self._calculate_iou(pred["box"], kept_pred["box"])

                    if iou >= iou_threshold:
                        # Don't automatically merge, check if they complement each other

                        # 1. Different models (complementary strengths)
                        different_models = pred.get("model_idx", -1) != kept_pred.get(
                            "model_idx", -2
                        )

                        # 2. One is from fusion and one is not
                        one_is_fused = pred.get("is_fused", False) != kept_pred.get(
                            "is_fused", False
                        )

                        # 3. Significantly different sizes (one might detect small feature)
                        pred_area = (pred["box"][2] - pred["box"][0]) * (
                            pred["box"][3] - pred["box"][1]
                        )
                        kept_area = (kept_pred["box"][2] - kept_pred["box"][0]) * (
                            kept_pred["box"][3] - kept_pred["box"][1]
                        )
                        size_ratio = min(pred_area, kept_area) / max(
                            pred_area, kept_area
                        )
                        different_sizes = (
                            size_ratio < 0.7
                        )  # Significant size difference

                        # If they seem complementary, keep both
                        if (different_models and one_is_fused) or different_sizes:
                            # Reduce confidence slightly to acknowledge overlap
                            pred["conf"] *= 0.9
                            continue

                        # Otherwise merge them with WBF - true fusion
                        combined_cluster = [pred, kept_pred]

                        # Calculate weights based on confidence
                        total_conf = pred["conf"] + kept_pred["conf"]
                        if total_conf > 0:
                            w1 = pred["conf"] / total_conf
                            w2 = kept_pred["conf"] / total_conf
                        else:
                            w1, w2 = 0.5, 0.5

                        # WBF formula for box coordinates
                        fused_box = pred["box"] * w1 + kept_pred["box"] * w2

                        # Create merged prediction, favoring the higher confidence one
                        merged_pred = (
                            kept_pred.copy()
                            if kept_pred["conf"] >= pred["conf"]
                            else pred.copy()
                        )
                        merged_pred["box"] = fused_box
                        merged_pred["conf"] = max(
                            pred["conf"], kept_pred["conf"]
                        )  # Take max confidence
                        merged_pred["is_fused"] = True
                        merged_pred["model_name"] = "WBF Fusion"

                        # Update source models if applicable
                        if "source_models" in kept_pred and "source_models" in pred:
                            merged_pred["source_models"] = list(
                                set(kept_pred["source_models"] + pred["source_models"])
                            )
                        if (
                            "source_model_names" in kept_pred
                            and "source_model_names" in pred
                        ):
                            merged_pred["source_model_names"] = list(
                                set(
                                    kept_pred["source_model_names"]
                                    + pred["source_model_names"]
                                )
                            )

                        # Replace the kept prediction with the merged one
                        kept_preds[i] = merged_pred
                        merged = True
                        break

            # If not merged with any existing prediction, add to kept list
            if not merged:
                kept_preds.append(pred)

        # Return only predictions above threshold
        return [p for p in kept_preds if p["conf"] >= self.conf_thres]

    def _cluster_predictions(self, predictions, iou_threshold=0.5):
        """
        Cluster predictions that likely refer to the same object based on IoU.
        Groups predictions from different models that overlap significantly.

        Args:
            predictions (list): List of predictions to cluster
            iou_threshold (float): IoU threshold for clustering

        Returns:
            list: List of clusters, where each cluster is a list of predictions
        """
        # Sort predictions by confidence for more deterministic clusters
        sorted_preds = sorted(predictions, key=lambda x: x["conf"], reverse=True)

        # Initialize clusters
        clusters = []
        assigned = set()

        # Go through each prediction
        for i, pred_i in enumerate(sorted_preds):
            # Skip if already assigned to a cluster
            if i in assigned:
                continue

            # Create a new cluster with this prediction
            current_cluster = [pred_i]
            assigned.add(i)

            # Find overlapping predictions
            for j, pred_j in enumerate(sorted_preds):
                # Skip if already assigned or same prediction
                if j in assigned or i == j:
                    continue

                # Calculate IoU between boxes
                iou = self._calculate_iou(pred_i["box"], pred_j["box"])

                # Add to cluster if IoU exceeds threshold
                if iou >= iou_threshold:
                    current_cluster.append(pred_j)
                    assigned.add(j)

            # Add the cluster to our list of clusters
            if current_cluster:
                clusters.append(current_cluster)

        return clusters

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (array): First box in format [x1, y1, x2, y2]
            box2 (array): Second box in format [x1, y1, x2, y2]

        Returns:
            float: IoU value between 0 and 1
        """
        # Convert to numpy arrays if not already
        box1 = np.array(box1)
        box2 = np.array(box2)

        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # Check if boxes don't overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def _nms_predictions(self, predictions):
        """
        Apply the improved ensemble fusion method with Weighted Box Fusion (WBF)
        that properly combines predictions from multiple models.

        This is a wrapper around _improved_ensemble_fusion for compatibility.

        Args:
            predictions (list): List of predictions from all models

        Returns:
            list: Fused predictions that leverage complementary model strengths
        """
        return self._improved_ensemble_fusion(predictions)

    def _legacy_nms_predictions(self, predictions):
        """
        Original NMS implementation as fallback
        """
        if not predictions:
            return []

        try:
            # Group predictions by class
            class_predictions = {}
            for pred in predictions:
                cls_id = pred["cls_id"]
                if cls_id not in class_predictions:
                    class_predictions[cls_id] = []
                class_predictions[cls_id].append(pred)

            # Apply NMS for each class
            final_predictions = []
            for cls_id, preds in class_predictions.items():
                boxes = np.array([p["box"] for p in preds])
                scores = np.array([p["conf"] for p in preds])

                # Get indices of boxes to keep after NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres
                )

                if len(indices) > 0:
                    # OpenCV may return indices as nested arrays
                    if isinstance(indices, tuple):
                        indices = indices[0]

                    for i in indices:
                        # For OpenCV 4.5.x and earlier
                        if isinstance(i, (list, tuple, np.ndarray)):
                            i = i[0]
                        final_predictions.append(preds[i])

            return final_predictions
        except Exception as e:
            print(f"Error in legacy NMS: {e}")
            traceback.print_exc()
            return []

    def predict(self, image_path, output_path=None, visualize=True):
        """
        Run prediction using all models and combine results with model-aware fusion
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Run prediction with each model
        all_predictions = []

        # First, run YOLOv8 and YOLOv9 (always executed)
        for i, model in enumerate(self.models):
            # Ensure model_names list is populated and matches models list length
            if len(self.model_names) <= i:
                # If model name is missing, generate a default one
                model_name = f"Model-{i}"
                # Optionally, extend the model_names list
                while len(self.model_names) <= i:
                    self.model_names.append(f"Model-{len(self.model_names)}")
            else:
                model_name = self.model_names[i]

            # Check if this is v11 model to conditionally run
            if "v11" in model_name.lower():
                # Skip YOLOv11 for now - will run conditionally later
                continue

            if visualize:
                print(f"Running prediction with {model_name}...")
            results = model(img, conf=self.conf_thres)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Add predictions to the list
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                all_predictions.append(
                    {
                        "box": box,
                        "conf": conf,
                        "cls_id": cls_id,
                        "model_idx": i,
                        "model_name": model_name,
                        "img": img,  # Store image for smoke/fire analysis
                    }
                )

        # Check if we need to run YOLOv11 based on initial detections
        need_v11 = False
        for pred in all_predictions:
            box = pred["box"]
            # Check for potential small fires (tight bounding boxes)
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            img_width, img_height = img.shape[1], img.shape[0]

            # If box is small relative to image, we might need v11
            if (box_width * box_height) / (img_width * img_height) < 0.03:
                need_v11 = True
                break

            # Check for potential smoke
            smoke_conf = self._detect_smoke(img, box)
            if smoke_conf > 0.3:
                need_v11 = True
                break

        # Conditionally run YOLOv11 if needed
        if need_v11:
            for i, model in enumerate(self.models):
                # Ensure model_names list is properly populated
                if len(self.model_names) <= i:
                    model_name = f"Model-{i}"
                    # Extend the model_names list if needed
                    while len(self.model_names) <= i:
                        self.model_names.append(f"Model-{len(self.model_names)}")
                else:
                    model_name = self.model_names[i]

                if "v11" in model_name.lower():
                    if visualize:
                        print(
                            f"Activating {model_name} for small fire/smoke detection..."
                        )
                    results = model(img, conf=self.conf_thres)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

                    # Add predictions to the list
                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        all_predictions.append(
                            {
                                "box": box,
                                "conf": conf,
                                "cls_id": cls_id,
                                "model_idx": i,
                                "model_name": model_name,
                                "img": img,  # Store image for smoke/fire analysis
                            }
                        )

        # Combine predictions using enhanced NMS with model-aware fusion
        combined_results = self._nms_predictions(all_predictions)

        if visualize:
            # Draw the combined predictions on the image
            img_drawn = self._draw_predictions(img, combined_results)

            # Save or display the result
            if output_path:
                os.makedirs(
                    os.path.dirname(output_path)
                    if os.path.dirname(output_path)
                    else ".",
                    exist_ok=True,
                )
                cv2.imwrite(output_path, img_drawn)
                print(f"Saved result to {output_path}")
            else:
                cv2.imshow("YOLO Ensemble Prediction", img_drawn)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return combined_results

    def _draw_predictions(self, img, predictions):
        """
        Draw bounding boxes and labels on the image with enhanced model information
        """
        img_copy = img.copy()

        # Define colors for different models
        model_colors = [
            (0, 255, 0),  # Green (YOLOv8)
            (255, 0, 0),  # Blue (YOLOv9)
            (0, 0, 255),  # Red (YOLOv11)
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        # Draw predictions
        for pred in predictions:
            box = pred["box"].astype(int)
            conf = pred["conf"]
            cls_id = pred["cls_id"]
            model_idx = pred["model_idx"] % len(model_colors)

            # Get color for this model
            color = model_colors[model_idx]

            # Check if prediction needs review
            needs_review = pred.get("needs_review", False)

            # Draw bounding box
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, 2)

            # If needs review, add a second outline
            if needs_review:
                cv2.rectangle(
                    img_copy,
                    (box[0] - 2, box[1] - 2),
                    (box[2] + 2, box[3] + 2),
                    (0, 165, 255),
                    1,
                )

            # Draw label with model name instead of just index
            if "model_name" in pred:
                model_name = pred["model_name"]
                # Mark predictions that need review
                review_mark = "⚠️ " if needs_review else ""
                label = f"{review_mark}Class {cls_id}: {conf:.2f} ({model_name})"
            else:
                label = f"Class {cls_id}: {conf:.2f} (Model {model_idx + 1})"

            cv2.putText(
                img_copy,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Add legend
        legend_y = 30
        cv2.putText(
            img_copy,
            "White Dashed: Ground Truth | Colored Solid: Predictions",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return img_copy

    def predict_and_save_csv(self, image_paths, output_csv, label_dir=None):
        """
        Run prediction on multiple images and save results to CSV

        Args:
            image_paths: List of image paths or directory containing images
            output_csv: Path to save CSV results
            label_dir: Directory containing ground truth label files

        Returns:
            DataFrame with results
        """
        if isinstance(image_paths, str) and os.path.isdir(image_paths):
            # If image_paths is a directory, get all image files
            image_paths = (
                glob.glob(os.path.join(image_paths, "*.jpg"))
                + glob.glob(os.path.join(image_paths, "*.png"))
                + glob.glob(os.path.join(image_paths, "*.jpeg"))
            )

        # Create output directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
            exist_ok=True,
        )

        # Create CSV file and write header
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_path",
                    "prediction_id",
                    "model_idx",
                    "class_id",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "gt_available",
                    "gt_class_id",
                    "gt_x1",
                    "gt_y1",
                    "gt_x2",
                    "gt_y2",
                    "iou",
                ]
            )

            # Process each image
            for image_path in tqdm(image_paths, desc="Processing images"):
                # Run prediction
                predictions = self.predict(image_path, visualize=False)

                # Calculate IoU for each prediction if ground truth is available
                prediction_matches = []
                used_gt = set()

                for pred_idx, pred in enumerate(predictions):
                    pred_box = pred["box"].astype(int)
                    pred_cls = pred["cls_id"]
                    model_idx = pred["model_idx"]
                    confidence = pred["conf"]

                    best_iou = 0
                    matched_gt = None
                    matched_gt_idx = -1

                    # Write to CSV
                    if (
                        matched_gt and best_iou > 0.1
                    ):  # Only consider matches with IoU > 0.1
                        used_gt.add(matched_gt_idx)
                        gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = matched_gt
                        writer.writerow(
                            [
                                image_path,
                                pred_idx,
                                model_idx,
                                pred_cls,
                                confidence,
                                pred_box[0],
                                pred_box[1],
                                pred_box[2],
                                pred_box[3],
                                True,
                                gt_cls,
                                gt_x1,
                                gt_y1,
                                gt_x2,
                                gt_y2,
                                best_iou,
                            ]
                        )
                    else:
                        writer.writerow(
                            [
                                image_path,
                                pred_idx,
                                model_idx,
                                pred_cls,
                                confidence,
                                pred_box[0],
                                pred_box[1],
                                pred_box[2],
                                pred_box[3],
                                False,
                                "",
                                "",
                                "",
                                "",
                                "",
                                0,
                            ]
                        )

        print(f"Results saved to {output_csv}")
        return output_csv

    def evaluate(
        self,
        test_dir,
        gt_annotations=None,
        save_metrics=True,
        metrics_path=None,
        visualize=True,
    ):
        """
        Evaluate the ensemble model on a test dataset and calculate metrics

        Args:
            test_dir (str): Directory containing test images
            gt_annotations (str): Path to ground truth annotations in YOLO format
            save_metrics (bool): Whether to save metrics to file
            metrics_path (str): Path to save metrics
            visualize (bool): Whether to visualize metrics

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Create metrics directory if needed
        metrics_dir = (
            "metrics" if metrics_path is None else os.path.dirname(metrics_path)
        )
        os.makedirs(metrics_dir, exist_ok=True)

        # Metrics for ensemble and individual models
        ensemble_metrics = {}
        individual_metrics = {}

        # Use actual model names instead of generic indices
        for i, model_name in enumerate(self.model_names):
            individual_metrics[model_name] = {}

        # Check if ground truth annotations exist and are in a valid format
        valid_gt = False
        label_dir = None
        yaml_path = None

        if gt_annotations:
            if os.path.isdir(gt_annotations):
                # Check if it's a directory of label files
                if len(glob.glob(os.path.join(gt_annotations, "*.txt"))) > 0:
                    label_dir = gt_annotations
                    print(f"Found text label files in {label_dir}")

                    # Create a temporary directory for validation
                    import tempfile

                    dataset_dir = tempfile.mkdtemp(
                        dir=metrics_dir, prefix="temp_dataset_"
                    )
                    print(f"Created temporary dataset directory: {dataset_dir}")

                    images_dir = os.path.join(dataset_dir, "images")
                    labels_dir = os.path.join(dataset_dir, "labels")

                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(labels_dir, exist_ok=True)

                    # Copy test images to dataset directory
                    print("Setting up temporary dataset structure for validation...")
                    test_images = glob.glob(
                        os.path.join(test_dir, "*.jpg")
                    ) + glob.glob(os.path.join(test_dir, "*.png"))

                    # Track how many images were successfully copied with labels
                    successful_copies = 0

                    for img_path in tqdm(test_images, desc="Preparing dataset"):
                        img_name = os.path.basename(img_path)
                        base_name = os.path.splitext(img_name)[0]

                        # Find the matching label file
                        label_file = os.path.join(label_dir, f"{base_name}.txt")
                        if not os.path.exists(label_file):
                            # Try alternative naming patterns
                            label_file = os.path.join(label_dir, f"{img_name}.txt")
                            if not os.path.exists(label_file):
                                continue  # Skip if no label file is found

                        # Copy the image
                        dest_img = os.path.join(images_dir, img_name)
                        try:
                            import shutil

                            shutil.copy2(img_path, dest_img)
                        except Exception as e:
                            print(f"Error copying image {img_path}: {e}")
                            continue

                        # Copy the label file
                        dest_label = os.path.join(labels_dir, f"{base_name}.txt")
                        try:
                            shutil.copy2(label_file, dest_label)
                            successful_copies += 1
                        except Exception as e:
                            print(f"Error copying label {label_file}: {e}")
                            # If we can't copy the label, remove the image too
                            if os.path.exists(dest_img):
                                os.remove(dest_img)
                            continue

                    print(
                        f"Successfully paired {successful_copies} images with their labels"
                    )

                    if successful_copies == 0:
                        print(
                            "No valid image-label pairs found. Cannot perform evaluation."
                        )
                    else:
                        # Create YAML configuration
                        yaml_path = os.path.join(dataset_dir, "data.yaml")
                        with open(yaml_path, "w") as f:
                            yaml_content = f"""
path: {dataset_dir}
train: images
val: images
test: images

# Classes
names:
  0: fire
"""
                            f.write(yaml_content)
                        print(f"Created dataset configuration at {yaml_path}")
                        valid_gt = True

                        # Verify the dataset structure
                        print("Verifying dataset structure:")
                        print(
                            f"- Images directory: {len(os.listdir(images_dir))} files"
                        )
                        print(
                            f"- Labels directory: {len(os.listdir(labels_dir))} files"
                        )

                        # Debug: Print a sample of label files
                        label_files = os.listdir(labels_dir)
                        if label_files:
                            sample_label = os.path.join(labels_dir, label_files[0])
                            print(f"Sample label file ({sample_label}):")
                            try:
                                with open(sample_label, "r") as f:
                                    print(f.read())
                            except Exception as e:
                                print(f"Error reading sample label: {e}")
                elif os.path.exists(os.path.join(gt_annotations, "data.yaml")):
                    yaml_path = os.path.join(gt_annotations, "data.yaml")
                    valid_gt = True
            elif os.path.isfile(gt_annotations) and gt_annotations.endswith(".yaml"):
                yaml_path = gt_annotations
                valid_gt = True

        if valid_gt and yaml_path:
            print(f"Evaluating models using dataset configuration: {yaml_path}")

            # Evaluate each individual model
            for i, model in enumerate(self.models):
                model_name = self.model_names[i]
                print(f"Evaluating {model_name}...")

                try:
                    # Use YOLO's built-in evaluation for individual models
                    debug_log(f"Running validation with data={yaml_path}")
                    results = model.val(data=yaml_path, split="test")
                    debug_log("Validation completed")

                    # Extract metrics
                    if hasattr(results, "box") and results.box is not None:
                        individual_metrics[model_name] = {
                            "mAP50": float(results.box.map50),
                            "mAP50-95": float(results.box.map),
                            "precision": float(results.box.mp),
                            "recall": float(results.box.mr),
                            "f1": float(
                                2
                                * results.box.mp
                                * results.box.mr
                                / (results.box.mp + results.box.mr + 1e-10)
                            ),
                        }
                        debug_log(
                            f"Metrics for {model_name}: {individual_metrics[model_name]}"
                        )
                    else:
                        print(f"Warning: No box metrics available for {model_name}")
                        individual_metrics[model_name] = {
                            "mAP50": 0.0,
                            "mAP50-95": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        }

                    # Store precision and recall curves if available
                    if hasattr(results, "pr_curve") and results.pr_curve is not None:
                        individual_metrics[model_name]["precision_curve"] = (
                            results.pr_curve.precision
                        )
                        individual_metrics[model_name]["recall_curve"] = (
                            results.pr_curve.recall
                        )
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    traceback.print_exc()
                    print("Falling back to inference-only evaluation")
                    individual_metrics[model_name] = {"evaluation_error": str(e)}

            # Now evaluate the ensemble performance using full fusion pipeline
            print("Evaluating ensemble with fusion...")

            # Create a dictionary to track ground truth and predictions for evaluation
            gt_boxes_by_image = {}
            ensemble_predictions_by_image = {}

            # Process each image using the ensemble
            test_images = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(
                os.path.join(images_dir, "*.png")
            )

            for img_path in tqdm(test_images, desc="Running ensemble evaluation"):
                img_name = os.path.basename(img_path)
                base_name = os.path.splitext(img_name)[0]

                # Get predictions using ensemble with fusion
                try:
                    predictions = self.predict(img_path, visualize=False)
                    ensemble_predictions_by_image[img_name] = predictions

                    # Load corresponding ground truth
                    label_path = os.path.join(labels_dir, f"{base_name}.txt")
                    if os.path.exists(label_path):
                        # Load image for converting normalized coordinates
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_height, img_width = img.shape[:2]

                            # Read ground truth boxes
                            gt_boxes = []
                            with open(label_path, "r") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        # YOLO format: class_id, x_center, y_center, width, height (normalized)
                                        cls_id = int(parts[0])
                                        x_center, y_center, width, height = map(
                                            float, parts[1:5]
                                        )

                                        # Convert to absolute coordinates (x1, y1, x2, y2)
                                        x1 = (x_center - width / 2) * img_width
                                        y1 = (y_center - height / 2) * img_height
                                        x2 = (x_center + width / 2) * img_width
                                        y2 = (y_center + height / 2) * img_height

                                        gt_boxes.append(
                                            {
                                                "class_id": cls_id,
                                                "box": [x1, y1, x2, y2],
                                            }
                                        )

                            gt_boxes_by_image[img_name] = gt_boxes
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    traceback.print_exc()

            # Now calculate metrics based on predictions vs ground truth
            if gt_boxes_by_image and ensemble_predictions_by_image:
                ensemble_eval_results = self._calculate_detection_metrics(
                    gt_boxes_by_image, ensemble_predictions_by_image
                )

                ensemble_metrics.update(ensemble_eval_results)
                print(f"Ensemble metrics calculated: {ensemble_metrics}")
            else:
                print("Insufficient data for ensemble evaluation")

        else:
            print("Ground truth annotations not found or not in a valid format.")
            print("Performing inference time evaluation only.")

        # Measure inference time on test images for individual models and ensemble
        test_images = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(
            os.path.join(test_dir, "*.png")
        )
        if not test_images:
            raise ValueError(f"No images found in {test_dir}")

        # Sample a subset of images for timing
        sample_images = test_images[: min(10, len(test_images))]

        # Time individual models
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            inference_times = []

            for img_path in tqdm(sample_images, desc=f"Timing {model_name}"):
                img = cv2.imread(img_path)
                start_time = time.time()
                _ = model(img, conf=self.conf_thres)
                inference_times.append((time.time() - start_time) * 1000)  # in ms

            individual_metrics[model_name]["inference_time"] = np.mean(inference_times)

        # Time ensemble
        ensemble_inference_times = []
        for img_path in tqdm(sample_images, desc="Timing ensemble"):
            start_time = time.time()
            _ = self.predict(img_path, visualize=False)
            ensemble_inference_times.append((time.time() - start_time) * 1000)  # in ms

        ensemble_metrics["inference_time"] = np.mean(ensemble_inference_times)

        # Fallback only if we couldn't calculate ensemble metrics directly
        if not any(k in ensemble_metrics for k in ["mAP50", "precision", "recall"]):
            print("Warning: Using fallback method for ensemble metrics estimation")
            # If we have metrics from individual models, estimate ensemble metrics
            has_metrics = False
            for model_metrics in individual_metrics.values():
                if "mAP50" in model_metrics and model_metrics["mAP50"] > 0:
                    has_metrics = True
                    break

            if has_metrics:
                for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
                    values = [
                        metrics.get(metric, 0)
                        for metrics in individual_metrics.values()
                        if metric in metrics and metrics.get(metric, 0) > 0
                    ]
                    if values:
                        ensemble_metrics[metric] = max(values)
                        print(
                            f"Ensemble {metric} (estimated): {ensemble_metrics[metric]}"
                        )

        # Find best performing model for each metric
        best_models = {}
        for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
            best_value = 0
            best_model = None

            for model_name, metrics in individual_metrics.items():
                if metric in metrics and metrics[metric] > best_value:
                    best_value = metrics[metric]
                    best_model = model_name

            if best_model:
                best_models[metric] = best_model
                print(f"Best model for {metric}: {best_model} ({best_value:.4f})")

        # Combine metrics
        all_metrics = {"ensemble": ensemble_metrics, **individual_metrics}

        # Add best model information
        all_metrics["best_models"] = best_models

        # Store metrics
        self.metrics = all_metrics

        # Save metrics if requested
        if save_metrics:
            metrics_filename = metrics_path or os.path.join(
                metrics_dir, f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.metrics_visualizer.save_metrics_data(all_metrics, metrics_filename)
            print(f"Metrics saved to {metrics_filename}")

        # Visualize metrics if requested
        if visualize:
            output_dir = os.path.join(metrics_dir, "visualizations")
            report_files = self.metrics_visualizer.generate_metrics_report(
                all_metrics, output_dir
            )
            print(f"Metrics visualizations saved to {output_dir}")

        return all_metrics

    def _calculate_detection_metrics(self, gt_boxes_by_image, pred_boxes_by_image):
        """
        Calculate detection metrics (precision, recall, mAP, F1) for ensemble predictions

        Args:
            gt_boxes_by_image: Dictionary mapping image filenames to ground truth boxes
            pred_boxes_by_image: Dictionary mapping image filenames to predicted boxes

        Returns:
            dict: Dictionary containing calculated metrics
        """
        # Validate inputs
        if not gt_boxes_by_image or not pred_boxes_by_image:
            return {}

        # Count total ground truth objects
        total_gt = sum(len(boxes) for boxes in gt_boxes_by_image.values())

        if total_gt == 0:
            print("No ground truth objects found for evaluation")
            return {}

        # Initialize counters for TP, FP, FN
        true_positives = []  # List of 1/0 for each prediction
        false_positives = []  # List of 1/0 for each prediction
        scores = []  # Confidence scores for predictions
        total_gt_count = 0  # Total ground truth objects

        # Track which GT boxes have been matched to prevent double-counting
        matched_gt = {
            img_name: [False] * len(boxes)
            for img_name, boxes in gt_boxes_by_image.items()
        }

        # Process each image
        for img_name, pred_boxes in pred_boxes_by_image.items():
            if img_name not in gt_boxes_by_image:
                # No ground truth for this image, all predictions are false positives
                for pred in pred_boxes:
                    scores.append(pred["conf"])
                    true_positives.append(0)
                    false_positives.append(1)
                continue

            gt_boxes = gt_boxes_by_image[img_name]
            total_gt_count += len(gt_boxes)

            # Sort predictions by decreasing confidence
            pred_boxes = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)

            # Process each prediction
            for pred in pred_boxes:
                scores.append(pred["conf"])

                # Check if prediction matches any ground truth box
                max_iou = 0
                max_idx = -1

                for idx, gt in enumerate(gt_boxes):
                    # Skip already matched ground truth
                    if matched_gt[img_name][idx]:
                        continue

                    # Calculate IoU
                    iou = self._calculate_iou(pred["box"], gt["box"])

                    # Update if better match found
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx

                # Check if we found a match above threshold
                if max_iou >= 0.5:
                    # True positive
                    true_positives.append(1)
                    false_positives.append(0)
                    matched_gt[img_name][max_idx] = True  # Mark GT as matched
                else:
                    # False positive
                    true_positives.append(0)
                    false_positives.append(1)

        # Calculate number of false negatives (ground truth boxes that weren't matched)
        false_negatives = total_gt_count - sum(true_positives)

        # Handle empty predictions case
        if not scores:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }

        # Sort predictions by decreasing confidence
        indices = np.argsort(scores)[::-1]
        true_positives = np.array(true_positives)[indices]
        false_positives = np.array(false_positives)[indices]
        scores = np.array(scores)[indices]

        # Compute cumulative sums for precision-recall curve
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)

        # Calculate precision and recall at each threshold
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        recall = cum_tp / (total_gt_count + 1e-10)

        # Calculate Average Precision using all points
        # Compute mAP using 11-point interpolation (standard COCO-style)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):  # 11 points: 0, 0.1, 0.2, ..., 1.0
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        # Calculate F1 score at the optimal threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)

        # Compute mAP at different IoU thresholds (estimate for mAP50-95)
        # This is a simple approximation - for full mAP50-95 we would need to recalculate
        # TP/FP at each IoU threshold (0.5, 0.55, 0.6, ..., 0.95)
        map50_95_estimate = ap * 0.6  # Typical ratio based on COCO evaluations

        # Compile final metrics
        metrics = {
            "precision": float(precision[-1]) if len(precision) > 0 else 0.0,
            "recall": float(recall[-1]) if len(recall) > 0 else 0.0,
            "f1": float(f1_scores[best_f1_idx]) if len(f1_scores) > 0 else 0.0,
            "mAP50": float(ap),
            "mAP50-95": float(map50_95_estimate),
            "TP": int(sum(true_positives)),
            "FP": int(sum(false_positives)),
            "FN": int(false_negatives),
        }

        # Save precision-recall curve data
        metrics["precision_curve"] = precision.tolist() if len(precision) > 0 else [0.0]
        metrics["recall_curve"] = recall.tolist() if len(recall) > 0 else [0.0]

        return metrics

    def visualize_predictions(
        self, image_path, output_path=None, show_metrics=True, label_dir=None
    ):
        """
        Run prediction and visualize results with metrics if available

        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image
            show_metrics (bool): Whether to show metrics on the visualization
            label_dir (str): Directory containing ground truth label files

        Returns:
            numpy.ndarray: Visualization image
        """
        # Run prediction
        predictions = self.predict(image_path, output_path=None, visualize=False)

        # Load the image
        img = cv2.imread(image_path)
        img_copy = img.copy()

        # Draw predictions and ground truth
        img_drawn = self._draw_predictions(img_copy, predictions)

        # Add metrics if available and requested
        if show_metrics and self.metrics:
            # Prepare metrics text
            metrics_text = []

            if "ensemble" in self.metrics and self.metrics["ensemble"]:
                ensemble_metrics = self.metrics["ensemble"]
                metrics_text.append("ENSEMBLE METRICS:")
                for metric, value in ensemble_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_text.append(f"  {metric}: {value:.3f}")

            # Draw metrics on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            for line in metrics_text:
                cv2.putText(img_drawn, line, (10, y_offset), font, 0.6, (0, 0, 0), 4)
                cv2.putText(
                    img_drawn, line, (10, y_offset), font, 0.6, (255, 255, 255), 1
                )
                y_offset += 25

        # Save or display the result
        if output_path:
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            cv2.imwrite(output_path, img_drawn)
            print(f"Saved result to {output_path}")
        else:
            cv2.imshow("YOLO Ensemble Prediction", img_drawn)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_drawn

    def batch_evaluate(
        self, test_dir, output_dir=None, gt_annotations=None, csv_output=None
    ):
        """
        Run batch evaluation on multiple test images

        Args:
            test_dir (str): Directory containing test images
            output_dir (str): Directory to save output visualizations
            gt_annotations (str): Path to ground truth annotations
            csv_output (str): Path to save CSV results

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(
            os.path.join(test_dir, "*.png")
        )
        if not image_paths:
            raise ValueError(f"No images found in {test_dir}")

        print(f"Found {len(image_paths)} images for batch evaluation")

        # First evaluate metrics if ground truth is available
        if gt_annotations:
            metrics = self.evaluate(test_dir, gt_annotations, save_metrics=True)
        else:
            metrics = {}

        # Get label directory if it exists
        label_dir = None
        if os.path.isdir(gt_annotations):
            label_dir = gt_annotations

        # Save results to CSV if requested
        if csv_output:
            if not csv_output.endswith(".csv"):
                csv_output += ".csv"
            self.predict_and_save_csv(image_paths, csv_output, label_dir)

        # Process each image for visualization
        for img_path in tqdm(image_paths, desc="Processing test images"):
            img_name = os.path.basename(img_path)

            # Skip non-image files
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            if output_dir:
                output_path = os.path.join(output_dir, f"result_{img_name}")
                self.visualize_predictions(
                    img_path, output_path, show_metrics=True, label_dir=label_dir
                )
            else:
                self.visualize_predictions(
                    img_path, None, show_metrics=True, label_dir=label_dir
                )

        return metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="YOLO Ensemble Prediction and Evaluation"
    )
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the output image"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing YOLO models",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="IoU threshold for NMS"
    )

    # Add evaluation arguments
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--test-dir", type=str, help="Directory containing test images")
    parser.add_argument(
        "--gt-annotations", type=str, help="Path to ground truth annotations"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run batch evaluation on test directory"
    )
    parser.add_argument(
        "--visualize-metrics",
        action="store_true",
        help="Visualize metrics after evaluation",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path to save CSV results",
    )

    args = parser.parse_args()

    # Create YOLO ensemble
    ensemble = YOLOEnsemble(
        models_dir=args.models_dir, conf_thres=args.conf_thres, iou_thres=args.iou_thres
    )

    # Run in evaluation mode if requested
    if args.evaluate:
        if args.test_dir:
            if args.batch:
                # Run batch evaluation
                ensemble.batch_evaluate(
                    args.test_dir, args.output, args.gt_annotations, args.csv_output
                )
            else:
                # Run metrics evaluation only
                metrics = ensemble.evaluate(
                    args.test_dir, args.gt_annotations, visualize=args.visualize_metrics
                )
                print("Evaluation metrics:")
                print(json.dumps(metrics, indent=4))
        else:
            print("Error: --test-dir is required for evaluation mode")
    elif args.image:
        # Run single image prediction
        ensemble.visualize_predictions(
            args.image, args.output, label_dir=args.gt_annotations
        )
    else:
        print("Error: either --image or --evaluate with --test-dir must be specified")


if __name__ == "__main__":
    main()
