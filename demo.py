import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import glob
import argparse


class YOLOEnsemble:
    def __init__(self, models_dir="models", conf_thres=0.3, iou_thres=0.5):
        """
        Initialize the YOLO ensemble with models from the specified directory
        """
        self.models = []
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Only load models from directory if a directory is specified
        if models_dir:
            # Load all YOLO models from the models directory
            model_files = glob.glob(os.path.join(models_dir, "*.pt"))
            if not model_files:
                raise ValueError(f"No model files found in {models_dir}")

            print(f"Loading {len(model_files)} YOLO models...")
            for model_path in model_files:
                model_name = os.path.basename(model_path)
                print(f"Loading {model_name}...")
                model = YOLO(model_path)
                self.models.append(model)
            print("All models loaded successfully!")

    def predict(self, image_path, output_path=None, visualize=True):
        """
        Run prediction using all models and combine results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Run prediction with each model
        all_predictions = []
        for i, model in enumerate(self.models):
            print(f"Running prediction with model {i + 1}/{len(self.models)}...")
            results = model(img, conf=self.conf_thres)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Add predictions to the list
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                all_predictions.append(
                    {"box": box, "conf": conf, "cls_id": cls_id, "model_idx": i}
                )

        # Combine predictions using non-maximum suppression
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

    def _nms_predictions(self, predictions):
        """
        Apply non-maximum suppression to combine predictions from multiple models
        """
        if not predictions:
            return []

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

    def _draw_predictions(self, img, predictions):
        """
        Draw bounding boxes and labels on the image
        """
        img_copy = img.copy()

        # Define colors for different models
        model_colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for pred in predictions:
            box = pred["box"].astype(int)
            conf = pred["conf"]
            cls_id = pred["cls_id"]
            model_idx = pred["model_idx"] % len(model_colors)

            # Get color for this model
            color = model_colors[model_idx]

            # Draw bounding box
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw label
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

        return img_copy


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLO Ensemble Prediction")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
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
    args = parser.parse_args()

    # Create YOLO ensemble
    ensemble = YOLOEnsemble(
        models_dir=args.models_dir, conf_thres=args.conf_thres, iou_thres=args.iou_thres
    )

    # Run prediction
    ensemble.predict(args.image, args.output)


if __name__ == "__main__":
    main()
