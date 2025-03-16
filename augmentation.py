"""
Augmentation module for the Satellite Inferno Detector.
Provides satellite imagery specific augmentations for wildfire detection.
"""

import os
import yaml
import random
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import cv2
import glob
import logging

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    logging.warning(
        "Albumentations not installed. Please install with: pip install albumentations"
    )
    ALBUMENTATIONS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_augmentation_pipeline(input_size=640):
    """
    Create an augmentation pipeline using Albumentations.

    Args:
        input_size: Size for resizing images (default: 640x640)

    Returns:
        Albumentations transform pipeline
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logging.error(
            "Cannot create augmentation pipeline: Albumentations not installed"
        )
        return None

    # General augmentations suitable for satellite imagery
    transform = A.Compose(
        [
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=15
            ),
            # Color augmentations
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                p=0.3, hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10
            ),
            # Weather and environmental simulations for satellite imagery
            A.RandomFog(p=0.01, fog_coef_lower=0.1, fog_coef_upper=0.2),
            A.RandomShadow(p=0.01, shadow_roi=(0, 0, 1, 1)),
            # Blur augmentations (as specified in your prompt)
            A.Blur(p=0.01, blur_limit=(3, 7)),
            A.MedianBlur(p=0.01, blur_limit=(3, 7)),
            A.ToGray(p=0.01, num_output_channels=3, method="weighted_average"),
            A.CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
            # Fire-specific augmentations
            A.ColorJitter(
                p=0.2,
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.8, 1.2],
                hue=0.2,
            ),
            # Resize to target size
            A.Resize(height=input_size, width=input_size, p=1.0),
        ],
        # This will tell albumentations that we're working with bounding box annotations
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    return transform


def parse_yolo_label(label_path):
    """
    Parse YOLO format labels from a file

    Args:
        label_path: Path to the label file

    Returns:
        List of bounding boxes in YOLO format (class_id, x_center, y_center, width, height)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return []

    with open(label_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    bboxes.append([class_id, x_center, y_center, width, height])
    return bboxes


def apply_augmentation(image_path, label_path, transform):
    """
    Apply augmentation to an image and its labels

    Args:
        image_path: Path to the image file
        label_path: Path to the label file
        transform: Albumentations transform pipeline

    Returns:
        Augmented image and bounding boxes
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image: {image_path}")
        return None, None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read bounding boxes
    yolo_bboxes = parse_yolo_label(label_path)
    if not yolo_bboxes:
        # If no bounding boxes, still augment the image
        aug = transform(image=image, bboxes=[], class_labels=[])
        return aug["image"], []

    # Extract bounding box coordinates and class labels
    bboxes = []
    class_labels = []

    for bbox in yolo_bboxes:
        class_id, x_center, y_center, width, height = bbox
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)

    # Apply augmentation
    aug = transform(image=image, bboxes=bboxes, class_labels=class_labels)

    # Format the augmented bounding boxes back to YOLO format
    aug_bboxes = []
    for i, bbox in enumerate(aug["bboxes"]):
        x_center, y_center, width, height = bbox
        class_id = aug["class_labels"][i]
        aug_bboxes.append([class_id, x_center, y_center, width, height])

    return aug["image"], aug_bboxes


def save_augmented_data(image, bboxes, img_output_path, label_output_path):
    """
    Save augmented image and labels

    Args:
        image: Augmented image (RGB format)
        bboxes: Augmented bounding boxes in YOLO format
        img_output_path: Path to save the augmented image
        label_output_path: Path to save the augmented labels
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)

    # Convert RGB to BGR for saving with OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_output_path, image_bgr)

    # Save the bounding boxes
    with open(label_output_path, "w") as f:
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")


def create_augmented_dataset(
    source_dir, output_dir, augmentation_factor=3, input_size=640
):
    """
    Create an augmented dataset based on the source dataset

    Args:
        source_dir: Path to the source dataset directory
        output_dir: Path to save the augmented dataset
        augmentation_factor: Number of augmented samples to generate per original sample
        input_size: Size for resizing images

    Returns:
        Path to the augmented dataset's data.yaml file
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logging.error(
            "Augmentation requires Albumentations. Install with: pip install albumentations"
        )
        return None

    logging.info(f"Creating augmented dataset from {source_dir} to {output_dir}")
    logging.info(f"Augmentation factor: {augmentation_factor}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find the data.yaml file
    data_yaml_path = os.path.join(source_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        logging.error(f"data.yaml not found at {data_yaml_path}")
        return None

    # Read the data.yaml file
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    # Create augmentation pipeline
    transform = get_augmentation_pipeline(input_size=input_size)
    if transform is None:
        return None

    # Process each dataset split (train, valid, test)
    for split in ["train", "val", "test"]:
        # Handle different keys in data.yaml ('val' vs 'valid')
        yaml_key = split
        if split == "val" and "valid" in data_config:
            yaml_key = "valid"
        elif split == "valid" and "val" in data_config:
            yaml_key = "val"

        if yaml_key not in data_config:
            logging.warning(f"Split '{yaml_key}' not found in data.yaml")
            continue

        # Get the split directory
        split_dir = data_config[yaml_key]
        if not os.path.isabs(split_dir):
            split_dir = os.path.join(source_dir, split_dir)

        # Output directories
        output_imgs_dir = os.path.join(output_dir, split, "images")
        output_labels_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(output_imgs_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # Find all images
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        img_files = []
        for ext in img_extensions:
            img_files.extend(
                glob.glob(os.path.join(split_dir, f"**/*{ext}"), recursive=True)
            )
            img_files.extend(
                glob.glob(os.path.join(split_dir, f"**/*{ext.upper()}"), recursive=True)
            )

        logging.info(f"Found {len(img_files)} images in {split_dir}")

        # For test split, just copy the data
        if split == "test":
            logging.info(f"Copying test data without augmentation")
            for img_path in tqdm(img_files, desc=f"Copying test data"):
                # Get corresponding label path
                label_path = (
                    img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
                )

                # Output paths
                img_output_path = os.path.join(
                    output_imgs_dir, os.path.basename(img_path)
                )
                label_output_path = os.path.join(
                    output_labels_dir, os.path.basename(label_path)
                )

                # Copy files
                shutil.copy2(img_path, img_output_path)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, label_output_path)

            continue

        # Process train and validation splits with augmentation
        logging.info(f"Augmenting {split} data")
        for img_idx, img_path in enumerate(
            tqdm(img_files, desc=f"Augmenting {split} data")
        ):
            # Get the corresponding label path
            label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

            # Copy the original image and label
            img_output_path = os.path.join(output_imgs_dir, os.path.basename(img_path))
            label_output_path = os.path.join(
                output_labels_dir, os.path.basename(label_path)
            )

            shutil.copy2(img_path, img_output_path)
            if os.path.exists(label_path):
                shutil.copy2(label_path, label_output_path)

            # Only augment training data
            if split != "test":
                # Create augmented versions
                for aug_idx in range(augmentation_factor):
                    # Generate augmented filename
                    base_name = os.path.basename(img_path).rsplit(".", 1)[0]
                    ext = os.path.basename(img_path).rsplit(".", 1)[1]
                    aug_img_name = f"{base_name}_aug{aug_idx}.{ext}"
                    aug_label_name = f"{base_name}_aug{aug_idx}.txt"

                    # Output paths for augmented files
                    aug_img_path = os.path.join(output_imgs_dir, aug_img_name)
                    aug_label_path = os.path.join(output_labels_dir, aug_label_name)

                    # Apply augmentation
                    aug_image, aug_bboxes = apply_augmentation(
                        img_path, label_path, transform
                    )

                    if aug_image is not None:
                        # Save augmented data
                        save_augmented_data(
                            aug_image, aug_bboxes, aug_img_path, aug_label_path
                        )

    # Create the new data.yaml
    output_yaml_path = os.path.join(output_dir, "data.yaml")

    # Update the paths in data.yaml
    new_data_config = data_config.copy()
    new_data_config["train"] = os.path.join("train", "images")

    # Handle val/valid key in the same way as the original
    if "val" in data_config:
        new_data_config["val"] = os.path.join("val", "images")
    if "valid" in data_config:
        new_data_config["valid"] = os.path.join("valid", "images")

    new_data_config["test"] = os.path.join("test", "images")

    # Write the new data.yaml
    with open(output_yaml_path, "w") as f:
        yaml.dump(new_data_config, f, default_flow_style=False)

    logging.info(f"Augmented dataset created at {output_dir}")
    logging.info(f"Data configuration saved to {output_yaml_path}")

    return output_yaml_path
