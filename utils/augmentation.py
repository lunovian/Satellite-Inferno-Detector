import os
import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    print("Installing albumentations...")
    import subprocess

    subprocess.check_call(["pip", "install", "albumentations"])
    import albumentations as A

# Print version info to help diagnose issues
try:
    print(f"Albumentations version: {A.__version__}")
except AttributeError:
    print("Albumentations version information not available")

from pathlib import Path
import random
import shutil
from tqdm import tqdm
import glob
import yaml

# Import console utilities
try:
    from console_utils import (
        console,
        print_header,
        print_success,
        print_warning,
        print_error,
        print_info,
        print_section,
        create_progress_bar,
    )

    has_rich_console = True
except ImportError:
    # If console utilities are not available, use standard print
    print("Note: Rich console not available, using standard output")
    console = print
    print_header = print_success = print_warning = print_error = print_info = (
        print_section
    ) = print
    has_rich_console = False

    def create_progress_bar(*args, **kwargs):
        return None


class SatelliteFireAugmentation:
    """
    Specialized data augmentation for satellite wildfire detection.
    Handles unique characteristics of satellite imagery and fire patterns.
    """

    def __init__(self, config=None):
        """
        Initialize augmentation pipeline with configuration.

        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config or {}
        self.input_size = self.config.get("input_size", 640)

        # Define augmentation pipeline specifically for satellite imagery
        # Using only transformations that are widely available across albumentations versions
        self.transform = A.Compose(
            [
                # Spatial augmentations - basic transformations available in all versions
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                # Use Affine instead of ShiftScaleRotate as recommended
                A.Affine(
                    scale=(0.8, 1.2),  # Scale limit
                    translate_percent=0.1,  # Shift limit
                    rotate=(-45, 45),  # Rotate limit
                    interpolation=cv2.INTER_LINEAR,
                    mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                # Satellite imagery specific augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.7
                ),
                # Weather/atmospheric condition simulation
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # Atmospheric blur
                        A.GaussNoise(
                            p=1.0
                        ),  # Fixed: removed incorrect var_limit parameter
                        A.MultiplicativeNoise(
                            multiplier=(0.9, 1.1), p=1.0, elementwise=True
                        ),  # Scattered clouds
                    ],
                    p=0.3,
                ),
                # Wildfire-specific color adjustments
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=15,
                            val_shift_limit=10,
                            p=1.0,
                        ),
                        A.RGBShift(p=1.0),
                    ],
                    p=0.5,
                ),
                # Fire intensity simulation - using only ToGray which is available in all versions
                A.ToGray(p=0.2),  # Smoke-covered areas
                # Preserve aspect ratio while resizing
                A.LongestMaxSize(max_size=self.input_size, p=1.0),
                A.PadIfNeeded(
                    min_height=self.input_size,
                    min_width=self.input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.3, label_fields=["class_labels"]
            ),
        )

        # Fire-specific intensity augmentation - using only core transformations
        self.fire_intensity_transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),  # Boost brightness for fire regions
                    contrast_limit=(0.1, 0.3),  # Enhance contrast for fire regions
                    p=0.8,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,  # Preserve fire color range
                    sat_shift_limit=15,  # Boost saturation for fire visibility
                    val_shift_limit=15,  # Enhance value for fire intensity
                    p=0.7,
                ),
            ]
        )

    def augment_dataset(self, dataset_dir, output_dir, augmentation_factor=2):
        """
        Augment an entire YOLO dataset with images and labels.

        Args:
            dataset_dir: Root directory of the dataset with train/val folders
            output_dir: Output directory for augmented dataset
            augmentation_factor: Number of augmented samples per original image
        """
        print(f"Augmenting dataset from {dataset_dir} to {output_dir}")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)

        # Copy data.yaml if it exists
        yaml_src = os.path.join(dataset_dir, "data.yaml")
        if os.path.exists(yaml_src):
            yaml_dst = os.path.join(output_dir, "data.yaml")

            # Read existing yaml
            with open(yaml_src, "r") as f:
                yaml_data = yaml.safe_load(f)

            # Update paths in yaml
            for key in ["train", "val", "test"]:
                if key in yaml_data:
                    # Replace dataset_dir with output_dir in the path
                    yaml_data[key] = yaml_data[key].replace(dataset_dir, output_dir)

            # Save updated yaml
            with open(yaml_dst, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False)

            print(f"Updated data.yaml with new paths")

        # Process each split (train, val)
        for split in ["train", "valid", "test"]:
            src_img_dir = os.path.join(dataset_dir, split, "images")
            src_label_dir = os.path.join(dataset_dir, split, "labels")

            # Skip if directory doesn't exist
            if not os.path.exists(src_img_dir):
                continue

            dst_img_dir = os.path.join(output_dir, split, "images")
            dst_label_dir = os.path.join(output_dir, split, "labels")

            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_label_dir, exist_ok=True)

            # First, copy all original files
            print(f"Copying original {split} files...")
            for img_file in glob.glob(os.path.join(src_img_dir, "*")):
                img_name = os.path.basename(img_file)

                # Copy image
                shutil.copy2(img_file, os.path.join(dst_img_dir, img_name))

                # Copy corresponding label if it exists
                base_name = os.path.splitext(img_name)[0]
                label_file = os.path.join(src_label_dir, base_name + ".txt")
                if os.path.exists(label_file):
                    shutil.copy2(
                        label_file, os.path.join(dst_label_dir, base_name + ".txt")
                    )

            # Then generate augmented samples
            if augmentation_factor > 0:
                print(
                    f"Generating {augmentation_factor} augmented samples per image for {split}..."
                )
                self._augment_split(
                    src_img_dir,
                    src_label_dir,
                    dst_img_dir,
                    dst_label_dir,
                    augmentation_factor,
                )

    def _augment_split(
        self,
        src_img_dir,
        src_label_dir,
        dst_img_dir,
        dst_label_dir,
        augmentation_factor,
    ):
        """Augment images and labels for a specific split"""
        img_files = glob.glob(os.path.join(src_img_dir, "*"))

        for img_file in tqdm(img_files, desc="Augmenting"):
            img_name = os.path.basename(img_file)
            base_name = os.path.splitext(img_name)[0]
            label_file = os.path.join(src_label_dir, base_name + ".txt")

            # Skip if no label file exists
            if not os.path.exists(label_file):
                continue

            # Read image and label
            image = cv2.imread(img_file)
            if image is None:
                print(f"Warning: Could not read {img_file}, skipping.")
                continue

            # Read bbox annotations from YOLO format
            with open(label_file, "r") as f:
                lines = f.readlines()

            bboxes = []
            class_labels = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

            # Generate augmented samples
            for i in range(augmentation_factor):
                # Apply spatial augmentation to image and bboxes
                augmented = self.transform(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )

                aug_image = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_class_labels = augmented["class_labels"]

                # Create augmented file names
                aug_img_name = f"{base_name}_aug{i + 1}{os.path.splitext(img_name)[1]}"
                aug_label_name = f"{base_name}_aug{i + 1}.txt"

                # Save augmented image
                cv2.imwrite(os.path.join(dst_img_dir, aug_img_name), aug_image)

                # Save augmented label
                with open(os.path.join(dst_label_dir, aug_label_name), "w") as f:
                    for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                        x_center, y_center, width, height = bbox
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def create_augmented_dataset(
    source_dir, output_dir, augmentation_factor=3, input_size=640
):
    """
    Create an augmented dataset ready for YOLO training

    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for augmented dataset
        augmentation_factor: Number of augmentations per image
        input_size: Image size for resizing
    """
    print_header("Satellite Fire Image Augmentation")
    print_info(f"Source directory: {source_dir}")
    print_info(f"Output directory: {output_dir}")
    print_info(f"Augmentation factor: {augmentation_factor}x")
    print_info(f"Image size: {input_size}px")

    config = {
        "input_size": input_size,
    }

    augmenter = SatelliteFireAugmentation(config)
    augmenter.augment_dataset(
        dataset_dir=source_dir,
        output_dir=output_dir,
        augmentation_factor=augmentation_factor,
    )

    print_success(f"Augmentation complete. Augmented dataset created at {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create augmented satellite wildfire dataset"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Source dataset directory"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--factor",
        type=int,
        default=3,
        help="Augmentation factor (new samples per image)",
    )
    parser.add_argument("--size", type=int, default=640, help="Input image size")

    args = parser.parse_args()

    create_augmented_dataset(
        source_dir=args.source,
        output_dir=args.output,
        augmentation_factor=args.factor,
        input_size=args.size,
    )
