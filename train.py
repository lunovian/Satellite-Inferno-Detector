import os
import torch
from ultralytics import YOLO
from pathlib import Path
import glob
import re
import shutil
import argparse
import concurrent.futures
import numpy as np
from tqdm import tqdm

try:
    import yaml
except ImportError:
    print("YAML module not found. Installing pyyaml...")
    import subprocess

    subprocess.check_call(["pip", "install", "pyyaml"])
    import yaml

# Import console utilities
try:
    from utils.console_utils import (
        console,
        print_header,
        print_success,
        print_warning,
        print_error,
        print_info,
        print_section,
        create_progress_bar,
    )
except ImportError:
    # If console utilities are not available, use standard print
    print("Note: Rich console not available, using standard output")
    console = print
    print_header = print_success = print_warning = print_error = print_info = (
        print_section
    ) = print

    def create_progress_bar(*args, **kwargs):
        return None


# Ensure the augmentation module is available
try:
    from augmentation import create_augmented_dataset

    AUGMENTATION_AVAILABLE = True
except ImportError:
    print(
        "Warning: Could not import augmentation module. Augmentation will be disabled."
    )
    AUGMENTATION_AVAILABLE = False

    # Create a dummy function as a fallback
    def create_augmented_dataset(*args, **kwargs):
        print("Augmentation module not available. Skipping augmentation.")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLO model for fire detection")

    # Model version selection with expanded range
    parser.add_argument(
        "--model",
        type=str,
        default="v8",
        choices=["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"],
        help="YOLO model version (v3-v12)",
    )

    # Model size selection
    parser.add_argument(
        "--size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )

    # Additional training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (empty for auto)"
    )

    # Add augmentation parameters
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply satellite wildfire specific augmentations",
    )
    parser.add_argument(
        "--aug-factor",
        type=int,
        default=3,
        help="Augmentation factor (samples per image)",
    )
    parser.add_argument(
        "--aug-dir",
        type=str,
        default="",
        help="Directory for augmented data (default: runs/augmented)",
    )

    return parser.parse_args()


def get_model_path(version="v8", size="m"):
    """Generate model path based on version and size"""
    # Special case for YOLO v3-v5 which use different naming conventions
    if version in ["v3", "v4", "v5"]:
        return f"yolo{version}{size}"  # YOLOv5 uses yolov5s, yolov5m, etc.
    # Special case for v10 formatting
    elif version == "v10":
        return f"yolo{version}-{size}"
    # Special case for v11 and v12 which don't have the "v" prefix
    elif version in ["v11", "v12"]:
        # Remove the "v" prefix for these versions (yolo11n instead of yolov11n)
        version_num = version[1:]  # Extract just the number
        return f"yolo{version_num}{size}"
    # Handle other cases (v6-v9)
    else:
        return f"yolo{version}{size}"


def fix_label_classes(dataset_path, class_count=1):
    """Fix labels that have class IDs greater than the configured number of classes.
    Optimized for speed using parallel processing and batch operations."""
    print_info(f"Checking for label files with class IDs >= {class_count}...")

    # Find all label files recursively
    label_paths = []
    for root, dirs, files in os.walk(dataset_path):
        if "labels" in root:
            label_paths.extend(
                [os.path.join(root, file) for file in files if file.endswith(".txt")]
            )

    print_info(f"Found {len(label_paths)} label files to check")

    # Print first few paths to verify we're finding the right files
    if label_paths and len(label_paths) > 0:
        print("Example label paths found:")
        for path in label_paths[: min(5, len(label_paths))]:
            print(f"  - {path}")

    # Process files in batches for better performance
    batch_size = 500  # Adjust depending on your system's memory
    num_batches = (len(label_paths) + batch_size - 1) // batch_size
    fixed_count = 0

    # Function to process a single file
    def process_file(label_path):
        try:
            needs_fix = False
            # Read the entire file at once
            with open(label_path, "r") as f:
                content = f.read()

            lines = content.strip().split("\n")
            new_lines = []

            for line in lines:
                if not line.strip():  # Skip empty lines
                    new_lines.append(line)
                    continue

                parts = line.strip().split()
                if not parts:
                    new_lines.append(line)
                    continue

                try:
                    class_id = int(parts[0])
                    if class_id >= class_count:
                        # Replace class ID with 0 (the only valid class)
                        parts[0] = "0"
                        new_line = " ".join(parts)
                        needs_fix = True
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                except ValueError:
                    new_lines.append(line)

            if needs_fix:
                # Only write to disk if modifications were made
                backup_path = label_path + ".bak"
                shutil.copy2(label_path, backup_path)

                with open(label_path, "w") as f:
                    f.write("\n".join(new_lines))
                return 1
            return 0
        except Exception as e:
            print_error(f"Error processing {label_path}: {e}")
            return 0

    print_info("Processing label files in parallel...")
    progress = create_progress_bar("Fixing labels")

    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(32, os.cpu_count() * 4)
    ) as executor:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(label_paths))
            batch_paths = label_paths[start_idx:end_idx]

            if progress:
                task = progress.add_task(
                    f"Batch {i + 1}/{num_batches}", total=len(batch_paths)
                )
                futures = {
                    executor.submit(process_file, path): path for path in batch_paths
                }

                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    try:
                        fixed_count += future.result()
                        if progress:
                            progress.update(task, advance=1)
                    except Exception as e:
                        print_error(f"Error processing {path}: {e}")
                        if progress:
                            progress.update(task, advance=1)
            else:
                # Fallback to tqdm if rich progress not available
                results = list(
                    tqdm(
                        executor.map(process_file, batch_paths),
                        total=len(batch_paths),
                        desc=f"Batch {i + 1}/{num_batches}",
                    )
                )
                fixed_count += sum(results)

    if progress:
        progress.stop()

    if fixed_count > 0:
        print_success(f"Fixed {fixed_count} label files with incorrect class IDs")
    else:
        print_info("No files needed fixing")

    return fixed_count


def preprocess_dataset():
    """Perform all preprocessing steps before training"""
    print_section("Dataset Preprocessing")
    dataset_path = os.path.dirname(os.path.abspath(__file__))
    print_info(f"Preprocessing dataset at {dataset_path}")

    # Fix label classes
    fixed_files = fix_label_classes(dataset_path, class_count=1)

    # If files were fixed, give a success message
    if fixed_files > 0:
        print_success(f"Successfully preprocessed dataset: {fixed_files} files updated")
    else:
        print_info("Dataset preprocessing complete. No files needed updating.")


def train_yolo(args):
    """Train YOLO model with specified parameters"""
    print_header("YOLO Satellite Wildfire Detection Training")

    # Suppress verbose Albumentations version messages
    import logging

    logging.getLogger("albumentations").setLevel(logging.WARNING)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")

    # Handle data augmentation if requested
    data_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")

    # Only run augmentation when explicitly enabled with --augment flag
    if args.augment:
        if not AUGMENTATION_AVAILABLE:
            print_warning(
                "Augmentation was requested but is not available. Will proceed with original dataset."
            )
            print_warning(
                "Install required packages with: pip install albumentations opencv-python"
            )
        else:
            print_section("Data Augmentation")
            print_info("Running satellite-specific data augmentation...")

            # Ensure we use absolute path for augmentation directory
            if args.aug_dir:
                aug_dir = os.path.abspath(args.aug_dir)
            else:
                aug_dir = os.path.abspath(os.path.join("runs", "augmented"))

            os.makedirs(aug_dir, exist_ok=True)
            print_info(f"Augmentation directory: {aug_dir}")

            # Get source dataset directory from data.yaml
            try:
                with open(data_yaml, "r") as f:
                    try:
                        data_config = yaml.safe_load(f)
                        if data_config is None:
                            print_warning("Empty or invalid YAML file")
                            data_config = {}
                    except Exception as e:
                        print_error(f"Error parsing YAML: {e}")
                        data_config = {}

                # Get the source directory, with fallback to current directory
                source_dir = os.path.dirname(os.path.abspath(__file__))
                print_info(f"Source dataset directory: {source_dir}")

                # Create augmented dataset and save to disk
                create_augmented_dataset(
                    source_dir=source_dir,
                    output_dir=aug_dir,
                    augmentation_factor=args.aug_factor,
                    input_size=args.imgsz,
                )

                # Use augmented data.yaml
                data_yaml = os.path.join(aug_dir, "data.yaml")

                # Verify the augmented data.yaml file
                if os.path.exists(data_yaml):
                    with open(data_yaml, "r") as f:
                        aug_config = yaml.safe_load(f)
                        print_info(f"Augmented train path: {aug_config.get('train')}")
                        print_info(f"Augmented val path: {aug_config.get('val')}")

                    print_success(f"Using augmented dataset config: {data_yaml}")
                else:
                    print_error(f"Augmented data.yaml not found at {data_yaml}")
                    raise FileNotFoundError(
                        f"Augmented data.yaml not found at {data_yaml}"
                    )

            except Exception as e:
                print_error(f"Error during augmentation setup: {e}")
                print_warning("Proceeding with original dataset")
                data_yaml = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "data.yaml"
                )
    else:
        print_info("Augmentation not requested. Using original dataset.")

    # Preprocessing step - handle all labels before training
    preprocess_dataset()
    print_info("Preprocessing complete, starting model training")

    # Get model path based on version and size
    model_path = get_model_path(args.model, args.size)
    print_info(f"Using model: {model_path}")

    # Add explanation about model loading
    print_info("Note: YOLO() can directly create or load models:")
    print_info("  - For new model: model = YOLO('yoloXXn.yaml')")
    print_info("  - For pretrained: model = YOLO('yoloXXn.pt')")

    # Simplify model loading - direct approach without extensive fallbacks
    yaml_path = f"{model_path}.yaml"
    pt_path = f"{model_path}.pt"

    print_info(f"Attempting to load {args.model} model...")

    try:
        # First, try to create from YAML - simplest approach
        print_info(f"Creating new model from configuration: {yaml_path}")
        model = YOLO(yaml_path)
        print_success(f"Successfully created model from {yaml_path}")
    except Exception as yaml_error:
        print_warning(f"Could not create model from YAML: {yaml_error}")

        # If YAML fails, try loading pretrained weights
        try:
            print_info(f"Trying to load pretrained weights: {pt_path}")
            model = YOLO(pt_path)
            print_success(f"Successfully loaded pretrained model: {pt_path}")
        except Exception as pt_error:
            print_error(f"Could not load pretrained model: {pt_error}")
            raise ValueError(
                f"Failed to create or load YOLO model. Error with YAML: {yaml_error}, Error with PT: {pt_error}"
            )

    print_info("Task set to detection - using only bounding boxes (ignoring segments)")

    # Use the correct data.yaml path
    print_info(f"Using data configuration: {data_yaml}")

    # Check if config file exists
    if not os.path.exists(data_yaml):
        print_error(f"Data configuration file not found: {data_yaml}")
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")

    # Get model version for run name (use format consistent with model type)
    if args.model in ["v3", "v4", "v5"]:
        model_version = f"yolo{args.model}{args.size}"
    elif args.model == "v10":
        model_version = f"yolo{args.model}-{args.size}"
    # Handle v11-v12 without the "v" prefix for the run name
    elif args.model in ["v11", "v12"]:
        version_num = args.model[1:]  # Extract just the number
        model_version = f"yolo{version_num}_{args.size}"
    else:
        model_version = f"yolo{args.model}_{args.size}"

    # Train the model
    print_section("Training Model")
    print_info(f"Starting training with {args.epochs} epochs, batch size {args.batch}")

    # Configure environment variables to reduce verbosity
    os.environ["ALBUMENTATIONS_QUIET"] = "1"  # Try to suppress albumentations messages

    # Unified training approach for all versions
    try:
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            project="runs/train",
            name=f"{model_version}_fire_detector",
            patience=50,
            save=True,
            exist_ok=False,
            task="detect",
            single_cls=True,
            verbose=False,  # Reduce verbosity
        )
    except TypeError as e:
        print_warning(
            f"Training parameter error: {e}. Trying with simplified parameters."
        )
        # Fallback with fewer parameters for older YOLO versions
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project="runs/train",
            name=f"{model_version}_fire_detector",
            verbose=False,  # Reduce verbosity
        )

    # Validate the model
    print_section("Validating Model")
    val_results = model.val()
    print_success(f"Validation results: {val_results}")

    # Export the model to different formats
    print_section("Exporting Model")
    print_info("Exporting to ONNX format...")
    model.export(format="onnx")
    print_success("Export complete")

    return model, results


if __name__ == "__main__":
    try:
        args = parse_arguments()
        model, results = train_yolo(args)
        print_success(
            f"Training of YOLO{args.model}-{args.size} completed successfully!"
        )
    except Exception as e:
        print_error(f"Training failed: {e}")
        raise
