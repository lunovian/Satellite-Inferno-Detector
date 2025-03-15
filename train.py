import os
import torch
from ultralytics import YOLO
from pathlib import Path
import glob
import re
import shutil
import argparse


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLO model for fire detection")

    # Model version selection
    parser.add_argument(
        "--model",
        type=str,
        default="v8",
        choices=["v8", "v10", "v11"],
        help="YOLO model version (v8, v10, v11)",
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
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (empty for auto)"
    )

    return parser.parse_args()


def get_model_path(version="v8", size="m"):
    """Generate model path based on version and size"""
    # Handle special case for v10 formatting
    if version == "v10":
        return f"yolo{version}-{size}.pt"
    # Handle normal cases (v8, v11)
    return f"yolo{version}{size}.pt"


def fix_label_classes(dataset_path, class_count=1):
    """Fix labels that have class IDs greater than the configured number of classes."""
    print(f"Checking for label files with class IDs >= {class_count}...")

    # Find all label files recursively
    label_paths = []
    for root, dirs, files in os.walk(dataset_path):
        if "labels" in root:
            for file in files:
                if file.endswith(".txt"):
                    label_paths.append(os.path.join(root, file))

    print(f"Found {len(label_paths)} label files to check")

    # Print first few paths to verify we're finding the right files
    if label_paths:
        print("Example label paths found:")
        for path in label_paths[:5]:
            print(f"  - {path}")

    fixed_count = 0
    for label_path in label_paths:
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            modified = False
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:  # Skip empty lines
                    new_lines.append(line)
                    continue

                try:
                    class_id = int(parts[0])
                    if class_id >= class_count:
                        # Replace class ID with 0 (the only valid class)
                        parts[0] = "0"
                        new_line = " ".join(parts) + "\n"
                        modified = True
                        new_lines.append(new_line)
                        # Print which files are being modified
                        if not modified:  # Only print once per file
                            print(
                                f"Fixing class IDs in: {os.path.basename(label_path)}"
                            )
                    else:
                        new_lines.append(line)
                except ValueError:
                    # If we can't convert to int, keep the line as is
                    new_lines.append(line)

            if modified:
                # Create a backup of the original file
                backup_path = label_path + ".bak"
                shutil.copy2(label_path, backup_path)

                # Write the modified content
                with open(label_path, "w") as f:
                    f.writelines(new_lines)
                fixed_count += 1

        except Exception as e:
            print(f"Error processing {label_path}: {e}")

    print(f"Fixed {fixed_count} label files with incorrect class IDs")
    return fixed_count


def preprocess_dataset():
    """Perform all preprocessing steps before training"""
    dataset_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Preprocessing dataset at {dataset_path}")

    # Fix label classes
    fixed_files = fix_label_classes(dataset_path, class_count=1)

    # If files were fixed, give a success message
    if fixed_files > 0:
        print(f"Successfully preprocessed dataset: {fixed_files} files updated")
    else:
        print("Dataset preprocessing complete. No files needed updating.")


def train_yolo(args):
    """Train YOLO model with specified parameters"""
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocessing step - handle all labels before training
    print("Starting dataset preprocessing...")
    preprocess_dataset()
    print("Preprocessing complete, starting model training")

    # Get model path based on version and size
    model_path = get_model_path(args.model, args.size)
    print(f"Using model: {model_path}")

    # Try to load or download the model
    try:
        model = YOLO(model_path, task="detect")
        print(f"Successfully loaded/downloaded {model_path}")
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        # Fallback to YAML if PT file not available
        yaml_path = model_path.replace(".pt", ".yaml")
        try:
            print(f"Falling back to configuration file: {yaml_path}")
            model = YOLO(yaml_path)
        except Exception as yaml_error:
            print(f"Error with fallback: {yaml_error}")
            print("Falling back to default YOLOv8m model")
            model = YOLO("yolov8m.pt", task="detect")

    print("Task set to detection - using only bounding boxes (ignoring segments)")

    # Use the correct data.yaml path
    data_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")
    print(f"Using data configuration: {data_yaml}")

    # Check if config file exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")

    # Get model version for run name
    model_version = f"yolo{args.model}_{args.size}"

    # Train the model
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
        mode="train",
        rect=False,
        overlap_mask=False,
        mask_ratio=4,
        single_cls=True,
    )

    # Validate the model
    val_results = model.val()
    print(f"Validation results: {val_results}")

    # Export the model to different formats
    model.export(format="onnx")

    return model, results


if __name__ == "__main__":
    args = parse_arguments()
    model, results = train_yolo(args)
    print(f"Training of YOLO{args.model}-{args.size} completed successfully!")
