# Satellite Inferno Detector

A Streamlit application for wildfire detection in satellite and drone imagery using YOLO ensemble learning.

## Features

- Upload and analyze satellite/drone images
- Use multiple YOLO models in an ensemble for better detection results
- Interactive UI with model selection and parameter tuning
- Detailed visualization of detection results

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/Satellite-Inferno-Detector.git
cd Satellite-Inferno-Detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your YOLO model files (`.pt`) in the `models` directory.

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser at http://localhost:8501

3. Use the interface to:
   - Select models from the sidebar
   - Adjust detection parameters
   - Upload images
   - Run detection
   - View results

## Model Compatibility

- YOLOv9, YOLOv10, YOLOv11, and YOLOv12 models require Ultralytics 8.1.0+
- YOLOv8 models work with most Ultralytics versions (8.0.0+ recommended)
- YOLOv5 models are backward compatible with newer Ultralytics versions
- See `model_compatibility.md` for detailed information

## License

Copyright © 2025 Satellite Inferno Detector

## Overview

The Satellite Inferno Detector is designed to identify and locate fires in satellite imagery. This repository provides tools for:

1. Training YOLO models on satellite wildfire detection datasets
2. Uploading and downloading datasets to/from Hugging Face
3. Applying specialized data augmentation techniques for satellite imagery

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/Satellite-Inferno-Detector.git
cd Satellite-Inferno-Detector

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- PyTorch
- Ultralytics YOLO
- Hugging Face libraries (huggingface_hub, datasets)
- Albumentations (for data augmentation)
- OpenCV
- PyYAML
- tqdm

## Dataset Management

### Uploading Datasets to Hugging Face

The repository includes tools for uploading datasets to Hugging Face to facilitate model training and dataset sharing.

#### Standard Upload

Upload a full dataset (train, valid, test folders and data.yaml) to Hugging Face:

```bash
python upload_and_unpack.py upload --local_dir /path/to/dataset --dataset_name username/dataset-name --token YOUR_HF_TOKEN
```

#### Zip Upload (More Efficient)

Upload a zipped dataset:

```bash
# Upload an existing zip file
python upload_and_unpack.py upload_zip --zip_path /path/to/dataset.zip --dataset_name username/dataset-name --token YOUR_HF_TOKEN

# Or create a zip from a directory and upload it
python upload_and_unpack.py upload_zip --local_dir /path/to/dataset --dataset_name username/dataset-name --token YOUR_HF_TOKEN
```

### Downloading Datasets from Hugging Face

#### Standard Download

Download a dataset from Hugging Face:

```bash
python upload_and_unpack.py download --dataset_name username/dataset-name --local_dir /path/to/save --token YOUR_HF_TOKEN
```

#### Zip Download

Download a zip file and optionally extract it:

```bash
python upload_and_unpack.py download_zip --dataset_name username/dataset-name --zip_path dataset.zip --local_dir /path/to/save --extract --token YOUR_HF_TOKEN
```

## Data Augmentation

The project includes a specialized data augmentation module for satellite imagery with wildfire-specific transformations.

### How Augmentation Works

The project includes specialized data augmentation for satellite imagery with wildfire detection:

1. **Automatic Augmentation**: By default, the training script will attempt to import the augmentation module, but will proceed with training even if augmentation is unavailable.

2. **Explicit Augmentation**: To explicitly enable augmentation with customized parameters, use the `--augment` flag:

```bash
python train.py --model v8 --size m --augment --aug-factor 3
```

3. **Augmentation Only Runs When Explicitly Enabled**: The training script will only perform dataset augmentation when the `--augment` flag is provided. Without this flag, training will proceed with the original dataset.

4. **Troubleshooting Augmentation Warnings**: If you see warnings like:
   ```
   Warning: Could not import augmentation module. Augmentation will be disabled.
   ```
   This means:
   - Either the albumentations package is not installed
   - Or there was an error importing the augmentation module
   
   To fix this, ensure you've installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Augmentation Transformations

The augmentation pipeline includes:

- Geometric transformations (flips, rotations, scaling)
- Color adjustments (brightness, contrast, hue, saturation)
- Weather simulations (fog, shadows)
- Blur effects (Gaussian blur, median blur)
- Fire-specific adjustments (color jitter)

### Using the Augmentation Module

The augmentation module is automatically used during training when the `--augment` flag is provided.

You can also use it directly to create an augmented dataset:

```python
from augmentation import create_augmented_dataset

# Create an augmented dataset
create_augmented_dataset(
    source_dir="/path/to/original/dataset",
    output_dir="/path/to/augmented/output",
    augmentation_factor=3,  # Generate 3 augmented versions per original image
    input_size=640  # Target image size
)
```

## Training Models

The project uses YOLO models from the Ultralytics framework for training fire detection models.

### Basic Training Command

```bash
python train.py --model v8 --size m --epochs 300 --batch 16 --imgsz 640
```

### Training with Data Augmentation

```bash
python train.py --model v8 --size m --epochs 300 --batch 16 --imgsz 640 --augment --aug-factor 3
```

### Training Parameters

- `--model`: YOLO model version (v3-v12)
- `--size`: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Image size for training
- `--workers`: Number of workers for data loading
- `--device`: Device to use (empty for auto)
- `--augment`: Enable augmentation
- `--aug-factor`: Augmentation factor (samples per image)
- `--aug-dir`: Directory for augmented data (default: runs/augmented)

## Dataset Structure

Expected dataset structure:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file should contain:

```yaml
train: train/images
val: valid/images  # or 'valid: valid/images'
test: test/images

nc: 1  # Number of classes
names: ['fire']  # Class names
```

## Model Export

After training, models are automatically exported to ONNX format and saved in the `runs/train/` directory.

## License


## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Hugging Face](https://huggingface.co/)
- [Albumentations](https://albumentations.ai/)
