# Satellite Inferno Detector

A project for detecting wildfires and infernos from satellite imagery using YOLO-based object detection models.

## Overview

The Satellite Inferno Detector is designed to identify and locate fires in satellite imagery. This repository provides tools for:

1. Training YOLO models on satellite wildfire detection datasets
2. Uploading and downloading datasets to/from Hugging Face
3. Preprocessing datasets for optimal model training

## Installation

### Requirements

- Python 3.7+
- PyTorch
- Ultralytics YOLO
- Hugging Face libraries

```bash
# Clone the repository
git clone https://github.com/yourusername/Satellite-Inferno-Detector.git
cd Satellite-Inferno-Detector

# Install dependencies
pip install -r requirements.txt
```

## Dataset Management

The repository includes tools for managing datasets via Hugging Face. The `upload_and_unpack.py` script provides functionality for:

### Uploading Datasets

Upload a full dataset (train, valid, test folders and data.yaml) to Hugging Face:

```bash
python upload_and_unpack.py upload --local_dir /path/to/dataset --dataset_name username/dataset-name --token YOUR_HF_TOKEN
```

Upload a zipped dataset (more efficient):

```bash
# Upload an existing zip file
python upload_and_unpack.py upload_zip --zip_path /path/to/dataset.zip --dataset_name username/dataset-name --token YOUR_HF_TOKEN

# Or create a zip from a directory and upload it
python upload_and_unpack.py upload_zip --local_dir /path/to/dataset --dataset_name username/dataset-name --token YOUR_HF_TOKEN
```

### Downloading Datasets

Download a dataset from Hugging Face:

```bash
python upload_and_unpack.py download --dataset_name username/dataset-name --local_dir /path/to/save --token YOUR_HF_TOKEN
```

Download a zip file:

```bash
python upload_and_unpack.py download_zip --dataset_name username/dataset-name --zip_path dataset.zip --local_dir /path/to/save --extract --token YOUR_HF_TOKEN
```

## Training Models

The project uses YOLO models from the Ultralytics framework for training fire detection models.

```bash
python train.py --model v8 --size m --epochs 300 --batch 16 --imgsz 640
```

### Training Parameters

- `--model`: YOLO model version (v3-v12)
- `--size`: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Image size for training
- `--workers`: Number of workers for data loading
- `--device`: Device to use (empty for auto)

### Data Augmentation

Enable satellite-specific data augmentation:

```bash
python train.py --model v8 --size m --augment --aug-factor 3
```

Augmentation parameters:

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
val: valid/images
test: test/images

nc: 1  # Number of classes
names: ['fire']  # Class names
```

## Model Export

After training, models are automatically exported to ONNX format and saved in the `runs/train/` directory.

## License

[Insert your license information here]
