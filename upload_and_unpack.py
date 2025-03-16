# Upload from local to HuggingFace datasets
# Unpack the dataset from HuggingFace to local

import os
import shutil
import yaml
from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset
import argparse
import logging
from tqdm import tqdm
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def login_huggingface(token=None):
    """Login to HuggingFace using token"""
    api = HfApi()
    if token:
        # Use token parameter directly in API calls rather than setting it on the API object
        logging.info("Using provided token for authentication")
    else:
        logging.info("No token provided. Using HfFolder's cached token if available.")
    return api, token


def upload_to_huggingface(local_dir, dataset_name, token=None):
    """
    Upload train, val, test folders and data.yaml to HuggingFace

    Args:
        local_dir: Directory containing train, val, test folders and data.yaml
        dataset_name: Name of the dataset on HuggingFace (username/dataset_name)
        token: HuggingFace API token (optional if logged in)
    """
    required_folders = ["train", "valid", "test"]
    required_files = ["data.yaml"]

    # Verify required folders and files exist
    for folder in required_folders:
        folder_path = os.path.join(local_dir, folder)
        if not os.path.exists(folder_path):
            logging.error(f"Required folder not found: {folder_path}")
            return False

    for file in required_files:
        file_path = os.path.join(local_dir, file)
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            return False

    # Login to HuggingFace
    api, token = login_huggingface(token)

    try:
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=dataset_name, repo_type="dataset", exist_ok=True, token=token
            )
            logging.info(f"Repository created or already exists: {dataset_name}")
        except Exception as e:
            logging.error(f"Failed to create repository: {e}")
            return False

        # Upload all files
        logging.info("Starting upload...")

        # Upload data.yaml
        yaml_path = os.path.join(local_dir, "data.yaml")
        api.upload_file(
            path_or_fileobj=yaml_path,
            path_in_repo="data.yaml",
            repo_id=dataset_name,
            repo_type="dataset",
            token=token,
        )
        logging.info("Uploaded data.yaml")

        # Upload folders
        for folder in tqdm(required_folders, desc="Uploading folders"):
            folder_path = os.path.join(local_dir, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file_path, local_dir)

                    # Upload file
                    api.upload_file(
                        path_or_fileobj=local_file_path,
                        path_in_repo=rel_path,
                        repo_id=dataset_name,
                        repo_type="dataset",
                        token=token,
                    )

        logging.info(f"Successfully uploaded dataset to {dataset_name}")
        return True

    except Exception as e:
        logging.error(f"Error uploading to HuggingFace: {e}")
        return False


def download_from_huggingface(dataset_name, output_dir, token=None):
    """
    Download and unpack dataset from HuggingFace to local directory

    Args:
        dataset_name: Name of the dataset on HuggingFace (username/dataset_name)
        output_dir: Directory to save the downloaded dataset
        token: HuggingFace API token (optional if logged in)
    """
    try:
        # Login to HuggingFace
        api, token = login_huggingface(token)

        logging.info(f"Downloading dataset {dataset_name} to {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download all files from the repo
        api.snapshot_download(
            repo_id=dataset_name, repo_type="dataset", local_dir=output_dir, token=token
        )

        logging.info(f"Successfully downloaded dataset to {output_dir}")
        return True

    except Exception as e:
        logging.error(f"Error downloading from HuggingFace: {e}")
        return False


def zip_dataset(local_dir, zip_path=None):
    """
    Zip the dataset into a single zip file

    Args:
        local_dir: Directory containing the dataset
        zip_path: Path to save the zip file (default: local_dir + '.zip')

    Returns:
        Path to the created zip file
    """
    if zip_path is None:
        zip_path = f"{local_dir.rstrip('/')}.zip"

    logging.info(f"Zipping dataset at {local_dir} to {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(local_dir))
                zipf.write(file_path, arcname)

    logging.info(f"Dataset zipped successfully to {zip_path}")
    return zip_path


def unzip_dataset(zip_path, output_dir):
    """
    Unzip a dataset zip file

    Args:
        zip_path: Path to the zip file
        output_dir: Directory to extract the zip file to
    """
    logging.info(f"Unzipping {zip_path} to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    logging.info(f"Dataset unzipped successfully to {output_dir}")
    return True


def upload_zip_to_huggingface(zip_path, dataset_name, token=None):
    """
    Upload a zip file to HuggingFace datasets

    Args:
        zip_path: Path to the zip file
        dataset_name: Name of the dataset on HuggingFace (username/dataset_name)
        token: HuggingFace API token (optional if logged in)
    """
    # Login to HuggingFace
    api, token = login_huggingface(token)

    try:
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=dataset_name, repo_type="dataset", exist_ok=True, token=token
            )
            logging.info(f"Repository created or already exists: {dataset_name}")
        except Exception as e:
            logging.error(f"Failed to create repository: {e}")
            return False

        # Upload the zip file
        logging.info(f"Uploading {zip_path} to {dataset_name}")
        filename = os.path.basename(zip_path)

        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=filename,
            repo_id=dataset_name,
            repo_type="dataset",
            token=token,
        )

        logging.info(f"Successfully uploaded zip file to {dataset_name}")
        return True

    except Exception as e:
        logging.error(f"Error uploading to HuggingFace: {e}")
        return False


def download_zip_from_huggingface(
    dataset_name, zip_filename, output_dir, extract=True, token=None
):
    """
    Download a zip file from HuggingFace and optionally extract it

    Args:
        dataset_name: Name of the dataset on HuggingFace (username/dataset_name)
        zip_filename: Name of the zip file in the repository
        output_dir: Directory to save the downloaded zip file
        extract: Whether to extract the zip file after download
        token: HuggingFace API token (optional if logged in)
    """
    try:
        # Login to HuggingFace
        api, token = login_huggingface(token)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download the zip file
        zip_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{zip_filename}"
        zip_path = os.path.join(output_dir, zip_filename)

        logging.info(f"Downloading zip file from {zip_url} to {zip_path}")

        # Use the api to download the file
        api.hf_hub_download(
            repo_id=dataset_name,
            filename=zip_filename,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
        )

        logging.info(f"Successfully downloaded zip file to {zip_path}")

        # Extract if requested
        if extract:
            extract_dir = os.path.join(output_dir, "extracted")
            unzip_dataset(zip_path, extract_dir)

        return True

    except Exception as e:
        logging.error(f"Error downloading from HuggingFace: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload to or download from HuggingFace datasets"
    )
    parser.add_argument(
        "action",
        choices=["upload", "download", "upload_zip", "download_zip"],
        help="Action to perform",
    )
    parser.add_argument(
        "--local_dir",
        help="Local directory containing dataset (for upload) or to save to (for download)",
    )
    parser.add_argument(
        "--dataset_name", help="HuggingFace dataset name (username/dataset_name)"
    )
    parser.add_argument("--token", help="HuggingFace API token (optional if logged in)")
    parser.add_argument(
        "--zip_path",
        help="Path to zip file for upload_zip or filename for download_zip",
    )
    parser.add_argument(
        "--extract", action="store_true", help="Extract zip file after download"
    )

    args = parser.parse_args()

    if args.action == "upload":
        if not args.local_dir or not args.dataset_name:
            parser.error("upload requires --local_dir and --dataset_name")
        upload_to_huggingface(args.local_dir, args.dataset_name, args.token)

    elif args.action == "download":
        if not args.dataset_name or not args.local_dir:
            parser.error("download requires --dataset_name and --local_dir")
        download_from_huggingface(args.dataset_name, args.local_dir, args.token)

    elif args.action == "upload_zip":
        if not args.zip_path or not args.dataset_name:
            parser.error("upload_zip requires --zip_path and --dataset_name")

        if not os.path.exists(args.zip_path):
            # Try to create zip from local_dir if provided
            if args.local_dir and os.path.isdir(args.local_dir):
                args.zip_path = zip_dataset(args.local_dir, args.zip_path)
            else:
                parser.error(f"Zip file not found: {args.zip_path}")

        upload_zip_to_huggingface(args.zip_path, args.dataset_name, args.token)

    elif args.action == "download_zip":
        if not args.dataset_name or not args.zip_path or not args.local_dir:
            parser.error(
                "download_zip requires --dataset_name, --zip_path (filename), and --local_dir"
            )

        download_zip_from_huggingface(
            args.dataset_name,
            args.zip_path,
            args.local_dir,
            extract=args.extract,
            token=args.token,
        )


if __name__ == "__main__":
    main()
