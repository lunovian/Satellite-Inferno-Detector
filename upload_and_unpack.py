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


def main():
    parser = argparse.ArgumentParser(
        description="Upload to or download from HuggingFace datasets"
    )
    parser.add_argument(
        "action", choices=["upload", "download"], help="Action to perform"
    )
    parser.add_argument(
        "--local_dir",
        help="Local directory containing dataset (for upload) or to save to (for download)",
    )
    parser.add_argument(
        "--dataset_name", help="HuggingFace dataset name (username/dataset_name)"
    )
    parser.add_argument("--token", help="HuggingFace API token (optional if logged in)")

    args = parser.parse_args()

    if args.action == "upload":
        if not args.local_dir or not args.dataset_name:
            parser.error("upload requires --local_dir and --dataset_name")
        upload_to_huggingface(args.local_dir, args.dataset_name, args.token)

    elif args.action == "download":
        if not args.dataset_name or not args.local_dir:
            parser.error("download requires --dataset_name and --local_dir")
        download_from_huggingface(args.dataset_name, args.local_dir, args.token)


if __name__ == "__main__":
    main()
