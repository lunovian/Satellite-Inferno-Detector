import os
import sys
import argparse
import logging
from typing import Optional, List
from utils.csv_splitter import split_csv, get_csv_info

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Command-line tool to split large CSV files into smaller chunks"""
    parser = argparse.ArgumentParser(
        description="Split large CSV files into smaller chunks"
    )

    parser.add_argument("file", help="Path to the CSV file to split")

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save split files (defaults to same dir as input)",
    )

    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=0,
        help="Number of rows per chunk (default: auto-calculate based on file size)",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        help="Custom prefix for output files (defaults to original filename)",
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Don't include header in each chunk file",
    )

    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show file information without splitting",
    )

    args = parser.parse_args()

    try:
        # Check if file exists
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)

        # Get file info
        info = get_csv_info(args.file)

        # Print file information
        print("\nCSV File Information:")
        print(f"  Path: {info['file_path']}")
        print(f"  Size: {info['file_size_mb']:.2f} MB")
        print(f"  Rows: {info['row_count']:,}")
        print(f"  Columns: {info['column_count']}")
        print(f"  Recommended chunk size: {info['recommended_chunk_size']:,} rows")
        print(f"  Estimated chunks: {info['estimated_chunks']}")

        if args.info_only:
            print("\nColumns:")
            for i, col in enumerate(info["columns"]):
                print(f"  {i + 1}. {col}")
            return

        # Determine chunk size
        chunk_size = (
            args.chunk_size if args.chunk_size > 0 else info["recommended_chunk_size"]
        )

        # Confirm with user
        if not args.output_dir and not args.prefix:
            print(f"\nReady to split into chunks of {chunk_size:,} rows each.")
            response = input("Continue? [y/N]: ").strip().lower()
            if response != "y":
                print("Operation cancelled.")
                return

        # Split the file
        output_files = split_csv(
            input_file=args.file,
            output_dir=args.output_dir,
            chunk_size=chunk_size,
            keep_header=not args.no_header,
            prefix=args.prefix,
        )

        # Print summary
        print(f"\nSplit complete! Created {len(output_files)} files:")
        for i, file_path in enumerate(output_files):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {i + 1}. {os.path.basename(file_path)} ({file_size:.2f} MB)")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
