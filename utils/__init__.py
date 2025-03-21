"""
Utility modules for the Satellite Inferno Detector application.
This package includes utilities for Microsoft Planetary Computer,
CSV processing, and image handling.
"""

# Make all utility functions available at the package level
# Use explicit relative imports for better compatibility with different environments

try:
    # First try explicit relative imports (works better in many environments)
    from .mpc_utils import (
        init_planetary_computer,
        search_satellite_imagery,
        process_satellite_image,
        get_available_collections,
        create_preview_image,
        setup_mpc,
        process_satellite_data,
    )

    from .image_utils import (
        create_tiled_image,
        should_tile_image,
    )

    from .csv_utils import (
        validate_wildfire_data,
        parse_date_column,
        get_common_date_formats,
        validate_satellite_dates,
        detect_numeric_columns,
        detect_columns,
        validate_column_selection,
    )

    # Signal that imports were successful
    __all__ = [
        # MPC utils
        "init_planetary_computer",
        "search_satellite_imagery",
        "process_satellite_image",
        "get_available_collections",
        "create_preview_image",
        "setup_mpc",
        "process_satellite_data",
        # Image utils
        "create_tiled_image",
        "should_tile_image",
        # CSV utils
        "validate_wildfire_data",
        "parse_date_column",
        "get_common_date_formats",
        "validate_satellite_dates",
        "detect_numeric_columns",
        "detect_columns",
        "validate_column_selection",
    ]

except ImportError as e:
    import logging
    import sys
    import os

    logging.warning(f"Error importing utils modules: {e}")

    # Try to fix the import path if possible
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logging.info(f"Added parent directory to path: {parent_dir}")
