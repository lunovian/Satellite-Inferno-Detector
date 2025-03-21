"""
Utility modules for the Satellite Inferno Detector application.
This package includes utilities for Microsoft Planetary Computer,
CSV processing, and image handling.
"""

# Import all utility functions to make them available via utils.*
try:
    from .mpc_utils import (
        init_planetary_computer,
        search_satellite_imagery,
        process_satellite_image,
        get_available_collections,
        create_preview_image,
    )

    from .image_utils import (
        create_tiled_image,
        TiledImage,
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
except ImportError as e:
    # This will be caught by the import_utils function in app.py
    import logging

    logging.warning(f"Error importing utils modules: {e}")
