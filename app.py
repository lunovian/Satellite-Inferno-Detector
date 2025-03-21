import os
import io
import base64
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import tempfile
import pandas as pd
import time
import glob
import sys
import traceback
import uuid
import logging as logger
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

# After existing imports, add:
from utils.mpc_utils import (
    init_planetary_computer,
    search_satellite_imagery,
    process_satellite_image,
    get_available_collections,
    create_preview_image,  # Add this import
)
from datetime import datetime, timedelta

# Add to imports
from utils.csv_utils import (
    validate_wildfire_data,
    parse_date_column,
    get_common_date_formats,
    validate_satellite_dates,
    detect_numeric_columns,  # Add this import
    detect_columns,
    validate_column_selection,
)

# Add import for image tiling utilities
from utils.image_utils import create_tiled_image, TiledImage, should_tile_image

# Set page config - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Satellite Inferno Detector",
    page_icon="🔥",
    layout="wide",
)

# Fix for torch._classes issue with Streamlit
# Add this before importing torch or YOLO
os.environ["PYTHONPATH"] = os.getcwd()
import asyncio

# Set correct event loop policy based on platform
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Fix for torch.classes.__path__ error in Streamlit
try:
    import torch

    # This resolves "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!"
    torch.classes.__path__ = []
except ImportError:
    st.warning("Could not import torch. Some functionality may be limited.")

# Now we can safely import YOLO and the YOLOEnsemble
try:
    from ultralytics import YOLO
    from model import YOLOEnsemble
except Exception as e:
    st.error(f"Error importing YOLO: {e}")
    st.info(
        "If you're seeing an error related to torch._classes, you might need to restart the application."
    )
    traceback.print_exc()


# Load custom CSS
def load_css():
    with open("app.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Try to load CSS, but continue if file not found
try:
    load_css()
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Application directories
UPLOAD_FOLDER = "uploads"
MODELS_DIR = "models"

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state for persistent app state
if "ensemble" not in st.session_state:
    st.session_state.ensemble = None
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "tiled_images" not in st.session_state:
    st.session_state.tiled_images = {}


# Helper to get available models
def get_available_models():
    if os.path.exists(MODELS_DIR):
        return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    return []


# Helper to detect YOLO version from filename
def detect_yolo_version(filename):
    filename = filename.lower()
    if "v12" in filename or "yolov12" in filename:
        return "v12"
    elif "v11" in filename or "yolov11" in filename:
        return "v11"
    elif "v10" in filename or "yolov10" in filename:
        return "v10"
    elif "v9" in filename or "yolov9" in filename:
        return "v9"
    elif "v8" in filename or "yolov8" in filename:
        return "v8"
    elif "v5" in filename or "yolov5" in filename:
        return "v5"
    else:
        return "other"


# Helper to detect model size from filename
def detect_model_size(filename):
    filename = filename.lower()
    if "nano" in filename or "tiny" in filename or "-n" in filename:
        return "Small"
    elif "large" in filename or "-l" in filename or "-x" in filename:
        return "Large"
    else:
        return "Medium"


# Process a single image with the YOLO ensemble
def process_image(image_path, ensemble, models_to_use):
    try:
        # Get image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image file"}

        # Get predictions
        predictions = ensemble.predict(image_path, visualize=False)

        # Draw predictions on the image
        img_result = ensemble._draw_predictions(img, predictions)

        # Convert the result image to RGB for Streamlit
        img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # Prepare prediction details
        pred_details = []
        for pred in predictions:
            model_idx = pred["model_idx"]
            if model_idx < len(models_to_use):
                model_name = os.path.basename(models_to_use[model_idx])
            else:
                model_name = f"Model {model_idx + 1}"

            pred_details.append(
                {
                    "class_id": int(pred["cls_id"]),
                    "confidence": float(pred["conf"]),
                    "model": int(model_idx + 1),
                    "model_name": model_name,
                    "box": pred["box"].tolist(),
                }
            )

        return {
            "image": img_result_rgb,
            "predictions": pred_details,
            "count": len(predictions),
        }
    except AttributeError as e:
        if "'Conv' object has no attribute 'bn'" in str(e):
            return {
                "error": "Model compatibility error: 'Conv' object has no attribute 'bn'. This usually happens with YOLOv10+ models running on older Ultralytics versions. Please update to Ultralytics 8.1.0 or later."
            }
        else:
            return {"error": f"AttributeError: {str(e)}"}
    except Exception as e:
        st.error(f"Error in process_image: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def render_mpc_section():
    """Render the Microsoft Planetary Computer section"""
    st.header("Search Satellite Imagery")

    # Initialize MPC
    catalog = init_planetary_computer()
    if not catalog:
        st.error("Failed to connect to Microsoft Planetary Computer")
        return

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitude", value=0.0, min_value=-90.0, max_value=90.0)
        # Add text input for faster date entry
        date_text = st.text_input(
            "Date (YYYY-MM-DD)",
            value=datetime.now().strftime("%Y-%m-%d"),
            help="Enter date in YYYY-MM-DD format",
        )
        try:
            # Parse the entered date
            selected_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError:
            st.error("Invalid date format. Please use YYYY-MM-DD")
            selected_date = datetime.now().date()

        # Date picker as backup/visual selector
        date_start = st.date_input(
            "Or select from calendar",
            value=selected_date,
            help="Selected date will be used for both start and end date",
        )

    with col2:
        lon = st.number_input("Longitude", value=0.0, min_value=-180.0, max_value=180.0)
        # Show the same date (read-only)
        st.text_input(
            "Search Date",
            value=date_start.strftime("%Y-%m-%d"),
            disabled=True,
            help="Using same date for start and end",
        )

    # Collection selection
    collections = get_available_collections()
    selected_collection = st.selectbox(
        "Satellite Collection",
        options=list(collections.keys()),
        format_func=lambda x: collections[x],
    )

    # Cloud cover slider
    max_cloud_cover = st.slider(
        "Maximum Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=20,
    )

    # Search button
    if st.button("Search Satellite Imagery"):
        with st.spinner("Searching for imagery..."):
            items = search_satellite_imagery(
                catalog,
                lat,
                lon,
                date_start.isoformat(),  # Use same date for start
                date_start.isoformat(),  # Use same date for end
                collection=selected_collection,
                max_cloud_cover=max_cloud_cover,
            )

            if not items:
                st.warning("No images found matching your criteria")
                return

            # Display results
            st.success(f"Found {len(items)} images")

            # Create image selection with previews
            for idx, item in enumerate(items):
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        # Display metadata
                        st.write(f"Image {idx + 1}")
                        st.write(f"Date: {item.datetime.strftime('%Y-%m-%d')}")
                        st.write(
                            f"Cloud Cover: {item.properties.get('eo:cloud_cover', 'N/A')}%"
                        )

                    with col2:
                        # Display preview image
                        preview = create_preview_image(item)
                        if preview:
                            st.image(
                                preview, caption="Preview", use_container_width=True
                            )
                        else:
                            st.warning("Preview not available")

                    with col3:
                        if st.button("Process", key=f"process_{idx}"):
                            with st.spinner("Processing image..."):
                                image_path = process_satellite_image(item)
                                if image_path:
                                    if "uploaded_images" not in st.session_state:
                                        st.session_state.uploaded_images = []

                                    st.session_state.uploaded_images.append(
                                        (image_path, f"satellite_image_{idx}.jpg")
                                    )
                                    st.success("Image ready for detection!")
                                    st.rerun()

                    st.divider()  # Add separator between images


def render_csv_import_section():
    """Render the enhanced CSV import section"""
    st.header("Import Wildfire Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing wildfire data",
        type=["csv"],
        help="CSV file should contain coordinates and dates of wildfires",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            total_rows = len(df)
            st.success(f"Successfully loaded CSV with {total_rows:,} records")

            # Show initial data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head(), use_container_width=True)

            # 1. Fire Confidence Filtering (Moved before row selection)
            filtered_df = df.copy()  # Create a copy for filtering
            numeric_cols = detect_numeric_columns(df)
            if numeric_cols:
                st.subheader("Fire Confidence Filtering")
                conf_col = st.selectbox(
                    "Fire Confidence Column (optional)",
                    ["None"] + numeric_cols,
                    help="Select the column containing confidence scores",
                )

                if conf_col != "None":
                    min_conf = float(df[conf_col].min())
                    max_conf = float(df[conf_col].max())
                    min_conf = round(min_conf, 1)
                    max_conf = round(max_conf, 1)

                    conf_range = st.slider(
                        "Confidence Range",
                        min_value=min_conf,
                        max_value=max_conf,
                        value=(min_conf, max_conf),
                        step=0.1,
                        format="%.1f",
                        help="Filter records by confidence value",
                    )

                    filtered_df = filtered_df[
                        filtered_df[conf_col].between(*conf_range)
                    ]
                    st.info(
                        f"Found {len(filtered_df):,} records within confidence range"
                    )

                    # Show filtered data preview
                    with st.expander("Preview Filtered Data"):
                        st.dataframe(filtered_df.head(), use_container_width=True)

            # 2. Row Selection (Now applies to filtered data)
            st.subheader("Data Selection")
            available_rows = len(filtered_df)

            # Helper functions for managing state
            def get_safe_default_limit(total_rows: int) -> int:
                """Get safe default limit that never exceeds available rows"""
                return min(100, total_rows)

            def get_valid_limit_value(value: int, max_value: int) -> int:
                """Ensure limit value is within valid range"""
                return max(1, min(value, max_value))

            # Sync session state with current dataset size
            if "current_dataset_size" not in st.session_state:
                st.session_state.current_dataset_size = available_rows

            # Reset limits if dataset size changes
            if st.session_state.current_dataset_size != available_rows:
                st.session_state.current_dataset_size = available_rows
                st.session_state.custom_limit = get_safe_default_limit(available_rows)
                st.session_state.dropdown_limit = str(
                    get_safe_default_limit(available_rows)
                )

            # Initialize session state with safe values
            if "custom_limit" not in st.session_state:
                st.session_state.custom_limit = get_safe_default_limit(available_rows)
            if "dropdown_limit" not in st.session_state:
                st.session_state.dropdown_limit = str(
                    get_safe_default_limit(available_rows)
                )

            # Get dynamic limit options based on dataset size
            def get_limit_options(total_rows: int) -> List[int]:
                if total_rows <= 100:
                    return [n for n in [5, 10, 25, 50, 100] if n <= total_rows]
                elif total_rows <= 1000:
                    return [n for n in [10, 50, 100, 250, 500, 1000] if n <= total_rows]
                else:
                    base_options = [10, 100, 500]
                    scaled_options = [
                        n for n in [1000, 5000, 10000, 50000] if n < total_rows
                    ]
                    return base_options + scaled_options

            # Create a container for the selection controls
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])

                # Event handlers with validation
                def on_dropdown_change():
                    value = st.session_state.dropdown_limit
                    if value != "All":
                        new_limit = get_valid_limit_value(int(value), available_rows)
                        st.session_state.custom_limit = new_limit

                def on_number_change():
                    new_limit = get_valid_limit_value(
                        st.session_state.custom_limit, available_rows
                    )
                    st.session_state.custom_limit = new_limit
                    st.session_state.dropdown_limit = str(new_limit)

                with col1:
                    # Ensure dropdown options are valid for current dataset
                    limit_options = ["All"] + [
                        str(x) for x in get_limit_options(available_rows)
                    ]
                    # Validate current dropdown value
                    if st.session_state.dropdown_limit not in limit_options:
                        st.session_state.dropdown_limit = str(
                            get_safe_default_limit(available_rows)
                        )

                    row_limit_dropdown = st.selectbox(
                        "Quick selection",
                        options=limit_options,
                        key="dropdown_limit",
                        on_change=on_dropdown_change,
                        help="Choose from preset numbers or use custom input",
                    )

                with col2:
                    # Ensure custom limit is within valid range
                    current_limit = get_valid_limit_value(
                        st.session_state.custom_limit, available_rows
                    )
                    row_limit_custom = st.number_input(
                        "Custom number of records",
                        min_value=1,
                        max_value=available_rows,
                        value=current_limit,
                        step=max(
                            1, min(available_rows // 100, 10)
                        ),  # Reasonable step size
                        key="custom_limit",
                        on_change=on_number_change,
                        help=f"Enter a specific number (1 to {available_rows:,})",
                    )

                with col3:
                    st.markdown("##### &nbsp;")  # Spacing for alignment
                    if st.button("Reset", help="Reset to default values"):
                        default_limit = get_safe_default_limit(available_rows)
                        st.session_state.custom_limit = default_limit
                        st.session_state.dropdown_limit = str(default_limit)
                        st.rerun()

            # Apply the final limit with validation
            if row_limit_dropdown == "All":
                row_limit = available_rows
                working_df = filtered_df
            else:
                row_limit = get_valid_limit_value(row_limit_custom, available_rows)
                working_df = filtered_df.head(row_limit)

            # Show selection summary with percentage
            if row_limit < available_rows:
                percentage = (row_limit / available_rows) * 100
                st.info(
                    f"Processing {row_limit:,} records "
                    f"({percentage:.1f}% of {available_rows:,} available records)"
                )
            else:
                st.info(f"Processing all {available_rows:,} records")

            # Show preview with loading indicator
            with st.expander("Preview Selected Data"):
                with st.spinner("Loading preview..."):
                    st.dataframe(
                        working_df.head(),
                        use_container_width=True,
                    )

            # Column mapping and remaining code uses working_df
            st.subheader("Map CSV Columns")
            suggested_columns = detect_columns(working_df)

            # Show auto-detection status
            if any(suggested_columns.values()):
                st.success("Automatically detected some columns!")

            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox(
                    "Latitude Column",
                    working_df.columns,
                    index=working_df.columns.get_loc(suggested_columns["latitude"])
                    if suggested_columns["latitude"]
                    else 0,
                    help="Column containing latitude values (-90 to 90)",
                )
                lon_col = st.selectbox(
                    "Longitude Column",
                    working_df.columns,
                    index=working_df.columns.get_loc(suggested_columns["longitude"])
                    if suggested_columns["longitude"]
                    else 0,
                    help="Column containing longitude values (-180 to 180)",
                )
                date_col = st.selectbox(
                    "Date Column",
                    working_df.columns,
                    index=working_df.columns.get_loc(suggested_columns["date"])
                    if suggested_columns["date"]
                    else 0,
                    help="Column containing dates",
                )

            with col2:
                time_col = st.selectbox(
                    "Time Column (optional)",
                    ["None"] + list(working_df.columns),
                    index=working_df.columns.get_loc(suggested_columns["time"]) + 1
                    if suggested_columns["time"]
                    else 0,
                    help="Optional column containing time information",
                )

                # Date format selection with preview
                date_formats = get_common_date_formats()
                format_key = st.selectbox(
                    "Date Format",
                    list(date_formats.keys()),
                    format_func=lambda x: f"{date_formats[x]} ({x})",
                )

            # Validate column selection
            valid, message = validate_column_selection(
                working_df,
                {
                    "latitude": lat_col,
                    "longitude": lon_col,
                    "date": date_col,
                    "time": None if time_col == "None" else time_col,
                },
            )

            if not valid:
                st.warning(f"Column validation warning: {message}")

            # Date validation preview
            try:
                # Add more informative error handling for date parsing
                preview_dates = pd.Series()
                date_parse_error = None

                try:
                    preview_dates = parse_date_column(
                        working_df.head(),
                        date_col,
                        None if time_col == "None" else time_col,
                        format_key if format_key != "auto" else None,
                    )
                except ValueError as e:
                    date_parse_error = str(e)

                if date_parse_error:
                    st.error(f"Date parsing error: {date_parse_error}")
                    st.info(
                        "Try selecting a different date format or check the date column values"
                    )
                    # Show some sample values from the date column
                    st.write("Sample values from date column:")
                    st.write(working_df[date_col].head().tolist())
                else:
                    st.success("Date format validation successful")
                    with st.expander("Date Parsing Preview"):
                        st.write("Sample of parsed dates:")
                        for orig, parsed in zip(
                            working_df[date_col].head(), preview_dates
                        ):
                            st.write(f"{orig} → {parsed.strftime('%Y-%m-%d')}")
            except Exception as e:
                st.error(f"Error previewing dates: {str(e)}")
                st.info("Please check the selected date column and format")

            column_mapping = {
                "latitude": lat_col,
                "longitude": lon_col,
                "date": date_col,
                "time": None if time_col == "None" else time_col,
            }

            # Process button with cancel option
            process_col, cancel_col = st.columns([3, 1])
            with process_col:
                process = st.button(
                    "Process Selected Fires", disabled=not working_df.shape[0] > 0
                )

            if process:
                process_placeholder = st.empty()
                progress_bar = st.progress(0)

                # Create a cancellation button
                if cancel_col.button("Cancel Processing"):
                    st.session_state.cancel_processing = True
                    st.warning("Cancelling process...")
                    return

                st.session_state.cancel_processing = False

                with st.spinner("Validating data..."):
                    # Update unpacking to handle all three return values
                    valid, message, validation_details = validate_wildfire_data(
                        working_df, column_mapping
                    )
                    if not valid:
                        st.error(message)
                        # Display validation details in expander
                        with st.expander("Validation Details"):
                            st.write(
                                f"Total records: {validation_details['total_rows']}"
                            )
                            st.write(
                                f"Valid records: {validation_details['valid_rows']}"
                            )
                            if validation_details["invalid_coordinates"] > 0:
                                st.warning(
                                    f"Invalid coordinates: {validation_details['invalid_coordinates']}"
                                )
                            if validation_details["invalid_dates"] > 0:
                                st.warning(
                                    f"Invalid dates: {validation_details['invalid_dates']}"
                                )
                        return

                    # Initialize MPC
                    catalog = init_planetary_computer()
                    if not catalog:
                        st.error("Failed to connect to Planetary Computer")
                        return

                    # Process each fire location
                    total = len(working_df)
                    processed = 0
                    found_images = 0
                    fire_results = []

                    for idx, row in working_df.iterrows():
                        if st.session_state.cancel_processing:
                            st.warning("Processing cancelled")
                            break

                        try:
                            # Update progress display
                            progress_text = f"Processing fire {processed + 1}/{total}"
                            process_placeholder.text(progress_text)
                            progress_bar.progress(processed / total)

                            # Get coordinates and date
                            lat = float(row[lat_col])
                            lon = float(row[lon_col])
                            date = parse_date_column(
                                pd.DataFrame([row]),
                                date_col,
                                column_mapping["time"],
                                format_key,
                            ).iloc[0]

                            # Validate date range for satellite
                            valid_date, msg, _ = validate_satellite_dates(
                                pd.Series([date]), "sentinel-2-l2a"
                            )

                            if not valid_date:
                                fire_results.append(
                                    {
                                        "status": "skipped",
                                        "location": f"({lat:.4f}, {lon:.4f})",
                                        "date": date.strftime("%d-%m-%Y"),
                                        "reason": msg,
                                    }
                                )
                                continue

                            # Search for imagery
                            items = search_satellite_imagery(
                                catalog=catalog,
                                lat=lat,
                                lon=lon,
                                date_start=(date - pd.Timedelta(days=5)).isoformat(),
                                date_end=(date + pd.Timedelta(days=5)).isoformat(),
                                collection="sentinel-2-l2a",
                                max_cloud_cover=30,
                            )

                            if items:
                                found_images += 1
                                try:
                                    # Initialize YOLO ensemble if needed
                                    if not st.session_state.ensemble:
                                        with st.spinner(
                                            "Initializing fire detection model..."
                                        ):
                                            ensemble = ensure_yolo_ensemble()
                                            if not ensemble:
                                                st.error(
                                                    "Failed to initialize fire detection model"
                                                )
                                                continue

                                    # Create progress displays for both steps
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        download_progress = st.empty()
                                        download_bar = st.progress(0)
                                    with col2:
                                        detection_progress = st.empty()
                                        detection_bar = st.progress(0)

                                    # Process satellite image with download progress
                                    try:
                                        image_path = process_satellite_image(
                                            items[0],
                                            progress_callback=lambda p: (
                                                download_progress.text(
                                                    f"Downloading: {p:.1f}%"
                                                ),
                                                download_bar.progress(p / 100),
                                            ),
                                        )

                                        if not image_path:
                                            raise ValueError(
                                                "Failed to process satellite image"
                                            )

                                        # Run fire detection
                                        detection_progress.text(
                                            "Running fire detection..."
                                        )
                                        detection_result = process_image(
                                            image_path,
                                            st.session_state.ensemble,
                                            [
                                                os.path.join(MODELS_DIR, model)
                                                for model in st.session_state.selected_models
                                            ],
                                        )

                                        if "error" in detection_result:
                                            raise ValueError(
                                                f"Detection error: {detection_result['error']}"
                                            )

                                        # Store both original and processed images
                                        if "uploaded_images" not in st.session_state:
                                            st.session_state.uploaded_images = []

                                        # Save original
                                        original_path = image_path
                                        processed_path = image_path.replace(
                                            ".jpg", "_detected.jpg"
                                        )

                                        # Save detection result image
                                        cv2.imwrite(
                                            processed_path,
                                            cv2.cvtColor(
                                                detection_result["image"],
                                                cv2.COLOR_RGB2BGR,
                                            ),
                                        )

                                        st.session_state.uploaded_images.extend(
                                            [
                                                (
                                                    original_path,
                                                    f"fire_{idx}_original.jpg",
                                                ),
                                                (
                                                    processed_path,
                                                    f"fire_{idx}_detected.jpg",
                                                ),
                                            ]
                                        )

                                        # Update fire results with detection info
                                        detection_info = {
                                            "status": "success",
                                            "location": f"({lat:.4f}, {lon:.4f})",
                                            "date": date.strftime("%d-%m-%Y"),
                                            "fires_detected": detection_result["count"],
                                            "confidence_avg": sum(
                                                p["confidence"]
                                                for p in detection_result["predictions"]
                                            )
                                            / len(detection_result["predictions"])
                                            if detection_result["predictions"]
                                            else 0,
                                            "model_predictions": detection_result[
                                                "predictions"
                                            ],
                                            "original_image": original_path,
                                            "detected_image": processed_path,
                                        }
                                        fire_results.append(detection_info)

                                        # Show mini preview of detection
                                        with st.expander(
                                            f"Detection Result - Location ({lat:.4f}, {lon:.4f})",
                                            expanded=False,
                                        ):
                                            preview_col1, preview_col2 = st.columns(2)
                                            with preview_col1:
                                                st.image(
                                                    original_path,
                                                    caption="Original",
                                                    width=300,
                                                )
                                            with preview_col2:
                                                st.image(
                                                    processed_path,
                                                    caption=f"Detected: {detection_result['count']} fires",
                                                    width=300,
                                                )

                                            if detection_result["predictions"]:
                                                st.write("Detection Details:")
                                                for pred in detection_result[
                                                    "predictions"
                                                ]:
                                                    st.write(
                                                        f"- Fire detected with {pred['confidence']:.1%} confidence"
                                                    )

                                    except TimeoutError:
                                        st.error(
                                            "Processing timeout. Moving to next location..."
                                        )
                                        fire_results.append(
                                            {
                                                "status": "error",
                                                "location": f"({lat:.4f}, {lon:.4f})",
                                                "error": "Processing timeout",
                                            }
                                        )
                                    finally:
                                        # Clean up progress displays
                                        download_progress.empty()
                                        download_bar.empty()
                                        detection_progress.empty()
                                        detection_bar.empty()

                                except Exception as e:
                                    logger.error(f"Processing error: {e}")
                                    st.error(
                                        f"Error processing location ({lat:.4f}, {lon:.4f}): {str(e)}"
                                    )
                                    traceback.print_exc()
                                    fire_results.append(
                                        {
                                            "status": "error",
                                            "location": f"({lat:.4f}, {lon:.4f})",
                                            "error": str(e),
                                        }
                                    )

                            processed += 1
                            overall_progress = processed / total
                            progress_bar.progress(overall_progress)
                            process_placeholder.text(
                                f"Overall Progress: {processed}/{total} locations ({overall_progress:.1%})"
                            )

                        except Exception as e:
                            st.error(f"Error processing location: {str(e)}")
                            traceback.print_exc()
                            fire_results.append(
                                {
                                    "status": "error",
                                    "location": f"Unknown location",
                                    "error": str(e),
                                }
                            )
                            processed += 1

                    # Show results summary after the loop completes
                    process_placeholder.empty()

                    # Calculate success rate only if records were processed
                    if processed > 0:
                        success_rate = (found_images / processed) * 100
                        success_msg = (
                            f"Processing complete!\n"
                            f"- Processed: {processed}/{total} records\n"
                            f"- Images found: {found_images}\n"
                            f"- Success rate: {success_rate:.1f}%"
                        )
                    else:
                        success_msg = (
                            "Processing complete - No records were processed.\n"
                            "This may be due to cancellation or filtering criteria."
                        )

                    st.success(success_msg)

                    # Display results table
                    results_df = pd.DataFrame(fire_results)
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        column_config={
                            "status": st.column_config.TextColumn(
                                "Status", help="Processing status for each record"
                            )
                        },
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            traceback.print_exc()


# Add this helper function after existing helper functions
def ensure_yolo_ensemble(models_dir: str = "models") -> Optional[YOLOEnsemble]:
    """Initialize or get existing YOLO ensemble"""
    try:
        if "ensemble" not in st.session_state or st.session_state.ensemble is None:
            ensemble = YOLOEnsemble(
                models_dir=None,
                conf_thres=0.3,  # Default confidence threshold
                iou_thres=0.5,  # Default IoU threshold
            )

            # Load all available models
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
            if not model_files:
                st.error("No YOLO models found in models directory")
                return None

            for model_path in model_files:
                try:
                    model = YOLO(os.path.join(models_dir, model_path))
                    ensemble.models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load model {model_path}: {e}")

            if not ensemble.models:
                st.error("Failed to load any YOLO models")
                return None

            st.session_state.ensemble = ensemble

        return st.session_state.ensemble

    except Exception as e:
        logger.error(f"Error initializing YOLO ensemble: {e}")
        st.error(f"Failed to initialize fire detection: {str(e)}")
        return None


# Add this function to handle file uploads with tiling support
def process_uploaded_files(uploaded_files):
    """Process uploaded files with automatic tiling for large images"""
    st.session_state.uploaded_images = []
    st.session_state.tiled_images = {}

    # Settings for tiling
    MAX_FILE_SIZE_MB = 150  # Maximum file size before tiling
    TILE_OVERLAP = 128  # Overlap between tiles in pixels

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                file_path = temp_file.name

            # Check if we need to tile this image
            if should_tile_image(file_path, MAX_FILE_SIZE_MB):
                st.info(f"Splitting large image '{uploaded_file.name}' into tiles...")

                # Create progress bar for tiling
                tiling_progress = st.progress(0)

                # Create tiled image
                tiled_image = create_tiled_image(
                    file_path=file_path,
                    file_name=uploaded_file.name,
                    overlap=TILE_OVERLAP,
                    max_size_mb=MAX_FILE_SIZE_MB,
                )

                if tiled_image and tiled_image.tiles:
                    # Add original file to uploaded images to be shown in preview
                    st.session_state.uploaded_images.append(
                        (file_path, uploaded_file.name)
                    )

                    # Store tiled image for later processing
                    st.session_state.tiled_images[file_path] = tiled_image

                    # Update progress
                    tiling_progress.progress(1.0)
                    st.success(
                        f"Split '{uploaded_file.name}' into {len(tiled_image.tiles)} tiles"
                    )
                else:
                    st.error(f"Failed to split '{uploaded_file.name}'")
                    # Add as regular file
                    st.session_state.uploaded_images.append(
                        (file_path, uploaded_file.name)
                    )
            else:
                # Normal file, no tiling needed
                st.session_state.uploaded_images.append((file_path, uploaded_file.name))


# Main application function
def main():
    # Application header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(
            "https://img.icons8.com/color/96/000000/fire-element--v2.png", width=80
        )
    with col2:
        st.title("Satellite Inferno Detector")
        st.markdown("### YOLO Ensemble for Wildfire Detection")

    # Sidebar - Model Selection Section
    st.sidebar.title("Model Selection")

    # Get available models
    # Add error handling for model loading
    try:
        available_models = get_available_models()
        if not available_models:
            st.sidebar.error(
                "No YOLO models found! Place .pt files in the 'models' directory."
            )
            st.error("Please add YOLO model files (.pt) to the 'models' directory.")
            st.info("Example model files: yolov8n.pt, yolov8s.pt, etc.")
            return
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.error("Error accessing the models directory. Please check permissions.")
        return

    if not available_models:
        st.sidebar.error(
            "No YOLO models found! Place .pt files in the 'models' directory."
        )
        st.stop()

    # Model information
    st.sidebar.success(f"{len(available_models)} Models Available")

    # Create model selection with metadata
    model_options = []
    model_metadata = []

    for model in available_models:
        version = detect_yolo_version(model)
        size = detect_model_size(model)

        # Format version display
        versionLabel = (
            f"YOLO{version.upper()}"
            if version in ["v5", "v8", "v9", "v10", "v11", "v12"]
            else version.upper()
        )

        model_options.append(model)
        model_metadata.append({"name": model, "version": versionLabel, "size": size})

    # Display model selection with search and filter capabilities
    st.sidebar.subheader("Select Models")
    # Search filter
    search_term = st.sidebar.text_input("🔍 Search models", "")

    # Filter models by search term
    filtered_models = []
    for i, meta in enumerate(model_metadata):
        if search_term.lower() in meta["name"].lower():
            filtered_models.append((i, meta))

    # Select/Deselect All buttons
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All"):
        st.session_state.selected_models = model_options.copy()
    if col2.button("Deselect All"):
        st.session_state.selected_models = []

    # Initialize selected models in session state if not already
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = model_options.copy()

    # Display filtered models with checkboxes
    selected_models = []

    for i, meta in filtered_models:
        model_name = meta["name"]

        # Create a unique key for each model
        checkbox_key = f"model_{i}"

        # Determine if model should be selected
        is_selected = model_name in st.session_state.selected_models

        # Display the model with metadata
        col1, col2, col3 = st.sidebar.columns([5, 2, 2])
        # Checkbox in first column
        if col1.checkbox(
            model_name,
            value=is_selected,
            key=checkbox_key,
            help=f"Select this model for the ensemble",
        ):
            selected_models.append(model_name)
            if model_name not in st.session_state.selected_models:
                st.session_state.selected_models.append(model_name)
        else:
            if model_name in st.session_state.selected_models:
                st.session_state.selected_models.remove(model_name)

        # Show version tag in second column
        version_colors = {
            "YOLOV5": "#198754",
            "YOLOV8": "#0d6efd",
            "YOLOV9": "#fd7e14",
            "YOLOV10": "#6f42c1",
            "YOLOV11": "#d63384",
            "YOLOV12": "#20c997",
        }
        version_color = version_colors.get(meta["version"], "#6c757d")
        col2.markdown(
            f"<span style='background-color:{version_color};color:white;padding:2px 6px;border-radius:10px;font-size:0.75rem;'>{meta['version']}</span>",
            unsafe_allow_html=True,
        )

        # Show size tag in third column
        size_colors = {"Small": "#20c997", "Medium": "#fd7e14", "Large": "#dc3545"}
        size_color = size_colors.get(meta["size"], "#6c757d")
        col3.markdown(
            f"<span style='background-color:{size_color};color:white;padding:2px 6px;border-radius:10px;font-size:0.75rem;'>{meta['size']}</span>",
            unsafe_allow_html=True,
        )

    if not selected_models:
        if search_term:
            st.sidebar.warning(f"No models matching '{search_term}'")
        else:
            st.sidebar.warning("No models selected. Please select at least one model.")
    else:
        st.sidebar.success(f"{len(selected_models)} models selected")

    # Detection Parameters
    st.sidebar.subheader("Detection Parameters")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        format="%.2f",
        help="Minimum confidence score for a detection to be considered",
    )

    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        format="%.2f",
        help="Intersection over Union threshold for non-maximum suppression",
    )

    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(
        [
            "Upload Images",
            "Search Satellite Imagery",
            "Import Wildfire Data",
        ]
    )

    with tab1:
        st.header("Upload Images")
        st.header("Upload Images")
        # Image upload
        uploaded_files = st.file_uploader(
            "Drag and drop image files here",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            help="Upload satellite or drone images for wildfire detection",
        )

        # Process uploaded images with tiling support
        if uploaded_files:
            process_uploaded_files(uploaded_files)

            # Display image previews (unchanged)
            st.subheader("Uploaded Images")
            cols = st.columns(4)
            for i, (file_path, file_name) in enumerate(
                st.session_state.uploaded_images
            ):
                with cols[i % 4]:
                    st.image(file_path, caption=file_name, width=150)

        # Detection button
        if st.button(
            "Detect Wildfires",
            disabled=not (
                st.session_state.selected_models and st.session_state.uploaded_images
            ),
        ):
            if not st.session_state.selected_models:
                st.error("Please select at least one model")
            elif not st.session_state.uploaded_images:
                st.error("Please upload at least one image")
            else:
                # Create model paths from session state
                selected_models = st.session_state.selected_models
                models_to_use = [
                    os.path.join(MODELS_DIR, model) for model in selected_models
                ]

                # ...rest of existing detection code...

    with tab2:
        render_mpc_section()

    with tab3:
        render_csv_import_section()

    # Display results if available
    if st.session_state.processed_images:
        st.header("Detection Results")

        # Summary info
        if st.session_state.results:
            models_used = st.session_state.results.get("models_used", [])
            st.info(
                f"Processed {len(st.session_state.processed_images)} images using {len(models_used)} models"
            )

        display_mode = st.radio(
            "Display Mode",
            ["Grid View", "List View"],
            help="Choose how to display the detection results",
            horizontal=True,
        )

        if display_mode == "Grid View":
            # Grid View - Show images in a grid with details in expandable sections
            num_cols = 3  # Number of columns in the grid
            # Create a grid of images
            rows = [
                st.columns(num_cols)
                for _ in range(
                    (len(st.session_state.processed_images) + num_cols - 1) // num_cols
                )
            ]
            for i, result in enumerate(st.session_state.processed_images):
                row_idx = i // num_cols
                col_idx = i % num_cols
                with rows[row_idx][col_idx]:
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.image(
                            result["image"],
                            caption=f"{result.get('filename', f'Image {i + 1}')} - {result['count']} detections",
                            use_container_width=True,
                        )
                        if result["predictions"]:
                            with st.expander("Detection Details"):
                                df = pd.DataFrame(result["predictions"])
                                df = df.rename(
                                    columns={
                                        "class_id": "Class ID",
                                        "confidence": "Confidence",
                                        "model_name": "Model",
                                        "box": "Bounding Box",
                                    }
                                )
                                # Format confidence as percentage
                                df["Confidence"] = df["Confidence"].map(
                                    lambda x: f"{x:.2%}"
                                )
                                # Format bounding box coordinates
                                df["Bounding Box"] = df["Bounding Box"].map(
                                    lambda box: f"[{', '.join([f'{coord:.1f}' for coord in box])}]"
                                )
                                if "model" in df.columns:
                                    # Drop model column (redundant)
                                    df = df.drop(columns=["model"])
                                st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No objects detected in this image")
        else:  # List View
            # List View - Show images in a vertical list with details alongside
            for i, result in enumerate(st.session_state.processed_images):
                st.subheader(result.get("filename", f"Image {i + 1}"))
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2 = st.columns([3, 2])
                    # Create two columns - one for image, one for details
                    with col1:
                        # Show image with detections
                        st.image(
                            result["image"],
                            caption=f"Detections: {result['count']}",
                            use_container_width=True,
                        )
                    with col2:
                        if result["predictions"]:
                            st.markdown("### Detection Details")
                            df = pd.DataFrame(result["predictions"])
                            df = df.rename(
                                columns={
                                    "class_id": "Class ID",
                                    "confidence": "Confidence",
                                    "model_name": "Model",
                                    "box": "Bounding Box",
                                }
                            )
                            # Format confidence as percentage
                            df["Confidence"] = df["Confidence"].map(
                                lambda x: f"{x:.2%}"
                            )
                            # Format bounding box coordinates
                            df["Bounding Box"] = df["Bounding Box"].map(
                                lambda box: f"[{', '.join([f'{coord:.1f}' for coord in box])}]"
                            )
                            if "model" in df.columns:
                                # Drop model column (redundant)
                                df = df.drop(columns=["model"])
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No objects detected in this image")
                st.divider()  # Add a divider between images in list view


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        traceback.print_exc()
