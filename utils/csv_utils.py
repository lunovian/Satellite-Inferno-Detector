from datetime import datetime
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from dateutil.parser import parse as dateutil_parse
import re

logger = logging.getLogger(__name__)

# Satellite operational date ranges
SATELLITE_DATE_RANGES = {
    "sentinel-2-l2a": {
        "start": "2015-06-23",  # Sentinel-2A launch
        "end": None,  # Still active
        "description": "Sentinel-2 Level 2A data",
    },
    "landsat-8-c2-l2": {
        "start": "2013-02-11",  # Landsat 8 launch
        "end": None,
    },
    "landsat-9-c2-l2": {
        "start": "2021-09-27",  # Landsat 9 launch
        "end": None,
    },
}

# Common column name patterns for auto-detection
COLUMN_PATTERNS = {
    "latitude": {
        "names": {"lat", "latitude", "y", "northing", "ylat"},
        "validation": lambda x: x.between(-90, 90).all(),
        "required": True,
    },
    "longitude": {
        "names": {"lon", "long", "longitude", "x", "easting", "xlong"},
        "validation": lambda x: x.between(-180, 180).all(),
        "required": True,
    },
    "date": {
        "names": {
            "date",
            "datetime",
            "acquisition",
            "acq_date",
            "timestamp",
            "obs_date",
        },
        "validation": lambda x: pd.to_datetime(x, errors="coerce").notna().all(),
        "required": True,
    },
    "time": {
        "names": {"time", "hour", "acq_time", "obs_time"},
        "validation": None,  # Optional field
        "required": False,
    },
    "confidence": {
        "names": {
            "conf",
            "confidence",
            "probability",
            "certainty",
            "detection_conf",
            "score",
            "frp",
            "brightness",
            "intensity",
            "detect_score",
        },
        "validation": lambda x: x.between(0, 100).all() or x.between(0, 1).all(),
        "required": False,
    },
}


def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str) -> bool:
    """Validate coordinate columns"""
    try:
        lat_valid = df[lat_col].between(-90, 90).all()
        lon_valid = df[lon_col].between(-180, 180).all()
        return lat_valid and lon_valid
    except Exception as e:
        logger.error(f"Coordinate validation error: {e}")
        return False


def detect_date_format(sample_date: str) -> str:
    """Try to detect the date format from a sample date string"""
    common_formats = [
        ("%Y-%m-%d", "YYYY-MM-DD"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
        ("%m/%d/%Y", "MM/DD/YYYY"),
        ("%Y%m%d", "YYYYMMDD"),
        ("%Y.%m.%d", "YYYY.MM.DD"),
        ("%d-%m-%Y", "DD-MM-YYYY"),
        ("%m-%d-%Y", "MM-DD-YYYY"),
        ("%d.%m.%Y", "DD.MM.YYYY"),
    ]

    for fmt, _ in common_formats:
        try:
            datetime.strptime(str(sample_date), fmt)
            return fmt
        except ValueError:
            continue

    return None


def clean_date_string(date_str: str) -> str:
    """
    Clean and normalize date strings by handling various edge cases.

    Args:
        date_str: Input date string that may contain extra information

    Returns:
        Cleaned date string containing only the date portion
    """
    try:
        # Convert to string and strip whitespace
        cleaned = str(date_str).strip()

        # Extract date pattern using regex
        date_patterns = [
            # YYYY-MM-DD or YYYY/MM/DD followed by anything
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            # DD-MM-YYYY or DD/MM/YYYY followed by anything
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
            # YYYYMMDD format
            r"(\d{4}\d{2}\d{2}(?!\d))",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, cleaned)
            if match:
                return match.group(1)

        # If no pattern matched, try splitting on common separators
        for sep in [" ", "T", "_", ";", ","]:
            if sep in cleaned:
                parts = cleaned.split(sep)
                # Try to parse each part to find the date
                for part in parts:
                    try:
                        # Quick validation of potential date string
                        if re.match(r"\d{4}|\d{1,2}", part.strip()):
                            pd.to_datetime(part.strip())
                            return part.strip()
                    except:
                        continue

        return cleaned
    except Exception as e:
        logger.debug(f"Error cleaning date string '{date_str}': {e}")
        return str(date_str)


def normalize_time(time_str: str) -> str:
    """
    Normalize time strings to HH:MM format.

    Args:
        time_str: Input time string (e.g., "457" or "4:57" or "04:57")

    Returns:
        Normalized time string in HH:MM format
    """
    try:
        # Remove any non-numeric characters
        nums = re.sub(r"[^0-9]", "", str(time_str))

        if len(nums) <= 0:
            return "00:00"

        # Handle different formats
        if len(nums) <= 2:  # Single or double digit
            hour = int(nums)
            return f"{hour:02d}:00"
        elif len(nums) == 3:  # Format like "457"
            hour = int(nums[0])
            minute = int(nums[1:])
            return f"{hour:02d}:{minute:02d}"
        elif len(nums) == 4:  # Format like "1234"
            hour = int(nums[:2])
            minute = int(nums[2:])
            return f"{hour:02d}:{minute:02d}"
        else:
            return "00:00"
    except Exception as e:
        logger.debug(f"Error normalizing time string '{time_str}': {e}")
        return "00:00"


def parse_date_column(
    df: pd.DataFrame,
    date_col: str,
    time_col: Optional[str] = None,
    date_format: str = None,
) -> pd.Series:
    """Parse date and optional time columns into datetime"""
    try:
        # Clean and prepare date strings
        dates = df[date_col].astype(str).apply(clean_date_string)

        # Prepare time strings if provided
        if time_col:
            times = df[time_col].astype(str).apply(normalize_time)
            datetime_str = dates + " " + times
        else:
            datetime_str = dates

        # Track parsing success for each row
        success_mask = pd.Series(False, index=df.index)
        parsed_dates = pd.Series(pd.NaT, index=df.index)

        # 1. Try user-specified format first
        if date_format:
            try:
                temp_parsed = pd.to_datetime(datetime_str, format=date_format)
                success_mask = ~temp_parsed.isna()
                parsed_dates[success_mask] = temp_parsed[success_mask]
                if success_mask.all():
                    return parsed_dates
            except:
                pass

        # 2. Try common formats for unparsed dates
        common_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y%m%d",
            "%d-%m-%Y",
            "%Y.%m.%d",
        ]

        remaining_mask = ~success_mask
        for fmt in common_formats:
            if not remaining_mask.any():
                break

            try:
                temp_parsed = pd.to_datetime(datetime_str[remaining_mask], format=fmt)
                newly_parsed = ~temp_parsed.isna()
                parsed_dates[remaining_mask][newly_parsed] = temp_parsed[newly_parsed]
                remaining_mask[remaining_mask] = ~newly_parsed
            except:
                continue

        # 3. Final attempt with flexible parser for remaining dates
        if remaining_mask.any():
            try:
                temp_parsed = pd.to_datetime(
                    datetime_str[remaining_mask],
                    format="mixed",
                    dayfirst=True,  # Assume DD-MM-YYYY for ambiguous dates
                )
                parsed_dates[remaining_mask] = temp_parsed
            except Exception as e:
                logger.warning(f"Flexible parsing failed: {e}")

        # Report parsing failures
        failed_mask = parsed_dates.isna()
        if failed_mask.any():
            failed_examples = datetime_str[failed_mask].head()
            logger.warning(
                f"Failed to parse {failed_mask.sum()} dates. "
                f"Examples: {failed_examples.tolist()}"
            )

        # Standardize format while preserving datetime objects
        # Instead of converting to string and back, format only for display
        formatted_dates = parsed_dates.dt.strftime("%d-%m-%Y")
        logger.info(
            f"Sample parsed dates (DD-MM-YYYY): {formatted_dates.head().tolist()}"
        )

        # Return the original parsed datetime objects
        return parsed_dates

    except Exception as e:
        logger.error(f"Date parsing error: {e}")
        raise ValueError(f"Error parsing dates: {e}")


def validate_wildfire_data(
    df: pd.DataFrame, column_mapping: Dict[str, str]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate wildfire data DataFrame.

    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping required columns to DataFrame columns

    Returns:
        Tuple of (is_valid, error_message, validation_details)
    """
    validation_details = {
        "total_rows": len(df),
        "valid_rows": 0,
        "invalid_coordinates": 0,
        "invalid_dates": 0,
    }

    # Check required columns
    required_cols = [k for k, v in COLUMN_PATTERNS.items() if v["required"]]
    missing_cols = [
        col for col in required_cols if column_mapping.get(col) not in df.columns
    ]
    if missing_cols:
        return (
            False,
            f"Missing required columns: {', '.join(missing_cols)}",
            validation_details,
        )

    # Validate coordinates
    try:
        lat_col, lon_col = column_mapping["latitude"], column_mapping["longitude"]
        valid_coords = df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)
        validation_details["invalid_coordinates"] = (~valid_coords).sum()

        if validation_details["invalid_coordinates"] > 0:
            return (
                False,
                f"Found {validation_details['invalid_coordinates']} invalid coordinates",
                validation_details,
            )

    except Exception as e:
        return False, f"Error validating coordinates: {str(e)}", validation_details

    validation_details["valid_rows"] = len(df) - (
        validation_details["invalid_coordinates"] + validation_details["invalid_dates"]
    )

    return True, "Validation successful", validation_details


def get_common_date_formats() -> Dict[str, str]:
    """Return common date format options"""
    return {
        "%Y-%m-%d": "YYYY-MM-DD",
        "%d/%m/%Y": "DD/MM/YYYY",
        "%m/%d/%Y": "MM/DD/YYYY",
        "%Y%m%d": "YYYYMMDD",
        "%d-%m-%Y": "DD-MM-YYYY",
        "%m-%d-%Y": "MM-DD-YYYY",
    }


def validate_satellite_dates(
    dates: pd.Series, collection: str
) -> Tuple[bool, str, List[str]]:
    """Validate dates against satellite collection constraints"""
    if collection not in SATELLITE_DATE_RANGES:
        return False, f"Unknown satellite collection: {collection}", []

    range_info = SATELLITE_DATE_RANGES[collection]
    start_date = pd.to_datetime(range_info["start"])
    end_date = (
        pd.to_datetime(range_info["end"]) if range_info["end"] else pd.Timestamp.now()
    )

    invalid_dates = []
    for date in dates:
        if not (start_date <= date <= end_date):
            invalid_dates.append(date.strftime("%Y-%m-%d"))

    if invalid_dates:
        msg = f"Dates outside valid range for {collection} ({range_info['start']} to present)"
        return False, msg, invalid_dates
    return True, "", []


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns suitable for confidence values"""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect column mappings based on column names and data patterns.
    Enhanced with better confidence detection.
    """
    suggestions: Dict[str, Optional[str]] = {k: None for k in COLUMN_PATTERNS.keys()}

    # First pass: Check column names
    for col in df.columns:
        col_lower = col.lower()
        for key, pattern in COLUMN_PATTERNS.items():
            if suggestions[key] is None:  # Only if not already assigned
                if any(name in col_lower for name in pattern["names"]):
                    if pattern["validation"] is None:
                        suggestions[key] = col
                    elif key == "confidence":
                        # Handle both percentage (0-100) and decimal (0-1) confidence
                        values = pd.to_numeric(df[col], errors="coerce")
                        if values.between(0, 1).all():
                            suggestions[key] = col
                        elif values.between(0, 100).all():
                            suggestions[key] = col
                    elif pattern["validation"](df[col]):
                        suggestions[key] = col
                    break

    # Second pass: Try to detect confidence by value patterns
    if suggestions["confidence"] is None:
        numeric_cols = df.select_dtypes(
            include=["float64", "float32", "int64", "int32"]
        ).columns
        for col in numeric_cols:
            values = df[col]
            # Check if values are in typical confidence ranges
            if values.between(0, 1).all() or values.between(0, 100).all():
                # Additional validation: should have some variation
                if values.nunique() > 1:
                    suggestions["confidence"] = col
                    logger.info(f"Auto-detected confidence column: {col}")
                    break

    return suggestions


def validate_column_selection(
    df: pd.DataFrame, column_mapping: Dict[str, str]
) -> Tuple[bool, str]:
    """Validate the selected columns"""
    # Check if required columns are selected
    if not all([column_mapping.get(col) for col in ["latitude", "longitude", "date"]]):
        return False, "Latitude, longitude, and date columns are required"

    # Validate coordinates
    try:
        lat_col = column_mapping["latitude"]
        lon_col = column_mapping["longitude"]
        if not pd.api.types.is_numeric_dtype(
            df[lat_col]
        ) or not pd.api.types.is_numeric_dtype(df[lon_col]):
            return False, "Coordinate columns must contain numeric values"

        if not df[lat_col].between(-90, 90).all():
            return False, f"Latitude values in {lat_col} must be between -90 and 90"
        if not df[lon_col].between(-180, 180).all():
            return False, f"Longitude values in {lon_col} must be between -180 and 180"
    except Exception as e:
        return False, f"Error validating coordinates: {str(e)}"

    # Enhanced date validation
    try:
        date_col = column_mapping["date"]
        dates = pd.to_datetime(df[date_col].apply(clean_date_string), errors="coerce")
        invalid_dates = dates.isna()
        if invalid_dates.any():
            invalid_examples = df[date_col][invalid_dates].head().tolist()
            return False, f"Invalid dates found. Examples: {invalid_examples}"
    except Exception as e:
        return False, f"Error validating date column: {str(e)}"

    return True, "Column validation successful"
