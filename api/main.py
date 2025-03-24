from datetime import datetime, timedelta
from typing import List
from pathlib import Path

from ..utils.mpc_utils import (  # Update the import path
    init_planetary_computer,
    search_satellite_imagery,
    process_satellite_image,
    ImageProcessingConfig,
    ImageQuality,
)


def search_images(
    lat: float,
    lon: float,
    date: datetime,
    expand_aoi_km: float = 5.0,
    max_cloud_cover: int = 20,
    collection: str = "sentinel-2-l2a",
) -> List[dict]:
    """Search for satellite imagery at a specific location and date"""
    catalog = init_planetary_computer()
    if not catalog:
        raise RuntimeError("Planetary Computer unavailable")

    items = search_satellite_imagery(
        catalog=catalog,
        lat=lat,
        lon=lon,
        date_start=date.isoformat(),
        date_end=(date + timedelta(days=1)).isoformat(),
        collection=collection,
        max_cloud_cover=max_cloud_cover,
        expand_aoi_km=expand_aoi_km,
    )

    return [
        {
            "id": item.id,
            "datetime": item.datetime.isoformat(),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "preview_url": item.assets.get("preview", {}).href,
        }
        for item in items
    ]


def detect_fires(image_id: str, quality: ImageQuality = ImageQuality.MEDIUM):
    """Run fire detection on a specific satellite image"""
    catalog = init_planetary_computer()
    if not catalog:
        raise RuntimeError("Planetary Computer unavailable")

    search = catalog.search(ids=[image_id])
    items = list(search.items())
    if not items:
        raise RuntimeError("Image not found")

    item = items[0]
    config = ImageProcessingConfig.from_quality_preset(quality)
    return process_satellite_image(item, config)
