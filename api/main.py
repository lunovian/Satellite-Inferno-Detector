from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional
import os
from pathlib import Path

from utils.mpc_utils import (
    init_planetary_computer,
    search_satellite_imagery,
    process_satellite_image,
    ImageProcessingConfig,
    ImageQuality,
)

app = FastAPI(
    title="Satellite Inferno Detector API",
    description="API for detecting wildfires in satellite imagery",
    version="1.0.0",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


# API Models
class LocationSearch(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    date: datetime
    expand_aoi_km: float = Field(default=5.0, ge=0, le=50)
    max_cloud_cover: int = Field(default=20, ge=0, le=100)
    collection: str = "sentinel-2-l2a"


class DetectionResult(BaseModel):
    image_id: str
    detection_count: int
    predictions: List[dict]
    result_url: str


# Setup auth
def verify_api_key(api_key: str = Depends(API_KEY_HEADER)) -> str:
    if api_key != os.getenv("API_KEY", "development"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.post("/search", response_model=List[dict])
async def search_images(search: LocationSearch, api_key: str = Depends(verify_api_key)):
    """Search for satellite imagery at a specific location and date"""
    try:
        catalog = init_planetary_computer()
        if not catalog:
            raise HTTPException(
                status_code=503, detail="Planetary Computer unavailable"
            )

        items = search_satellite_imagery(
            catalog=catalog,
            lat=search.lat,
            lon=search.lon,
            date_start=search.date.isoformat(),
            date_end=(search.date + timedelta(days=1)).isoformat(),
            collection=search.collection,
            max_cloud_cover=search.max_cloud_cover,
            expand_aoi_km=search.expand_aoi_km,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/{image_id}")
async def detect_fires(
    image_id: str,
    background_tasks: BackgroundTasks,
    quality: ImageQuality = ImageQuality.MEDIUM,
    api_key: str = Depends(verify_api_key),
):
    """Run fire detection on a specific satellite image"""
    try:
        # Initialize processing
        catalog = init_planetary_computer()
        if not catalog:
            raise HTTPException(
                status_code=503, detail="Planetary Computer unavailable"
            )

        # Find image
        search = catalog.search(ids=[image_id])
        items = list(search.items())
        if not items:
            raise HTTPException(status_code=404, detail="Image not found")

        item = items[0]

        # Process image with specified quality
        config = ImageProcessingConfig.from_quality_preset(quality)

        # Start processing in background
        task_id = f"detect_{image_id}_{datetime.now().timestamp()}"
        background_tasks.add_task(process_detection, item, config, task_id)

        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}")
async def get_task_status(task_id: str, api_key: str = Depends(verify_api_key)):
    """Get the status of a detection task"""
    # Implementation of task status tracking
    pass


# Add more endpoints as needed...
