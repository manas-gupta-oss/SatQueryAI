"""POST /api/upload - accept one image and register it for later queries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from backend.config import settings
from backend.schemas import UploadResponse
from backend.store import EXTENSION_FORMATS, ImageRecord, parse_modality, store
from orchestration.state import ImageFormat

logger = logging.getLogger(__name__)
router = APIRouter()

#: GeoKeyDirectoryTag. Its presence is what makes a TIFF a GeoTIFF.
_GEOKEY_TAG = 34735


def _probe(path: Path, fmt: ImageFormat) -> Tuple[Optional[int], Optional[int], ImageFormat, bool]:
    """Read the pixel dimensions and, for TIFFs, decide GeoTIFF vs plain TIFF."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            georeferenced = False
            if fmt is ImageFormat.TIFF:
                tags = getattr(img, "tag_v2", None)
                if tags is not None and _GEOKEY_TAG in tags:
                    fmt, georeferenced = ImageFormat.GEOTIFF, True
            return width, height, fmt, georeferenced
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"That file could not be read as an image ({type(exc).__name__}).",
        ) from exc


@router.post("/api/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    modality: Optional[str] = Form(None),
    sensor: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
    role_hint: Optional[str] = Form(None),
) -> UploadResponse:
    name = file.filename or "upload"
    extension = Path(name).suffix.lower()
    if extension not in EXTENSION_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{extension or name}' is not a supported image type. "
                f"Accepted: {', '.join(sorted(EXTENSION_FORMATS))}."
            ),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)}MB limit.",
        )

    settings.ensure_dirs()
    image_id = store.new_image_id()
    stored_name = f"{image_id}{extension}"
    destination = settings.uploads_dir / stored_name
    destination.write_bytes(payload)

    fmt = EXTENSION_FORMATS[extension]
    width, height, fmt, georeferenced = _probe(destination, fmt)

    record = store.put_image(ImageRecord(
        image_id=image_id,
        path=destination,
        filename=name,
        fmt=fmt,
        modality=parse_modality(modality),
        size_bytes=len(payload),
        width=width,
        height=height,
        georeferenced=georeferenced,
        sensor=(sensor or None),
        timestamp=(timestamp or None),
        role_hint=(role_hint or None),
    ))
    logger.info("stored %s (%s, %s bytes) as %s", name, fmt.value, len(payload), image_id)

    return UploadResponse(
        image_id=record.image_id,
        filename=record.filename,
        width=record.width,
        height=record.height,
        format=record.fmt.value,
        modality=record.modality.value,
        sensor=record.sensor,
        timestamp=record.timestamp,
        role_hint=record.role_hint,
        size_bytes=record.size_bytes,
        url=f"/media/uploads/{stored_name}",
    )
