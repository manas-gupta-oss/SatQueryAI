"""
SatQueryAI backend - upload registry and report registry.

In-process and intentionally so: this is a demo backend, not a service. Both
registries are dicts guarded by a lock; the uploaded bytes and generated PDFs
are the only things that survive a restart, and nothing here assumes they do.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestration.state import ImageFormat, ImageMeta, Modality

# extension -> the format the contract in tool_schema.py reasons about.
EXTENSION_FORMATS: Dict[str, ImageFormat] = {
    ".png": ImageFormat.PNG,
    ".jpg": ImageFormat.JPEG,
    ".jpeg": ImageFormat.JPEG,
    ".tif": ImageFormat.TIFF,
    ".tiff": ImageFormat.TIFF,
}

MODALITY_ALIASES: Dict[str, Modality] = {
    "optical": Modality.OPTICAL,
    "rgb": Modality.OPTICAL,
    "multispectral": Modality.MULTISPECTRAL,
    "msi": Modality.MULTISPECTRAL,
    "sar": Modality.SAR,
    "radar": Modality.SAR,
}


def parse_modality(raw: Optional[str]) -> Modality:
    if not raw:
        return Modality.UNKNOWN
    return MODALITY_ALIASES.get(raw.strip().lower(), Modality.UNKNOWN)


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    filename: str
    fmt: ImageFormat
    modality: Modality
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    georeferenced: bool = False
    sensor: Optional[str] = None
    timestamp: Optional[str] = None
    role_hint: Optional[str] = None
    uploaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_meta(self) -> ImageMeta:
        """The graph-side view. BOSS reads this and never the pixels."""
        return ImageMeta(
            image_id=self.image_id,
            path=str(self.path),
            modality=self.modality,
            format=self.fmt,
            width=self.width,
            height=self.height,
            georeferenced=self.georeferenced,
            timestamp=self.timestamp,
            sensor=self.sensor,
            role_hint=self.role_hint,
        )


@dataclass
class ReportRecord:
    request_id: str
    response: Dict[str, Any]
    pdf_path: Optional[Path] = None
    envelope: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._images: Dict[str, ImageRecord] = {}
        self._reports: Dict[str, ReportRecord] = {}

    # -- images ------------------------------------------------------------- #

    @staticmethod
    def new_image_id() -> str:
        return f"img_{uuid.uuid4().hex[:12]}"

    def put_image(self, record: ImageRecord) -> ImageRecord:
        with self._lock:
            self._images[record.image_id] = record
        return record

    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        with self._lock:
            return self._images.get(image_id)

    def get_images(self, image_ids: List[str]) -> List[ImageRecord]:
        """Preserves caller order; raises KeyError naming the first unknown id."""
        with self._lock:
            missing = [i for i in image_ids if i not in self._images]
            if missing:
                raise KeyError(missing[0])
            return [self._images[i] for i in image_ids]

    # -- reports ------------------------------------------------------------ #

    @staticmethod
    def new_request_id() -> str:
        return f"req_{uuid.uuid4().hex[:12]}"

    def put_report(self, record: ReportRecord) -> ReportRecord:
        with self._lock:
            self._reports[record.request_id] = record
        return record

    def get_report(self, request_id: str) -> Optional[ReportRecord]:
        with self._lock:
            return self._reports.get(request_id)


store = Store()
