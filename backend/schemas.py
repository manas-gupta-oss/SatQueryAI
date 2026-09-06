"""
SatQueryAI backend - HTTP request/response models.

These describe the *wire* format only. The graph's own contract lives in
orchestration/state.py and is not duplicated here; query responses are built by
orchestration.nodes.terminal_nodes.finalize_node and then augmented with the
report fields below.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: str
    modality: str
    sensor: Optional[str] = None
    timestamp: Optional[str] = None
    role_hint: Optional[str] = None
    size_bytes: int
    url: str = Field("", description="Server path where the stored image can be fetched.")


class QueryRequest(BaseModel):
    query: str
    image_ids: List[str] = Field(default_factory=list)
    co_registered: Optional[bool] = None
    same_location: Optional[bool] = None


class WorkerInfo(BaseModel):
    id: str
    display_name: str
    short_name: str
    description: str
    tuned_on: str
    tasks: List[str]
    min_images: int
    max_images: int
    implementation: str = Field(
        "stub", description="'model' when a fine-tuned specialist is wired in, else 'stub'."
    )


class HealthResponse(BaseModel):
    status: str
    router: str
    router_detail: str
    workers: Dict[str, str]
    model_loaded: bool
    model_error: Optional[str] = None
    gpu: Optional[str] = None
    benchmark_mode: bool
