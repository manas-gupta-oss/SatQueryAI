"""GET /api/health, GET /api/workers - what this deployment actually is."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter

from backend.config import settings
from backend.graph_runtime import router_detail, worker_implementations
from backend.schemas import HealthResponse, WorkerInfo
from backend.workers.reporter import reporter
from orchestration.tool_schema import WORKER_REGISTRY

router = APIRouter()

#: The compact labels the UI uses on the worker graph.
SHORT_NAMES = {
    "worker1": "Single Image Worker",
    "worker2": "Cross-Modal Worker",
    "worker3": "Bi-Temporal Worker",
}


def _gpu_name() -> Optional[str]:
    """Reported only if torch is already imported - never load CUDA just to answer /health."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        router=settings.router,
        router_detail=router_detail(),
        workers=worker_implementations(),
        model_loaded=reporter.available,
        model_error=reporter.error,
        gpu=_gpu_name(),
        benchmark_mode=settings.benchmark_mode,
    )


@router.get("/api/workers", response_model=List[WorkerInfo])
def workers() -> List[WorkerInfo]:
    """
    Derived from WORKER_REGISTRY, so the UI's architecture panel can never
    describe a worker the router does not actually have.
    """
    implementations = worker_implementations()
    return [
        WorkerInfo(
            id=spec.worker_id.value,
            display_name=spec.display_name,
            short_name=SHORT_NAMES.get(spec.worker_id.value, spec.display_name),
            description=spec.description,
            tuned_on=spec.tuned_on,
            tasks=sorted(task.value for task in spec.tasks),
            min_images=spec.min_images,
            max_images=spec.max_images,
            implementation=implementations.get(spec.worker_id.value, "stub"),
        )
        for spec in WORKER_REGISTRY.values()
    ]
