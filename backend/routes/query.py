"""
POST /api/query - run one query through the graph and produce its PDF report.

The response body is whatever finalize_node assembled, plus the report fields.
It is deliberately not rebuilt here: the orchestrator owns the response shape,
and duplicating it would let the two drift.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.graph_runtime import get_graph, router_detail
from backend.report.overlay import side_by_side
from backend.report.pdf import build_report
from backend.schemas import QueryRequest
from backend.store import ImageRecord, ReportRecord, store
from orchestration.state import InputBundle, WorkerOutput, initial_state

logger = logging.getLogger(__name__)
router = APIRouter()


def _extras(state: Dict[str, Any]) -> Dict[str, Any]:
    output = state.get("worker_output")
    if isinstance(output, WorkerOutput):
        return output.extras or {}
    return {}


def _build_strip(records: List[ImageRecord], request_id: str) -> Optional[Path]:
    """The before/after figure for a two-image report."""
    if len(records) != 2:
        return None
    out = settings.overlays_dir / f"pair_{request_id}.png"
    try:
        return side_by_side(records[0].path, records[1].path, out)
    except Exception as exc:
        logger.warning("could not build the before/after strip: %s", exc)
        return None


# `def`, not `async def`: graph.invoke() runs a 3B router and a vision model
# synchronously. Starlette dispatches sync handlers to its thread pool, which
# keeps the event loop free; making this async would block every other request.
@router.post("/api/query")
def run_query(request: QueryRequest) -> Dict[str, Any]:
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="A query is required.")
    if not request.image_ids:
        raise HTTPException(status_code=400, detail="Upload at least one image first.")

    try:
        records = store.get_images(request.image_ids)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown image id '{exc.args[0]}'. Re-upload the image and try again.",
        ) from exc

    settings.ensure_dirs()
    request_id = store.new_request_id()

    bundle = InputBundle(
        images=[record.to_meta() for record in records],
        co_registered=request.co_registered,
        same_location=request.same_location,
        # Both specialists are benchmark-tuned on PNG datasets, so PNG/JPEG is
        # the normal input here. See Settings.benchmark_mode.
        benchmark_mode=settings.benchmark_mode,
    )

    final_state = get_graph().invoke(initial_state(query, bundle, request_id))
    response: Dict[str, Any] = final_state.get("final_response") or {}
    response.setdefault("request_id", request_id)

    extras = _extras(final_state)
    worker_output = final_state.get("worker_output")

    # finalize_node reports status "ok" whenever a worker returned anything at
    # all, and only surfaces state["error"]. A worker that failed, or a stub that
    # stood in for an unavailable model, would otherwise be indistinguishable
    # from a real answer.
    response["degraded"] = False
    if isinstance(worker_output, WorkerOutput) and worker_output.error:
        response["error"] = response.get("error") or worker_output.error
        if extras.get("model_unavailable"):
            # The pipeline ran; the answer just is not a model result. Flagged
            # rather than failed, so routing stays demonstrable without the GPU.
            response["degraded"] = True
        else:
            response["status"] = "error"

    envelope = extras.get("envelope")
    validation = extras.get("validation")
    overlay_url = extras.get("overlay_url")
    overlay_path = Path(extras["overlay_path"]) if extras.get("overlay_path") else None
    if overlay_path is None and isinstance(final_state.get("worker_output"), WorkerOutput):
        raw = final_state["worker_output"].overlay_path
        overlay_path = Path(raw) if raw else None

    strip_path = _build_strip(records, request_id) if len(records) == 2 else None

    # The PDF is part of the deliverable, but a rendering failure must not lose
    # the analysis that was already produced.
    pdf_path: Optional[Path] = None
    report_error: Optional[str] = None
    try:
        pdf_path = build_report(
            out_path=settings.reports_dir / f"{request_id}.pdf",
            response=response,
            image_records=records,
            envelope=envelope,
            validation=validation,
            overlay_path=overlay_path,
            strip_path=strip_path,
            router=router_detail(),
        )
    except Exception as exc:
        report_error = f"{type(exc).__name__}: {exc}"
        logger.exception("PDF generation failed for %s", request_id)

    response["report_url"] = f"/api/report/{request_id}" if pdf_path else None
    response["report_error"] = report_error
    response["overlay_url"] = overlay_url
    response["images"] = [
        {
            "image_id": record.image_id,
            "filename": record.filename,
            "url": f"/media/uploads/{record.path.name}",
            "modality": record.modality.value,
            "format": record.fmt.value,
            "width": record.width,
            "height": record.height,
        }
        for record in records
    ]
    response["analysis"] = (envelope or {}).get("analysis")
    response["model"] = (envelope or {}).get("model")
    response["self_consistency"] = (
        {
            "severity": validation.get("severity"),
            "issue_count": len(validation.get("issues") or []),
            "user_message": validation.get("user_message") or "",
        }
        if validation
        else None
    )

    store.put_report(ReportRecord(
        request_id=request_id,
        response=response,
        pdf_path=pdf_path,
        envelope=envelope,
        validation=validation,
    ))
    return response
