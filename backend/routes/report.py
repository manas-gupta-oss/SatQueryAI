"""GET /api/report/{request_id} - download the PDF for a completed query."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.store import store

router = APIRouter()


@router.get("/api/report/{request_id}")
def download_report(request_id: str, download: bool = True) -> FileResponse:
    record = store.get_report(request_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No report for that request id. Reports live for the lifetime of the server.",
        )
    if record.pdf_path is None or not record.pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(record.response or {}).get("report_error")
            or "The report for that request could not be generated.",
        )
    return FileResponse(
        path=record.pdf_path,
        media_type="application/pdf",
        # inline (download=false) lets the browser preview it in a tab.
        filename=f"satquery-report-{request_id}.pdf" if download else None,
        headers=None if download else {"Content-Disposition": "inline"},
    )


@router.get("/api/report/{request_id}/json")
def report_json(request_id: str) -> dict:
    """The structured result behind the PDF - useful for judges and for debugging."""
    record = store.get_report(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No report for that request id.")
    return {
        "request_id": record.request_id,
        "created_at": record.created_at,
        "response": record.response,
        "envelope": record.envelope,
        "validation": record.validation,
    }
