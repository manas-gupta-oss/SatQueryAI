"""
SatQueryAI backend.

    uvicorn backend.app:app --reload --port 8000

Wires together the three layers that already existed and had nothing joining
them:

    frontend/       React UI            ->  this API
    orchestration/  LangGraph + router  ->  build_graph(worker_impls=, boss_impl=)
    models/         the fine-tuned VLMs ->  backend/workers/nodes.py

and adds the deliverable the pipeline was missing: a PDF report per query.

Endpoints
    GET  /api/health                  what is loaded, which router, which workers
    GET  /api/workers                 the live worker registry
    POST /api/upload                  one image, multipart/form-data
    POST /api/query                   run the graph, generate the report
    GET  /api/report/{id}             the PDF
    GET  /api/report/{id}/json        the structured result behind it
    GET  /media/uploads|overlays/...  stored imagery and annotated overlays
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.graph_runtime import get_graph, worker_implementations
from backend.routes import meta, query, report, upload
from backend.workers.reporter import reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("satquery")


def _preload() -> None:
    """Warm the vision model off the request path so the first query is fast."""
    if reporter.load():
        logger.info("specialists preloaded")
    else:
        logger.warning("preload failed: %s", reporter.error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    get_graph()  # compile once, and fail loudly at boot rather than mid-demo
    logger.info(
        "SatQueryAI backend ready | router=%s | workers=%s | data=%s",
        settings.router, worker_implementations(), settings.data_dir,
    )
    if settings.preload_models:
        # In a thread: a 40 s model load must not delay the port opening, or a
        # frontend started alongside the backend sees connection refused.
        threading.Thread(target=_preload, name="preload", daemon=True).start()
    else:
        logger.info(
            "models load lazily on the first query (~40 s). Set SATQUERY_PRELOAD=1 "
            "to warm them at startup instead."
        )
    yield


app = FastAPI(
    title="SatQueryAI",
    description="Agentic remote-sensing vision-language analysis with PDF reporting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # The frontend needs to read the filename off a report download.
    expose_headers=["Content-Disposition"],
)

app.include_router(meta.router)
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(report.router)

settings.ensure_dirs()
app.mount("/media", StaticFiles(directory=str(settings.data_dir)), name="media")


@app.get("/")
def index() -> dict:
    return {
        "service": "SatQueryAI",
        "docs": "/docs",
        "health": "/api/health",
        "frontend": "run `npm run dev` in frontend/ and open http://localhost:5173",
    }
