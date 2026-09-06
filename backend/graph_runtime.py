"""
Assemble the LangGraph pipeline once, with this deployment's router and workers.

Both extension points used here are the ones orchestration/graph.py documents:
`worker_impls=` for the specialists and `boss_impl=` for the router. Nothing in
orchestration/ is modified to plug the models in.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from backend.config import settings
from backend.workers.nodes import model_worker_impls
from orchestration.graph import build_graph
from orchestration.state import WorkerId

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_graph: Any = None
_router_detail: str = "not built"


def _boss_impl():
    """
    None means "use the real Qwen2.5-3B BOSS in orchestration/nodes/boss_node.py".
    Anything else is a node built from a (query, bundle) -> BossDecision callable.
    """
    global _router_detail

    if settings.router == "llm":
        from orchestration.nodes.boss_node import get_boss

        # Bind the singleton to this deployment's config before the graph can
        # instantiate it with defaults.
        get_boss(config_path=settings.boss_config_path)
        _router_detail = "Qwen2.5-3B-Instruct, 4-bit, constrained JSON tool call"
        logger.info("router: LLM BOSS (%s)", settings.boss_config_path)
        return None

    from backend.routing.heuristic import heuristic_decide
    from orchestration.nodes.boss_node import make_boss_node

    _router_detail = (
        "deterministic rule router implementing R1-R5 / V1-V7; no weights, no VRAM"
    )
    logger.info("router: deterministic heuristic (set SATQUERY_ROUTER=llm for the 3B BOSS)")
    return make_boss_node(heuristic_decide)


def get_graph():
    """The compiled graph. Built on first use and reused for the process lifetime."""
    global _graph
    if _graph is not None:
        return _graph
    with _lock:
        if _graph is None:
            impls: Dict[WorkerId, Any] = (
                {} if settings.workers == "stub" else model_worker_impls()
            )
            _graph = build_graph(worker_impls=impls, boss_impl=_boss_impl())
            logger.info(
                "graph compiled (workers=%s, model-backed=%s)",
                settings.workers,
                sorted(w.value for w in impls) or "none",
            )
    return _graph


def router_detail() -> str:
    if _graph is None:
        _boss_impl_probe()
    return _router_detail


def _boss_impl_probe() -> None:
    """Fill in the router description without compiling the graph."""
    global _router_detail
    if settings.router == "llm":
        _router_detail = "Qwen2.5-3B-Instruct, 4-bit, constrained JSON tool call"
    else:
        _router_detail = (
            "deterministic rule router implementing R1-R5 / V1-V7; no weights, no VRAM"
        )


def worker_implementations() -> Dict[str, str]:
    """
    worker id -> 'model' | 'stub', for /api/health and /api/workers.

    Reports what will *actually* run, not what was configured: once a model load
    has been attempted and failed, a model-backed worker is downgraded to 'stub'
    here, so the UI's architecture panel never claims a specialist the machine
    cannot run.
    """
    from backend.workers.reporter import reporter
    from orchestration.tool_schema import WORKER_REGISTRY

    backed = set() if settings.workers == "stub" else set(model_worker_impls())
    if reporter.attempted and not reporter.available:
        backed = set()
    return {
        worker.value: ("model" if worker in backed else "stub")
        for worker in WORKER_REGISTRY
    }
