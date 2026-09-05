"""
Worker node adapters - STUBS.

These are placeholders owned by the orchestration layer so the graph can be
built, routed and tested end to end today. They contain NO machine-learning
code, load no weights, and read no pixels. The real worker1 / worker2 / worker3
implementations belong to their owners.

To plug a real worker in, hand build_graph() a callable per worker:

    from orchestration.graph import build_graph
    from orchestration.state import WorkerId

    graph = build_graph(worker_impls={
        WorkerId.WORKER1: my_worker1_node,   # (GraphState) -> {"worker_output": WorkerOutput}
        WorkerId.WORKER2: my_worker2_node,
        WorkerId.WORKER3: my_worker3_node,
    })

The contract a worker node must honour:

    INPUT   state["boss_decision"] - task_type, params, image_assignment
            state["inputs"]        - InputBundle with per-image paths/metadata
    OUTPUT  {"worker_output": WorkerOutput(...), "trace": [ExecutionStep(...)]}

A worker must not re-route, must not modify boss_decision, and should surface
failures through WorkerOutput.error rather than raising.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from orchestration.state import (
    ExecutionStep,
    GraphState,
    WorkerId,
    WorkerOutput,
)
from orchestration.tool_schema import WORKER_REGISTRY

WorkerNode = Callable[[GraphState], Dict[str, Any]]


def make_worker_stub(worker_id: WorkerId) -> WorkerNode:
    """
    Build a stub node for one worker.

    It echoes back exactly what it was dispatched with, which is what makes the
    routing tests meaningful: if the stub reports the wrong task or the wrong
    image roles, the orchestration layer is at fault, not the model.
    """
    spec = WORKER_REGISTRY[worker_id]

    def _node(state: GraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        decision = state["boss_decision"]
        params = decision.params.model_dump(exclude_none=True)
        assignment = decision.image_assignment or {}

        answer = (
            f"[STUB {worker_id.value}] would run '{decision.task_type.value}' "
            f"using the {spec.tuned_on}-adapted specialist. "
            f"params={params} images={assignment or 'single input'}"
        )

        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "worker_output": WorkerOutput(
                worker=worker_id,
                task_type=decision.task_type,
                answer=answer,
                confidence=None,          # a stub has no confidence to report
                extras={"stub": True, "dispatched_params": params},
            ),
            "trace": [
                ExecutionStep(
                    node=worker_id.value,
                    detail=f"stub executed task={decision.task_type.value}",
                    latency_ms=elapsed,
                )
            ],
        }

    _node.__name__ = f"{worker_id.value}_stub"
    return _node


def default_worker_stubs() -> Dict[WorkerId, WorkerNode]:
    """One stub per registered worker."""
    return {worker_id: make_worker_stub(worker_id) for worker_id in WORKER_REGISTRY}
