"""
Terminal graph nodes: reject and finalize.

reject_node   - the safe exit taken whenever no worker may be executed.
finalize_node - assembles the response the API returns, including the auditable
                execution summary the problem statement grades.

Neither node runs a model or reads pixels.
"""

from __future__ import annotations

from typing import Any, Dict, List

from orchestration.router import verify_decision
from orchestration.state import (
    BossDecision,
    ExecutionStep,
    GraphState,
    ValidationStatus,
    WorkerId,
)
from orchestration.tool_schema import WORKER_REGISTRY


def reject_node(state: GraphState) -> Dict[str, Any]:
    """
    No worker runs here. Explains why, in the user's terms, and records it.

    The reason is recomputed through the same verification the router used, so
    the message the user sees is always the actual gate that fired.
    """
    _, reason = verify_decision(state)
    decision = state.get("boss_decision")
    code = (
        decision.validation.code.value
        if isinstance(decision, BossDecision)
        else "no_decision"
    )
    requested = (
        decision.task_type.value if isinstance(decision, BossDecision) else "unknown"
    )

    return {
        "error": reason,
        "trace": [
            ExecutionStep(
                node="reject",
                detail=f"no worker invoked (requested={requested}, code={code}): {reason}",
            )
        ],
    }


def _audit_lines(state: GraphState) -> List[str]:
    """The observable execution trace: task, worker, params, outcome."""
    decision = state.get("boss_decision")
    lines: List[str] = []
    if isinstance(decision, BossDecision):
        lines.append(f"Requested task: {decision.task_type.value}")
        if decision.target_worker != WorkerId.NONE:
            spec = WORKER_REGISTRY.get(decision.target_worker)
            lines.append(
                f"Selected tool: {decision.target_worker.value}"
                + (f" ({spec.display_name}, adapted on {spec.tuned_on})" if spec else "")
            )
            params = decision.params.model_dump(exclude_none=True)
            lines.append(f"Parameters: {params}")
            if decision.image_assignment:
                lines.append(f"Image roles: {decision.image_assignment}")
        else:
            lines.append("Selected tool: none (request rejected)")
        lines.append(
            f"Validation: {decision.validation.status.value}"
            + (f" - {decision.validation.reason}" if decision.validation.reason else "")
        )
        lines.append(f"Router confidence: {decision.confidence:.2f}")
        for assumption in decision.assumptions:
            lines.append(f"Assumption: {assumption}")
        if decision.audit_summary:
            lines.append(f"Summary: {decision.audit_summary}")
    for step in state.get("trace") or []:
        latency = f" ({step.latency_ms:.0f} ms)" if step.latency_ms else ""
        lines.append(f"[{step.node}]{latency} {step.detail}")
    return lines


def finalize_node(state: GraphState) -> Dict[str, Any]:
    """Assemble the API response. Shape mirrors what backend/schemas.py returns."""
    decision = state.get("boss_decision")
    worker_output = state.get("worker_output")

    response: Dict[str, Any] = {
        "request_id": state.get("request_id", ""),
        "query": state.get("query", ""),
        "status": "ok" if worker_output is not None else "rejected",
        "task_type": decision.task_type.value if isinstance(decision, BossDecision) else None,
        "worker": (
            decision.target_worker.value if isinstance(decision, BossDecision) else None
        ),
        "answer": worker_output.answer if worker_output else "",
        "confidence": worker_output.confidence if worker_output else None,
        "router_confidence": (
            decision.confidence if isinstance(decision, BossDecision) else None
        ),
        "visual_evidence": (
            {
                "boxes": worker_output.boxes,
                "box_labels": worker_output.box_labels,
                "mask_path": worker_output.mask_path,
                "overlay_path": worker_output.overlay_path,
            }
            if worker_output
            else None
        ),
        "validation": (
            decision.validation.model_dump() if isinstance(decision, BossDecision) else None
        ),
        "assumptions": (
            decision.assumptions if isinstance(decision, BossDecision) else []
        ),
        "audit_summary": (
            decision.audit_summary if isinstance(decision, BossDecision) else ""
        ),
        "execution_trace": _audit_lines(state),
        "error": state.get("error"),
    }

    if (
        isinstance(decision, BossDecision)
        and decision.validation.status == ValidationStatus.PASS
        and worker_output is None
        and not state.get("error")
    ):
        response["status"] = "error"
        response["error"] = "the selected worker produced no output"

    return {"final_response": response}
