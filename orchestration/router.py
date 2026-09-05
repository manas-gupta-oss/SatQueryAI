"""
SatQueryAI - LangGraph conditional routing.

    state["boss_decision"].target_worker
        |
        +-- worker1 --> "worker1" node
        +-- worker2 --> "worker2" node
        +-- worker3 --> "worker3" node
        +-- none / invalid / missing --> "reject" node

Scope, deliberately narrow. This module:
  * reads the decision THE BOSS already made;
  * verifies that decision is structurally valid and contract-compatible;
  * returns the name of the next graph node.

It does NOT: run any model, read image pixels, decide the semantic task, or
know anything about worker internals. Node names and compatibility rules both
come from the shared contract in tool_schema.py, so adding or retiring a worker
is a registry edit, not a router edit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from orchestration.state import (
    BossDecision,
    GraphState,
    InputBundle,
    ValidationStatus,
    WorkerId,
)
from orchestration.tool_schema import WORKER_REGISTRY, check_compatibility

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Node names
# --------------------------------------------------------------------------- #

NODE_BOSS = "boss"
NODE_REJECT = "reject"
NODE_FINALIZE = "finalize"

# worker1 -> "worker1", ... derived from the registry so the two never drift.
WORKER_NODE_NAMES: Dict[WorkerId, str] = {
    worker_id: worker_id.value for worker_id in WORKER_REGISTRY
}

#: Every node the BOSS conditional edge may hand control to.
ROUTE_TARGETS: List[str] = list(WORKER_NODE_NAMES.values()) + [NODE_REJECT]


def node_name_for(worker: WorkerId) -> Optional[str]:
    """Graph node name for a worker, or None if it is not a registered worker."""
    return WORKER_NODE_NAMES.get(worker)


# --------------------------------------------------------------------------- #
# Decision verification
# --------------------------------------------------------------------------- #


def _coerce_decision(raw: Any) -> Optional[BossDecision]:
    """Accept a BossDecision or a plain dict (e.g. replayed from a saved trace)."""
    if raw is None:
        return None
    if isinstance(raw, BossDecision):
        return raw
    if isinstance(raw, dict):
        try:
            return BossDecision(**raw)
        except Exception as exc:
            logger.warning("boss_decision dict does not match the contract: %s", exc)
            return None
    logger.warning("boss_decision has unexpected type %s", type(raw).__name__)
    return None


def _coerce_bundle(raw: Any) -> InputBundle:
    """The graph may carry the bundle as a model or as a plain dict from the API."""
    if isinstance(raw, InputBundle):
        return raw
    if isinstance(raw, dict):
        try:
            return InputBundle(**raw)
        except Exception:
            return InputBundle()
    return InputBundle()


def verify_decision(state: GraphState) -> tuple[bool, str]:
    """
    Is this decision safe to execute? Returns (ok, reason).

    Four gates, cheapest first:
      1. a decision exists and parses against the contract;
      2. BOSS marked it valid;
      3. it names a registered worker;
      4. the deterministic contract still accepts the task/worker/input combination.

    Gate 4 is redundant when the decision came from boss_node (which already ran
    validate_decision). It is kept because the router is the last thing standing
    between a decision and a worker call, and decisions can also be injected by
    the backend, a replayed trace, or a teammate's test harness.
    """
    decision = _coerce_decision(state.get("boss_decision"))
    if decision is None:
        return False, "no usable routing decision was produced"

    if decision.validation.status != ValidationStatus.PASS:
        return False, decision.validation.reason or "the router rejected the request"

    if decision.target_worker == WorkerId.NONE:
        return False, "no worker was selected for this request"

    if node_name_for(decision.target_worker) is None:
        return False, f"'{decision.target_worker.value}' is not a registered worker"

    verdict = check_compatibility(
        decision.task_type,
        decision.target_worker,
        _coerce_bundle(state.get("inputs")),
        decision.params,
    )
    if verdict.status != ValidationStatus.PASS:
        return False, verdict.reason

    return True, ""


# --------------------------------------------------------------------------- #
# The conditional edge
# --------------------------------------------------------------------------- #


def route_from_boss(state: GraphState) -> str:
    """
    LangGraph conditional-edge function. Returns the next node name.

    Wire it up as:
        graph.add_conditional_edges(NODE_BOSS, route_from_boss, ROUTE_TARGETS)
    """
    ok, reason = verify_decision(state)
    if not ok:
        logger.info("routing to %s: %s", NODE_REJECT, reason)
        return NODE_REJECT

    decision = _coerce_decision(state["boss_decision"])
    target = node_name_for(decision.target_worker)
    logger.info(
        "routing to %s (task=%s, confidence=%.2f)",
        target,
        decision.task_type.value,
        decision.confidence,
    )
    return target
