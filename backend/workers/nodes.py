"""
The real worker nodes: fine-tuned specialists wired into the LangGraph pipeline.

These satisfy the contract documented in orchestration/nodes/worker_node.py -
they read `boss_decision` and `inputs`, they never re-route, never mutate the
decision, and surface failures through `WorkerOutput.error` rather than raising.
They are injected via `build_graph(worker_impls=...)`, so nothing in
orchestration/ is edited to make this work.

    worker1  single-image specialist   adapters/report-adapter-v2   (VRSBench)
    worker3  bi-temporal specialist    adapters/bitemporal-adapter  (LEVIR-CC)

worker2 (cross-modal optical+SAR) has a registry entry but no trained model, so
it deliberately keeps the orchestration-layer stub. A query routed there returns
an honest "not implemented" rather than a fabricated answer.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.report.overlay import draw_boxes
from backend.workers.answers import derive_change, derive_single
from backend.workers.reporter import reporter, validate_envelope
from orchestration.state import (
    BossDecision,
    ExecutionStep,
    GraphState,
    ImageMeta,
    InputBundle,
    TaskType,
    WorkerId,
    WorkerOutput,
)

logger = logging.getLogger(__name__)


def _bundle(state: GraphState) -> InputBundle:
    raw = state.get("inputs")
    if isinstance(raw, InputBundle):
        return raw
    if isinstance(raw, dict):
        return InputBundle(**raw)
    return InputBundle()


def _image_by_id(bundle: InputBundle, image_id: str) -> Optional[ImageMeta]:
    return next((i for i in bundle.images if i.image_id == image_id), None)


def _pair_order(decision: BossDecision, bundle: InputBundle) -> Tuple[ImageMeta, ImageMeta]:
    """
    Resolve which upload is the EARLIER one. analyze_pair() is order-sensitive:
    its first argument must be the 'before' image, or the change description
    comes out reversed.

    validate_decision() already fills image_assignment from role hints and then
    timestamps, so this normally just reads that. Upload order is the last
    resort and is recorded as an assumption by the router, not invented here.
    """
    assignment = decision.image_assignment or {}
    pre = _image_by_id(bundle, assignment.get("pre", ""))
    post = _image_by_id(bundle, assignment.get("post", ""))
    if pre and post and pre.image_id != post.image_id:
        return pre, post
    return bundle.images[0], bundle.images[1]


def _failure(
    worker_id: WorkerId, decision: BossDecision, message: str, elapsed_ms: float
) -> Dict[str, Any]:
    return {
        "worker_output": WorkerOutput(
            worker=worker_id,
            task_type=decision.task_type,
            answer="",
            error=message,
        ),
        "trace": [
            ExecutionStep(node=worker_id.value, detail=f"failed: {message}", latency_ms=elapsed_ms)
        ],
    }


def _stub_fallback(worker_id: WorkerId, state: GraphState, reason: str) -> Dict[str, Any]:
    """
    SATQUERY_WORKERS=auto and the specialists would not load.

    Rather than failing the query, hand it to the orchestration-layer stub so
    routing, validation and the report pipeline stay demonstrable - but mark the
    output unmistakably, in the answer text and in the trace, so nobody can
    mistake a stub echo for a model result.
    """
    from orchestration.nodes.worker_node import make_worker_stub

    result = make_worker_stub(worker_id)(state)
    output: WorkerOutput = result["worker_output"]
    output.error = f"specialist unavailable, stub used: {reason}"
    output.extras = {**(output.extras or {}), "model_unavailable": reason}
    output.answer = (
        "MODEL NOT LOADED - this is a routing stub, not a model answer.\n\n"
        f"Reason: {reason}\n\n" + output.answer
    )
    result["trace"] = [
        ExecutionStep(
            node=worker_id.value,
            detail=f"STUB (specialist unavailable: {reason})",
            latency_ms=result["trace"][0].latency_ms if result.get("trace") else None,
        )
    ]
    return result


def _specialists_ready(worker_id: WorkerId, state: GraphState) -> Optional[Dict[str, Any]]:
    """None when the model is usable, otherwise the response to return instead."""
    if reporter.load():
        return None
    reason = reporter.error or "vision model unavailable"
    if settings.workers == "auto":
        return _stub_fallback(worker_id, state, reason)
    return _failure(WorkerId(worker_id), state["boss_decision"], reason, 0.0)


def _annotate(
    source: Path, boxes: List[List[float]], labels: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Render the evidence overlay. Returns (filesystem path, public URL)."""
    if not boxes:
        return None, None
    name = f"overlay_{uuid.uuid4().hex[:12]}.png"
    out = settings.overlays_dir / name
    try:
        written = draw_boxes(source, boxes, labels, out)
    except Exception as exc:  # never let a drawing bug fail a query
        logger.warning("overlay rendering failed: %s", exc)
        return None, None
    if written is None:
        return None, None
    return str(written), f"/media/overlays/{name}"


# --------------------------------------------------------------------------- #
# worker1 - single-image specialist
# --------------------------------------------------------------------------- #


def worker1_node(state: GraphState) -> Dict[str, Any]:
    started = time.perf_counter()
    decision: BossDecision = state["boss_decision"]
    bundle = _bundle(state)

    if not bundle.images:
        return _failure(WorkerId.WORKER1, decision, "no image reached the worker",
                        (time.perf_counter() - started) * 1000.0)

    unavailable = _specialists_ready(WorkerId.WORKER1, state)
    if unavailable is not None:
        return unavailable

    image = bundle.images[0]
    try:
        envelope = reporter.analyze_image(Path(image.path))
    except RuntimeError as exc:
        return _failure(WorkerId.WORKER1, decision, str(exc),
                        (time.perf_counter() - started) * 1000.0)

    elapsed = (time.perf_counter() - started) * 1000.0

    # Rule 2 in MODELS.md: analyze_* never raises, so an unchecked status fails
    # silently. Check it before touching `analysis`.
    if envelope.get("status") != "ok":
        message = envelope.get("error") or "the specialist could not produce a structured result"
        result = _failure(WorkerId.WORKER1, decision, message, elapsed)
        result["worker_output"].extras = {"envelope": envelope}
        return result

    analysis = envelope.get("analysis") or {}
    validation = validate_envelope(envelope)
    answer, boxes, labels = derive_single(decision.task_type, decision.params, analysis)

    overlay_path, overlay_url = (None, None)
    if decision.params.return_visual_evidence or decision.task_type is TaskType.GROUNDING:
        overlay_path, overlay_url = _annotate(Path(image.path), boxes, labels)

    if validation.get("severity") == "reject":
        # Rejected output is still stored and still reported - a validator
        # visibly catching a bad generation is worth more in a demo than
        # pretending bad generations never happen (models/validate.py).
        answer = validation.get("user_message") or answer

    return {
        "worker_output": WorkerOutput(
            worker=WorkerId.WORKER1,
            task_type=decision.task_type,
            answer=answer,
            confidence=None,  # the adapters emit no calibrated confidence; see the PDF note
            boxes=boxes,
            box_labels=labels,
            overlay_path=overlay_path,
            extras={
                "envelope": envelope,
                "validation": validation,
                "overlay_url": overlay_url,
                "inference_seconds": envelope.get("inference_seconds"),
            },
        ),
        "trace": [
            ExecutionStep(
                node="worker1",
                detail=(
                    f"single-image specialist ran task={decision.task_type.value}; "
                    f"{len(analysis.get('objects') or [])} objects, "
                    f"{len(analysis.get('qa_pairs') or [])} qa pairs, "
                    f"validation={validation.get('severity')}"
                ),
                latency_ms=elapsed,
            )
        ],
    }


# --------------------------------------------------------------------------- #
# worker3 - bi-temporal change specialist
# --------------------------------------------------------------------------- #


def worker3_node(state: GraphState) -> Dict[str, Any]:
    started = time.perf_counter()
    decision: BossDecision = state["boss_decision"]
    bundle = _bundle(state)

    if len(bundle.images) < 2:
        return _failure(WorkerId.WORKER3, decision, "change analysis needs two images",
                        (time.perf_counter() - started) * 1000.0)

    unavailable = _specialists_ready(WorkerId.WORKER3, state)
    if unavailable is not None:
        return unavailable

    pre, post = _pair_order(decision, bundle)
    try:
        envelope = reporter.analyze_pair(Path(pre.path), Path(post.path))
    except RuntimeError as exc:
        return _failure(WorkerId.WORKER3, decision, str(exc),
                        (time.perf_counter() - started) * 1000.0)

    elapsed = (time.perf_counter() - started) * 1000.0

    if envelope.get("status") != "ok":
        message = envelope.get("error") or "the specialist could not produce a structured result"
        result = _failure(WorkerId.WORKER3, decision, message, elapsed)
        result["worker_output"].extras = {"envelope": envelope}
        return result

    analysis = envelope.get("analysis") or {}
    validation = validate_envelope(envelope)
    answer, boxes, labels = derive_change(decision.task_type, decision.params, analysis)

    # Change regions are located in the LATER image, so that is what they are
    # drawn on.
    overlay_path, overlay_url = _annotate(Path(post.path), boxes, labels)

    if validation.get("severity") == "reject":
        answer = validation.get("user_message") or answer

    return {
        "worker_output": WorkerOutput(
            worker=WorkerId.WORKER3,
            task_type=decision.task_type,
            answer=answer,
            confidence=None,
            boxes=boxes,
            box_labels=labels,
            mask_path=overlay_path if decision.params.return_change_map else None,
            overlay_path=overlay_path,
            extras={
                "envelope": envelope,
                "validation": validation,
                "overlay_url": overlay_url,
                "pre_image_id": pre.image_id,
                "post_image_id": post.image_id,
                "inference_seconds": envelope.get("inference_seconds"),
            },
        ),
        "trace": [
            ExecutionStep(
                node="worker3",
                detail=(
                    f"bi-temporal specialist ran task={decision.task_type.value}; "
                    f"change_detected={analysis.get('change_detected')}, "
                    f"{len(analysis.get('change_regions') or [])} regions, "
                    f"validation={validation.get('severity')}"
                ),
                latency_ms=elapsed,
            )
        ],
    }


def model_worker_impls() -> Dict[WorkerId, Any]:
    """
    The workers to hand to build_graph(). worker2 is absent on purpose so it
    keeps the orchestration-layer stub.
    """
    return {WorkerId.WORKER1: worker1_node, WorkerId.WORKER3: worker3_node}
