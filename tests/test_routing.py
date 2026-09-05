"""
Deterministic routing tests for THE BOSS. NO MODEL REQUIRED.

Each case supplies the raw emission Qwen would produce for a query, then runs
the real production path:

    raw text -> decision_from_raw()  [parse + deterministic validation]
             -> route_from_boss()    [the LangGraph conditional edge]

so parsing, the compatibility gate, and dispatch are all exercised without
downloading 3B weights. Several cases deliberately feed a WRONG model emission
to prove the deterministic gate overrides the model rather than trusting it.

Run standalone:   python -m tests.test_routing
Run under pytest: pytest tests/test_routing.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

from orchestration.nodes.boss_node import decision_from_raw
from orchestration.router import NODE_REJECT, route_from_boss
from orchestration.state import (
    ImageFormat,
    ImageMeta,
    InputBundle,
    Modality,
    TaskType,
    ValidationCode,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def img(
    image_id: str,
    modality: Modality,
    fmt: ImageFormat = ImageFormat.GEOTIFF,
    timestamp: Optional[str] = None,
    role_hint: Optional[str] = None,
) -> ImageMeta:
    return ImageMeta(
        image_id=image_id,
        path=f"/data/{image_id}",
        modality=modality,
        format=fmt,
        georeferenced=fmt is ImageFormat.GEOTIFF,
        timestamp=timestamp,
        role_hint=role_hint,
    )


def optical_single() -> InputBundle:
    return InputBundle(images=[img("img_0", Modality.OPTICAL)])


def sar_single() -> InputBundle:
    return InputBundle(images=[img("img_0", Modality.SAR)])


def optical_sar_pair(co_registered=True, same_location=True) -> InputBundle:
    return InputBundle(
        images=[img("img_0", Modality.OPTICAL), img("img_1", Modality.SAR)],
        co_registered=co_registered,
        same_location=same_location,
    )


def temporal_pair(co_registered=True, same_location=True) -> InputBundle:
    return InputBundle(
        images=[
            img("img_0", Modality.OPTICAL, timestamp="2020-04-01", role_hint="pre"),
            img("img_1", Modality.OPTICAL, timestamp="2024-04-01", role_hint="post"),
        ],
        co_registered=co_registered,
        same_location=same_location,
    )


def call(name: str, args: dict, audit: str = "test", conf: float = 0.9) -> str:
    """Render a model emission exactly as the constrained decoder produces it."""
    import json

    return (
        "<tool_call>\n"
        + json.dumps(
            {"name": name, "arguments": args, "audit_summary": audit, "confidence": conf}
        )
        + "\n</tool_call>"
    )


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    name: str
    query: str
    bundle: InputBundle
    raw: str
    expect_node: str
    expect_task: Optional[TaskType] = None
    expect_code: Optional[ValidationCode] = None
    expect_assumption: bool = False
    expect_roles: dict = field(default_factory=dict)
    note: str = ""


CASES: List[Case] = [
    # ---- 1-3: single optical image, all three worker1 sub-tasks -------------
    Case(
        name="optical single-image VQA",
        query="How many runways are visible in this airport?",
        bundle=optical_single(),
        raw=call(
            "call_worker1",
            {
                "task_type": "single_vqa",
                "question": "How many runways are visible in this airport?",
            },
        ),
        expect_node="worker1",
        expect_task=TaskType.SINGLE_VQA,
    ),
    Case(
        name="optical captioning",
        query="Describe the land-cover and major objects visible in this image.",
        bundle=optical_single(),
        raw=call(
            "call_worker1", {"task_type": "captioning", "detail_level": "detailed"}
        ),
        expect_node="worker1",
        expect_task=TaskType.CAPTIONING,
    ),
    Case(
        name="optical grounding",
        query="Highlight the water body referred to in the query.",
        bundle=optical_single(),
        raw=call(
            "call_worker1",
            {"task_type": "grounding", "target_phrase": "the water body"},
        ),
        expect_node="worker1",
        expect_task=TaskType.GROUNDING,
    ),
    # ---- 4-6: worker2 -------------------------------------------------------
    Case(
        name="SAR single-image VQA goes to worker2, not worker1",
        query="How many vessels appear in this radar scene?",
        bundle=sar_single(),
        raw=call(
            "call_worker2",
            {"task_type": "single_vqa", "question": "How many vessels are visible?"},
        ),
        expect_node="worker2",
        expect_task=TaskType.SINGLE_VQA,
    ),
    Case(
        name="optical+SAR cross-modal fusion",
        query="Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        bundle=optical_sar_pair(),
        raw=call(
            "call_worker2",
            {
                "task_type": "cross_modal_fusion",
                "target_classes": ["built-up", "water"],
            },
        ),
        expect_node="worker2",
        expect_task=TaskType.CROSS_MODAL_FUSION,
        expect_roles={"optical": "img_0", "sar": "img_1"},
        note="image roles must be auto-filled when the model omits them",
    ),
    Case(
        name="optical+SAR question answering",
        query="Is the region under the cloud cover flooded, judging from both images?",
        bundle=optical_sar_pair(),
        raw=call(
            "call_worker2",
            {
                "task_type": "single_vqa",
                "question": "Is the cloud-covered region flooded?",
                "image_assignment": {"optical": "img_0", "sar": "img_1"},
            },
        ),
        expect_node="worker2",
        expect_task=TaskType.SINGLE_VQA,
    ),
    # ---- 7-8: worker3 -------------------------------------------------------
    Case(
        name="bi-temporal change VQA",
        query="Has the built-up area increased, decreased, or remained unchanged?",
        bundle=temporal_pair(),
        raw=call(
            "call_worker3",
            {
                "task_type": "change_vqa",
                "question": "Has the built-up area increased, decreased or remained "
                "unchanged?",
                "change_focus": "built-up area",
            },
        ),
        expect_node="worker3",
        expect_task=TaskType.CHANGE_VQA,
        expect_roles={"pre": "img_0", "post": "img_1"},
    ),
    Case(
        name="bi-temporal change description with change map",
        query="What changed between these two dates, and where did the change occur?",
        bundle=temporal_pair(),
        raw=call(
            "call_worker3",
            {"task_type": "change_description", "return_change_map": True},
        ),
        expect_node="worker3",
        expect_task=TaskType.CHANGE_DESCRIPTION,
        expect_roles={"pre": "img_0", "post": "img_1"},
    ),
    # ---- 9-11: rejections ---------------------------------------------------
    Case(
        name="change request with only one image",
        query="What changed between these two dates?",
        bundle=optical_single(),
        raw=call(
            "reject_request",
            {
                "task_type": "change_description",
                "code": "image_count_mismatch",
                "reason": "Change analysis needs two images but one was supplied.",
            },
        ),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.IMAGE_COUNT_MISMATCH,
    ),
    Case(
        name="non-co-registered temporal pair (model wrongly accepts)",
        query="What changed between these two dates?",
        bundle=temporal_pair(co_registered=False),
        raw=call("call_worker3", {"task_type": "change_description"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.NOT_CO_REGISTERED,
        note="deterministic gate must override the model's accept",
    ),
    Case(
        name="PNG input outside benchmark mode",
        query="Describe this scene.",
        bundle=InputBundle(images=[img("img_0", Modality.OPTICAL, ImageFormat.PNG)]),
        raw=call("call_worker1", {"task_type": "captioning"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.UNSUPPORTED_FORMAT,
    ),
    # ---- 12-13: ambiguity resolved by assumption ----------------------------
    Case(
        name="ambiguous 'what changed?' routes to change_description",
        query="What changed?",
        bundle=temporal_pair(co_registered=True, same_location=True),
        raw=call(
            "call_worker3",
            {"task_type": "change_description"},
            audit="Open change query over a co-registered temporal pair.",
            conf=0.72,
        ),
        expect_node="worker3",
        expect_task=TaskType.CHANGE_DESCRIPTION,
    ),
    Case(
        name="unknown co-registration is allowed, with an assumption recorded",
        query="What changed between these two images?",
        bundle=temporal_pair(co_registered=None, same_location=None),
        raw=call("call_worker3", {"task_type": "change_description"}),
        expect_node="worker3",
        expect_task=TaskType.CHANGE_DESCRIPTION,
        expect_assumption=True,
    ),
    # ---- 14-15: contract violations and malformed output --------------------
    Case(
        name="invalid worker/task pair (change_vqa on worker1)",
        query="What changed between these two dates?",
        bundle=temporal_pair(),
        raw=call("call_worker1", {"task_type": "change_vqa", "question": "What changed?"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.UNSUPPORTED_TASK_FOR_WORKER,
        note="worker1 does not implement change tasks",
    ),
    Case(
        name="malformed model output fails safe",
        query="Describe this scene.",
        bundle=optical_single(),
        raw="Sure! I think worker1 should handle this one.",
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.AMBIGUOUS_QUERY,
    ),
    # ---- 16-17: extra edge cases -------------------------------------------
    Case(
        name="grounding without a target phrase",
        query="Highlight it.",
        bundle=optical_single(),
        raw=call("call_worker1", {"task_type": "grounding"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.MISSING_REQUIRED_PARAM,
    ),
    Case(
        name="SAR image sent to the optical-only worker1",
        query="Describe this scene.",
        bundle=sar_single(),
        raw=call("call_worker1", {"task_type": "captioning"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.MODALITY_MISMATCH,
    ),
    Case(
        name="no images supplied",
        query="What changed between these two dates?",
        bundle=InputBundle(images=[]),
        raw=call("call_worker3", {"task_type": "change_description"}),
        expect_node=NODE_REJECT,
        expect_code=ValidationCode.NO_IMAGES,
        note="the empty-input gate fires before the image-count gate",
    ),
]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_case(case: Case) -> List[str]:
    """Execute one case through the real path. Returns a list of failure strings."""
    decision = decision_from_raw(case.raw, case.bundle)
    state = {"query": case.query, "inputs": case.bundle, "boss_decision": decision}
    node = route_from_boss(state)

    problems: List[str] = []
    if node != case.expect_node:
        problems.append(f"routed to '{node}', expected '{case.expect_node}'")
    if case.expect_task is not None and decision.task_type != case.expect_task:
        problems.append(
            f"task_type '{decision.task_type.value}', expected '{case.expect_task.value}'"
        )
    if case.expect_code is not None and decision.validation.code != case.expect_code:
        problems.append(
            f"validation code '{decision.validation.code.value}', "
            f"expected '{case.expect_code.value}'"
        )
    if case.expect_assumption and not decision.assumptions:
        problems.append("expected an assumption to be recorded, found none")
    for role, image_id in case.expect_roles.items():
        if decision.image_assignment.get(role) != image_id:
            problems.append(
                f"image role '{role}' = {decision.image_assignment.get(role)!r}, "
                f"expected {image_id!r}"
            )
    if not decision.audit_summary:
        problems.append("audit_summary is empty")
    return problems


def test_routing_cases():
    """Single pytest entry point covering every case."""
    failures = []
    for case in CASES:
        problems = run_case(case)
        if problems:
            failures.append(f"{case.name}: " + "; ".join(problems))
    assert not failures, "\n".join(failures)


def test_every_reject_case_names_a_reason():
    """A rejection the user cannot act on is a bug."""
    for case in CASES:
        if case.expect_node != NODE_REJECT:
            continue
        decision = decision_from_raw(case.raw, case.bundle)
        assert decision.validation.reason, f"{case.name}: empty rejection reason"
        assert decision.target_worker.value == "none", (
            f"{case.name}: rejected decision still names {decision.target_worker.value}"
        )


def main() -> int:
    width = max(len(c.name) for c in CASES) + 2
    passed = 0
    print(f"\nBOSS routing tests - {len(CASES)} cases (no model required)\n")
    print(f"{'':4}{'case':<{width}}{'-> node':<12}result")
    print("-" * (width + 40))
    for i, case in enumerate(CASES, 1):
        problems = run_case(case)
        ok = not problems
        passed += ok
        print(
            f"{i:>3} {case.name:<{width}}{case.expect_node:<12}"
            + ("PASS" if ok else "FAIL")
        )
        for problem in problems:
            print(f"{'':>4}  ! {problem}")
        if case.note and ok:
            print(f"{'':>4}  ({case.note})")
    print("-" * (width + 40))
    print(f"{passed}/{len(CASES)} passed\n")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
