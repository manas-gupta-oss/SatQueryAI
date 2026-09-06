"""
Routing tests for the deterministic rule router in backend/routing/heuristic.py.

Companion to tests/test_routing.py, which covers the same graph edge driven by
raw LLM emissions. This file drives it from real user phrasings instead, which
is what the demo actually exercises. No model, no GPU, no network.

Run standalone:   python -m tests.test_backend_routing
Run under pytest: pytest tests/test_backend_routing.py -v
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from backend.routing.heuristic import heuristic_decide
from orchestration.router import NODE_REJECT, route_from_boss
from orchestration.state import (
    GraphState,
    ImageFormat,
    ImageMeta,
    InputBundle,
    Modality,
    TaskType,
    ValidationCode,
)


def img(
    image_id: str,
    modality: Modality = Modality.OPTICAL,
    fmt: ImageFormat = ImageFormat.PNG,
    timestamp: Optional[str] = None,
    role_hint: Optional[str] = None,
) -> ImageMeta:
    return ImageMeta(
        image_id=image_id,
        path=f"/data/{image_id}.png",
        modality=modality,
        format=fmt,
        georeferenced=fmt is ImageFormat.GEOTIFF,
        timestamp=timestamp,
        role_hint=role_hint,
    )


def bundle(*images: ImageMeta, **kwargs) -> InputBundle:
    # benchmark_mode mirrors the backend default: both specialists are tuned on
    # PNG benchmark datasets, so PNG uploads are the normal input.
    kwargs.setdefault("benchmark_mode", True)
    return InputBundle(images=list(images), **kwargs)


ONE_OPTICAL = lambda: bundle(img("img_0"))
ONE_SAR = lambda: bundle(img("img_0", Modality.SAR))
PAIR = lambda: bundle(
    img("img_0", timestamp="2019-01-01", role_hint="pre"),
    img("img_1", timestamp="2024-01-01", role_hint="post"),
)
OPTICAL_SAR = lambda: bundle(img("img_0"), img("img_1", Modality.SAR))

#: (label, query, bundle, expected node, expected task, expected reject code)
CASES: List[Tuple[str, str, InputBundle, str, Optional[TaskType], Optional[ValidationCode]]] = [
    # --- worker1: single optical image ------------------------------------- #
    ("open description", "Describe this satellite scene in detail.",
     ONE_OPTICAL(), "worker1", TaskType.CAPTIONING, None),
    ("bare 'what is in this image'", "What is in this image?",
     ONE_OPTICAL(), "worker1", TaskType.CAPTIONING, None),
    ("unclassifiable query defaults to captioning", "satellite photo",
     ONE_OPTICAL(), "worker1", TaskType.CAPTIONING, None),
    ("counting question", "How many ships are in this image?",
     ONE_OPTICAL(), "worker1", TaskType.SINGLE_VQA, None),
    ("regional question is VQA, not captioning", "What is in the top right of the image?",
     ONE_OPTICAL(), "worker1", TaskType.SINGLE_VQA, None),
    ("yes/no question", "Is there a bridge over the river?",
     ONE_OPTICAL(), "worker1", TaskType.SINGLE_VQA, None),
    ("grounding by 'highlight'", "Highlight the ships in the harbour",
     ONE_OPTICAL(), "worker1", TaskType.GROUNDING, None),
    ("grounding by 'where is'", "Where is the airport runway?",
     ONE_OPTICAL(), "worker1", TaskType.GROUNDING, None),

    # --- worker2: SAR is never sent to the optical-only specialist ---------- #
    ("lone SAR question", "What structures are visible here?",
     ONE_SAR(), "worker2", TaskType.SINGLE_VQA, None),
    ("lone SAR caption request is restated as VQA", "Describe this scene.",
     ONE_SAR(), "worker2", TaskType.SINGLE_VQA, None),
    ("optical + SAR fusion", "Combine these two sources into one assessment.",
     OPTICAL_SAR(), "worker2", TaskType.CROSS_MODAL_FUSION, None),

    # --- worker3: bi-temporal ---------------------------------------------- #
    ("open 'what changed' is a description, not VQA", "What changed between these two images?",
     PAIR(), "worker3", TaskType.CHANGE_DESCRIPTION, None),
    ("'describe the differences'", "Describe the differences between these images.",
     PAIR(), "worker3", TaskType.CHANGE_DESCRIPTION, None),
    ("yes/no change question", "Has the built-up area increased?",
     PAIR(), "worker3", TaskType.CHANGE_VQA, None),
    ("quantity change question", "How many new buildings were added?",
     PAIR(), "worker3", TaskType.CHANGE_VQA, None),
    ("verb outside the common list still extracts a focus", "Has the water body shrunk?",
     PAIR(), "worker3", TaskType.CHANGE_VQA, None),
    ("two images, no stated task, assumes change", "Take a look at these.",
     PAIR(), "worker3", TaskType.CHANGE_DESCRIPTION, None),

    # --- rejections --------------------------------------------------------- #
    ("change request with one image", "What changed since last year?",
     ONE_OPTICAL(), NODE_REJECT, None, ValidationCode.IMAGE_COUNT_MISMATCH),
    ("grounding with nothing named", "Where is it?",
     ONE_OPTICAL(), NODE_REJECT, None, ValidationCode.MISSING_REQUIRED_PARAM),
    ("no images", "Describe the scene.",
     bundle(), NODE_REJECT, None, ValidationCode.NO_IMAGES),
    ("three images", "Describe these.",
     bundle(img("img_0"), img("img_1"), img("img_2")),
     NODE_REJECT, None, ValidationCode.IMAGE_COUNT_MISMATCH),
    ("explicitly non-co-registered pair", "What changed between these?",
     bundle(img("img_0"), img("img_1"), co_registered=False),
     NODE_REJECT, None, ValidationCode.NOT_CO_REGISTERED),
    ("explicitly different locations", "What changed between these?",
     bundle(img("img_0"), img("img_1"), same_location=False),
     NODE_REJECT, None, ValidationCode.NOT_SAME_LOCATION),
    ("PNG outside benchmark mode", "Describe this scene.",
     InputBundle(images=[img("img_0", fmt=ImageFormat.PNG)], benchmark_mode=False),
     NODE_REJECT, None, ValidationCode.UNSUPPORTED_FORMAT),
]


def _run(query: str, inputs: InputBundle):
    decision = heuristic_decide(query, inputs)
    state: GraphState = {"query": query, "inputs": inputs, "boss_decision": decision}
    return decision, route_from_boss(state)


def test_heuristic_routing_cases():
    failures = []
    for label, query, inputs, expect_node, expect_task, expect_code in CASES:
        decision, node = _run(query, inputs)
        problems = []
        if node != expect_node:
            problems.append(f"node={node} (want {expect_node})")
        if expect_task is not None and decision.task_type != expect_task:
            problems.append(f"task={decision.task_type.value} (want {expect_task.value})")
        if expect_code is not None and decision.validation.code != expect_code:
            problems.append(f"code={decision.validation.code.value} (want {expect_code.value})")
        if problems:
            failures.append(f"{label}: " + ", ".join(problems))
    assert not failures, "\n".join(failures)


def test_grounding_always_carries_a_target_phrase():
    """A grounding route without target_phrase would fail inside the worker."""
    for query in ("Highlight the ships in the harbour",
                  "Locate the storage tanks",
                  "Where is the airport runway?"):
        decision, node = _run(query, ONE_OPTICAL())
        assert node == "worker1", f"{query!r} -> {node}"
        if decision.task_type is TaskType.GROUNDING:
            assert (decision.params.target_phrase or "").strip(), query


def test_every_rejection_explains_itself():
    """A refusal the user cannot act on is a bug, not a safe default."""
    for label, query, inputs, expect_node, _task, _code in CASES:
        if expect_node != NODE_REJECT:
            continue
        decision, _ = _run(query, inputs)
        assert decision.validation.reason.strip(), f"{label} rejected with no reason"
        assert decision.audit_summary.strip(), f"{label} rejected with no audit summary"


def test_pair_roles_are_assigned_from_timestamps():
    """analyze_pair() is order-sensitive; the earlier image must be 'pre'."""
    inputs = bundle(
        img("img_late", timestamp="2024-01-01"),
        img("img_early", timestamp="2019-01-01"),
    )
    decision, node = _run("What changed between these two images?", inputs)
    assert node == "worker3"
    assert decision.image_assignment == {"pre": "img_early", "post": "img_late"}, \
        decision.image_assignment


def main() -> int:
    tests = [
        ("routing cases", test_heuristic_routing_cases),
        ("grounding carries a target phrase", test_grounding_always_carries_a_target_phrase),
        ("rejections explain themselves", test_every_rejection_explains_itself),
        ("pair roles from timestamps", test_pair_roles_are_assigned_from_timestamps),
    ]
    print("=" * 78)
    print("heuristic router - deterministic routing")
    print("=" * 78)

    for label, query, inputs, expect_node, _t, _c in CASES:
        decision, node = _run(query, inputs)
        ok = node == expect_node
        print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} -> {node}")

    failed = 0
    print("-" * 78)
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {exc}")
    print("=" * 78)
    print("all assertions passed" if not failed else f"{failed} test(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
