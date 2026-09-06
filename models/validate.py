r"""Self-consistency and hallucination checks for satquery report envelopes.

    from validate import validate
    report = reporter.analyze_image("scene.png")
    v = validate(report)
    if v["severity"] == "reject":
        show(v["user_message"])          # polite, specific, no stack trace
    store(report, validation=v)          # store it EITHER way -- see below

What this can and cannot claim
------------------------------
It checks INTERNAL CONSISTENCY and PLAUSIBILITY: the model contradicting
itself, inventing categories it was never trained on, or emitting structure
that cannot be rendered. Self-contradiction is strong evidence of a
hallucination, so this catches a real and useful class of bad output.

It CANNOT verify a claim against the actual image. Nothing downstream of the
model can. Describe this layer as "self-consistency and hallucination
detection", never as "we verify the answer is correct" -- the second claim
collapses under one pointed question.

Design notes
------------
Stdlib only, no GPU, no model. Safe to import anywhere -- the orchestrator
before writing to the database, and the PDF layer before rendering.

Thresholds are calibrated against the training targets rather than guessed.
Two consequences worth knowing:

  * POSITION vs BOX is checked. Objects labelled "left" sit at centre-x 0.12
    -0.15 (95th pct 0.29); "right" at 0.85-0.90 (5th pct 0.71). The bands do
    not overlap, so a contradiction here is real. The rule below still allows a
    generous margin (0.4/0.6) so only blatant disagreement fires.

  * SIZE vs BOX AREA is NOT checked, deliberately. In the training data
    obj_size "small" has median area 0.021 but 95th pct 0.533, and "large" has
    median 0.084 but 5th pct 0.0088 -- below small's median. The distributions
    overlap almost entirely, so such a check would reject correct output. It
    looks reasonable and is not.

Severity
--------
  ok      nothing found
  warn    store it, render it, flag it for review -- suspicious, not disqualifying
  reject  do not render as a finished report; show user_message instead

Rejected output should still be STORED, marked invalid with its reasons. You
want it for debugging, and a validation layer visibly catching a bad generation
is a better demo than pretending bad generations never happen.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# controlled vocabulary, taken from the training targets
# ---------------------------------------------------------------------------

# The exact 26 obj_cls values present in VRSBench train+val. Extracted from the
# data, NOT hand-written -- an earlier hand-written version misspelled three
# ('golf-field' for 'golffield', 'train-station' for 'trainstation') and invented
# 'expressway', which would have warned on correct output. Regenerate with:
#   python -c "import json,collections;c=collections.Counter(
#     str(o.get('obj_cls','')).strip().lower()
#     for f in ('train','val')
#     for l in open(f'data/vrsbench/{f}.jsonl',encoding='utf-8') if l.strip()
#     for o in json.loads(json.loads(l)['answer']).get('objects') or []);print(sorted(c))"
#
# NOTE 'building' is deliberately absent -- VRSBench has no such class. The model
# does emit it occasionally (4 of 93 objects). That is not a false statement about
# the image, it is a label outside the schema, so it warns rather than rejects.
KNOWN_OBJ_CLS = {
    "airplane", "airport", "baseball-diamond", "basketball-court", "bridge",
    "chimney", "container-crane", "dam", "expressway-service-area",
    "expressway-toll-station", "golffield", "ground-track-field", "harbor",
    "helicopter", "helipad", "overpass", "roundabout", "ship",
    "soccer-ball-field", "stadium", "storage-tank", "swimming-pool",
    "tennis-court", "trainstation", "vehicle", "windmill",
}

CHANGE_CLASSES = {"building", "road"}
CHANGE_EXTENTS = {"none", "minimal", "small", "moderate", "extensive",
                  "unquantified"}
REGION_SIZES = {"small", "medium", "large"}

# Object counts: 93 objects across 60 validation images, mean 1.55. A report
# claiming dozens is padding, not detail.
MAX_PLAUSIBLE_OBJECTS = 15
MAX_PLAUSIBLE_REGIONS = 12

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# "no change" phrasing, for catching a summary that contradicts change_detected
NEG_PHRASE = re.compile(
    r"\b(no difference|no change|same as before|identical|unchanged|"
    r"nothing has changed|remains the same|no significant change)\b", re.I)


def _issue(code: str, severity: str, message: str, where: str = "") -> dict:
    return {"code": code, "severity": severity, "message": message,
            "where": where}


def _centre(box: Any) -> tuple[float, float] | None:
    if not (isinstance(box, (list, tuple)) and len(box) >= 4):
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _degenerate(box: Any) -> bool:
    if not (isinstance(box, (list, tuple)) and len(box) >= 4):
        return True
    try:
        x0, y0, x1, y1 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return True
    return (x1 - x0) <= 0 or (y1 - y0) <= 0


# ---------------------------------------------------------------------------
# position vs box
# ---------------------------------------------------------------------------
# Margins are deliberately loose. Real "left" objects never exceed centre-x
# 0.29 at the 95th percentile, so requiring only < 0.6 fires on blatant
# disagreement and leaves correct output alone.

def _position_conflict(position: str, box: Any) -> str | None:
    c = _centre(box)
    if not c or not position:
        return None
    cx, cy = c
    p = position.lower()
    if "left" in p and cx > 0.6:
        return f"described as '{position}' but the box sits at x={cx:.2f}"
    if "right" in p and cx < 0.4:
        return f"described as '{position}' but the box sits at x={cx:.2f}"
    if "top" in p and cy > 0.6:
        return f"described as '{position}' but the box sits at y={cy:.2f}"
    if "bottom" in p and cy < 0.4:
        return f"described as '{position}' but the box sits at y={cy:.2f}"
    return None


# ---------------------------------------------------------------------------
# caption count vs object list
# ---------------------------------------------------------------------------
# "Five ships are visible" with one ship in objects[] is the counting failure
# this project already knows about, caught by the model disagreeing with itself.
# Only explicit numerals are checked -- "several" and "multiple" are ignored,
# since they carry no checkable count.

def _count_conflicts(caption: str, objects: list) -> list[tuple[str, int, int]]:
    if not caption:
        return []
    present: dict[str, int] = {}
    for o in objects:
        c = str(o.get("obj_cls") or "").strip().lower()
        if c:
            present[c] = present.get(c, 0) + 1

    out = []
    num = r"(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")"
    for cls in set(list(present) + list(KNOWN_OBJ_CLS)):
        # "storage-tank" is written "storage tank" or "storage tanks" in prose
        variants = {cls, cls.replace("-", " "), cls.replace("-", "")}
        for v in variants:
            pat = re.compile(rf"\b{num}\s+{re.escape(v)}s?\b", re.I)
            m = pat.search(caption)
            if not m:
                continue
            tok = m.group(1).lower()
            said = NUMBER_WORDS.get(tok, None)
            if said is None:
                try:
                    said = int(tok)
                except ValueError:
                    continue
            have = present.get(cls, 0)
            if said != have:
                out.append((cls, said, have))
            break
    return out


# ---------------------------------------------------------------------------
# per-task checks
# ---------------------------------------------------------------------------

def _check_single(a: dict) -> list[dict]:
    issues: list[dict] = []
    caption = str(a.get("caption") or "").strip()
    objects = [o for o in (a.get("objects") or []) if isinstance(o, dict)]

    if not caption:
        issues.append(_issue("empty_caption", "reject",
                             "no caption was produced", "analysis.caption"))
    if not objects:
        # Every one of the 60 validation images produced at least one object.
        # An empty list means the model answered in prose instead of the schema,
        # which is what a non-canonical prompt does.
        issues.append(_issue("no_objects", "reject",
                             "no objects were extracted, so the report would "
                             "carry no visual evidence", "analysis.objects"))
    if len(objects) > MAX_PLAUSIBLE_OBJECTS:
        issues.append(_issue("object_count_implausible", "warn",
                             f"{len(objects)} objects listed; typical is 1-3",
                             "analysis.objects"))
    if not (a.get("qa_pairs") or []):
        issues.append(_issue("no_qa_pairs", "warn",
                             "no question/answer pairs were produced",
                             "analysis.qa_pairs"))

    seen_boxes: dict[tuple, int] = {}
    for i, o in enumerate(objects):
        where = f"analysis.objects[{i}]"
        cls = str(o.get("obj_cls") or "").strip().lower()
        if not cls:
            issues.append(_issue("missing_class", "warn",
                                 "object has no class label", where))
        elif cls not in KNOWN_OBJ_CLS:
            issues.append(_issue("unknown_class", "warn",
                                 f"'{cls}' is outside the trained label set, so "
                                 f"class filters downstream will not match it",
                                 where))
        box = o.get("bbox_normalized")
        if box is None:
            issues.append(_issue("missing_box", "warn",
                                 "object has no drawable bounding box", where))
            continue
        if _degenerate(box):
            issues.append(_issue("degenerate_box", "warn",
                                 "bounding box has zero width or height", where))
            continue
        key = tuple(round(float(v), 3) for v in box[:4])
        if key in seen_boxes:
            issues.append(_issue("duplicate_box", "warn",
                                 f"identical box to objects[{seen_boxes[key]}]",
                                 where))
        else:
            seen_boxes[key] = i
        conflict = _position_conflict(str(o.get("obj_position") or ""), box)
        if conflict:
            issues.append(_issue("position_contradiction", "warn",
                                 conflict, where))

    for cls, said, have in _count_conflicts(caption, objects):
        issues.append(_issue("count_contradiction", "warn",
                             f"caption says {said} {cls}, but {have} "
                             f"{'was' if have == 1 else 'were'} listed",
                             "analysis.caption"))
    return issues


def _check_pair(a: dict) -> list[dict]:
    issues: list[dict] = []
    detected = bool(a.get("change_detected"))
    summary = str(a.get("change_summary") or "").strip()
    regions = [r for r in (a.get("change_regions") or []) if isinstance(r, dict)]
    classes = [str(c).strip().lower() for c in (a.get("changed_classes") or [])]
    extent = str(a.get("change_extent") or "").strip().lower()

    if not summary:
        issues.append(_issue("empty_summary", "reject",
                             "no change summary was produced",
                             "analysis.change_summary"))

    # The two strongest signals here are the model disagreeing with itself.
    if detected and NEG_PHRASE.search(summary):
        issues.append(_issue("summary_contradiction", "reject",
                             "change_detected is true but the summary says "
                             "nothing changed", "analysis.change_summary"))
    if not detected and regions:
        issues.append(_issue("regions_without_change", "reject",
                             "change_detected is false but change regions were "
                             "listed", "analysis.change_regions"))
    if detected and not regions and extent not in ("none", "unquantified", ""):
        issues.append(_issue("change_without_regions", "warn",
                             f"change reported with extent '{extent}' but no "
                             "region was localised", "analysis.change_regions"))
    if detected and not regions:
        issues.append(_issue("no_visual_evidence", "warn",
                             "change reported but no box to draw",
                             "analysis.change_regions"))

    if extent and extent not in CHANGE_EXTENTS:
        issues.append(_issue("unknown_extent", "warn",
                             f"'{extent}' is not a value this model was trained "
                             "to emit", "analysis.change_extent"))
    if len(regions) > MAX_PLAUSIBLE_REGIONS:
        issues.append(_issue("region_count_implausible", "warn",
                             f"{len(regions)} change regions listed",
                             "analysis.change_regions"))

    region_classes = set()
    seen_boxes: dict[tuple, int] = {}
    for i, r in enumerate(regions):
        where = f"analysis.change_regions[{i}]"
        cls = str(r.get("class") or "").strip().lower()
        if cls:
            region_classes.add(cls)
            if cls not in CHANGE_CLASSES:
                issues.append(_issue("unknown_change_class", "warn",
                                     f"'{cls}' is not a change category this "
                                     "model was trained on", where))
        size = str(r.get("size") or "").strip().lower()
        if size and size not in REGION_SIZES:
            issues.append(_issue("unknown_region_size", "warn",
                                 f"'{size}' is not a size label this model was "
                                 "trained to emit", where))
        box = r.get("bbox_normalized")
        if box is None:
            issues.append(_issue("missing_box", "warn",
                                 "change region has no drawable box", where))
            continue
        if _degenerate(box):
            issues.append(_issue("degenerate_box", "warn",
                                 "bounding box has zero width or height", where))
            continue
        key = tuple(round(float(v), 3) for v in box[:4])
        if key in seen_boxes:
            issues.append(_issue("duplicate_box", "warn",
                                 f"identical box to change_regions[{seen_boxes[key]}]",
                                 where))
        else:
            seen_boxes[key] = i

    # changed_classes should summarise the regions; a mismatch means the two
    # fields were generated independently rather than describing one analysis.
    if classes and region_classes and set(classes) != region_classes:
        issues.append(_issue("class_mismatch", "warn",
                             f"changed_classes {sorted(set(classes))} does not "
                             f"match the regions {sorted(region_classes)}",
                             "analysis.changed_classes"))
    return issues


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

REJECT_MESSAGE = {
    "single_image": ("We could not produce a reliable structured analysis for "
                     "this image. Try a clearer or higher-resolution scene."),
    "bitemporal_change": ("We could not produce a reliable change analysis for "
                          "this image pair. Check that both images cover the "
                          "same location and are correctly ordered, earlier "
                          "first."),
}
GENERIC_REJECT = ("We could not produce a reliable analysis for this input. "
                  "Please try again with a different image.")


def validate(report: dict) -> dict:
    """Check a generate_report_v2 envelope for self-consistency.

    Returns {valid, severity, issues, user_message, counts}. Never raises:
    a validator that throws is worse than the output it was meant to guard.
    """
    issues: list[dict] = []
    task = ""
    try:
        if not isinstance(report, dict):
            issues.append(_issue("not_a_report", "reject",
                                 "validator received something that is not a "
                                 "report object"))
        else:
            task = str(report.get("task") or "")
            status = report.get("status")
            if status != "ok":
                issues.append(_issue("model_error", "reject",
                                     f"the model did not return a usable "
                                     f"result ({report.get('error') or status})",
                                     "status"))
            analysis = report.get("analysis")
            if not isinstance(analysis, dict):
                issues.append(_issue("missing_analysis", "reject",
                                     "the report carries no analysis section",
                                     "analysis"))
            elif task == "single_image":
                issues += _check_single(analysis)
            elif task == "bitemporal_change":
                issues += _check_pair(analysis)
            else:
                issues.append(_issue("unknown_task", "reject",
                                     f"unrecognised task '{task}'", "task"))
    except Exception as exc:                              # noqa: BLE001
        issues.append(_issue("validator_error", "warn",
                             f"validation could not complete: {exc}"))

    n_reject = sum(1 for i in issues if i["severity"] == "reject")
    n_warn = sum(1 for i in issues if i["severity"] == "warn")
    severity = "reject" if n_reject else ("warn" if n_warn else "ok")
    return {
        "valid": severity != "reject",
        "severity": severity,
        "issues": issues,
        "counts": {"reject": n_reject, "warn": n_warn},
        "user_message": (REJECT_MESSAGE.get(task, GENERIC_REJECT)
                         if severity == "reject" else None),
    }


def summarize(v: dict) -> str:
    """One-line human summary, for logs."""
    if v["severity"] == "ok":
        return "ok"
    parts = [f"{i['severity']}:{i['code']}" for i in v["issues"]]
    return f"{v['severity']} ({v['counts']['reject']}R/{v['counts']['warn']}W) " \
           + ", ".join(parts[:6]) + (" ..." if len(parts) > 6 else "")
