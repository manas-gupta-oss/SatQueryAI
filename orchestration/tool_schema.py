"""
SatQueryAI - worker registry, function-calling tool schemas, and deterministic
input-compatibility validation.

THE BOSS knows three things about a worker and nothing else:
    1. its name          (worker1 / worker2 / worker3)
    2. the tasks it owns (task_type values)
    3. its input contract (image count, modalities, co-registration, formats)

Worker internals - checkpoints, LoRA adapters, decoding params, base model - are
NOT represented here and must never leak into a routing decision.

Contents:
    WORKER_REGISTRY          - the machine-readable input contracts
    build_tool_schemas()     - OpenAI/Qwen-style `tools` list for the chat template
    BOSS_DECISION_JSON_SCHEMA- JSON schema for constrained generation
    validate_decision()      - hard, deterministic guard run *after* the model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from orchestration.state import (
    BossDecision,
    ImageFormat,
    InputBundle,
    Modality,
    TaskParams,
    TaskType,
    Validation,
    ValidationCode,
    ValidationStatus,
    WorkerId,
)

# --------------------------------------------------------------------------- #
# Worker registry - input contracts only
# --------------------------------------------------------------------------- #

OPTICAL_LIKE: Set[Modality] = {Modality.OPTICAL, Modality.MULTISPECTRAL}
GEO_FORMATS: Set[ImageFormat] = {ImageFormat.GEOTIFF, ImageFormat.TIFF}
BENCHMARK_FORMATS: Set[ImageFormat] = {ImageFormat.PNG, ImageFormat.JPEG}


@dataclass(frozen=True)
class WorkerSpec:
    """Declarative input contract for one specialist worker."""

    worker_id: WorkerId
    display_name: str
    description: str
    tasks: Set[TaskType]

    min_images: int
    max_images: int

    # Modality gates.
    allowed_modalities: Set[Modality] = field(default_factory=set)
    requires_sar: bool = False           # at least one SAR image must be present
    requires_optical: bool = False       # at least one optical/multispectral image

    # Pair gates. `False` in the bundle is a hard fail; `None` (unknown) is allowed
    # through with an assumption recorded in the audit trail.
    requires_co_registration: bool = False
    requires_same_location: bool = False
    requires_distinct_timestamps: bool = False

    tuned_on: str = ""

    def accepts(self, task: TaskType) -> bool:
        return task in self.tasks


WORKER_REGISTRY: Dict[WorkerId, WorkerSpec] = {
    WorkerId.WORKER1: WorkerSpec(
        worker_id=WorkerId.WORKER1,
        display_name="Worker 1 - Single-Image Specialist",
        description=(
            "Single optical / multispectral image understanding: visual question "
            "answering, scene captioning, and text-guided region grounding."
        ),
        tasks={TaskType.SINGLE_VQA, TaskType.CAPTIONING, TaskType.GROUNDING},
        min_images=1,
        max_images=1,
        allowed_modalities=OPTICAL_LIKE | {Modality.UNKNOWN},
        tuned_on="VRSBench",
    ),
    WorkerId.WORKER2: WorkerSpec(
        worker_id=WorkerId.WORKER2,
        display_name="Worker 2 - Cross-Modal Optical+SAR Specialist",
        description=(
            "Joint reasoning over a co-registered optical/multispectral and SAR pair, "
            "and question answering on a standalone SAR image. Extracts complementary "
            "structural (SAR) and spectral (optical) information. Does NOT produce "
            "free-form captions - route captioning to worker1."
        ),
        tasks={TaskType.CROSS_MODAL_FUSION, TaskType.SINGLE_VQA},
        min_images=1,
        max_images=2,
        allowed_modalities=OPTICAL_LIKE | {Modality.SAR, Modality.UNKNOWN},
        requires_sar=True,
        requires_co_registration=True,   # enforced only when 2 images are supplied
        requires_same_location=True,     # enforced only when 2 images are supplied
        tuned_on="BigEarthNet.txt (Sentinel-1 SAR + Sentinel-2 MSI)",
    ),
    WorkerId.WORKER3: WorkerSpec(
        worker_id=WorkerId.WORKER3,
        display_name="Worker 3 - Bi-Temporal Change Specialist",
        description=(
            "Change understanding over two spatially corresponding images of the same "
            "area acquired at different times: change-based VQA, change description, "
            "and an optional spatial change map."
        ),
        tasks={TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION},
        min_images=2,
        max_images=2,
        allowed_modalities=OPTICAL_LIKE | {Modality.SAR, Modality.UNKNOWN},
        requires_co_registration=True,
        requires_same_location=True,
        requires_distinct_timestamps=False,  # timestamps are often unavailable
        tuned_on="LEVIR-CC",
    ),
}

# A task may legitimately be served by more than one worker (e.g. VQA on a lone
# SAR scene belongs to worker2, not the VRSBench-tuned worker1).
TASK_TO_WORKERS: Dict[TaskType, List[WorkerId]] = {
    TaskType.SINGLE_VQA: [WorkerId.WORKER1, WorkerId.WORKER2],
    TaskType.CAPTIONING: [WorkerId.WORKER1],
    TaskType.GROUNDING: [WorkerId.WORKER1],
    TaskType.CROSS_MODAL_FUSION: [WorkerId.WORKER2],
    TaskType.CHANGE_VQA: [WorkerId.WORKER3],
    TaskType.CHANGE_DESCRIPTION: [WorkerId.WORKER3],
}

# Params the worker cannot run without.
REQUIRED_PARAMS: Dict[TaskType, List[str]] = {
    TaskType.SINGLE_VQA: ["question"],
    TaskType.CAPTIONING: [],
    TaskType.GROUNDING: ["target_phrase"],
    TaskType.CROSS_MODAL_FUSION: [],
    TaskType.CHANGE_VQA: ["question"],
    TaskType.CHANGE_DESCRIPTION: [],
}


# --------------------------------------------------------------------------- #
# Function-calling tool schemas (Qwen `tools=` chat-template format)
# --------------------------------------------------------------------------- #

_PARAM_DEFS: Dict[str, Dict[str, Any]] = {
    "question": {
        "type": "string",
        "description": "The user's question, normalised into a single self-contained sentence.",
    },
    "target_phrase": {
        "type": "string",
        "description": "Referring expression to localise, e.g. 'the water body on the left'.",
    },
    "target_classes": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Land-cover or object classes of interest, e.g. ['built-up', 'water'].",
    },
    "change_focus": {
        "type": "string",
        "description": "The specific phenomenon to compare across dates, e.g. 'built-up area'.",
    },
    "detail_level": {
        "type": "string",
        "enum": ["brief", "detailed"],
        "description": "Verbosity for captioning / change description.",
    },
    "return_change_map": {
        "type": "boolean",
        "description": "Request a spatial change map in addition to the textual answer.",
    },
    "return_visual_evidence": {
        "type": "boolean",
        "description": "Request boxes / masks / overlays where the worker supports them.",
    },
    "image_assignment": {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": (
            "Maps a semantic role to an image_id. worker2: {'optical': ..., 'sar': ...}. "
            "worker3: {'pre': ..., 'post': ...}. Omit for single-image calls."
        ),
    },
}

# Which params each tool exposes, and which are mandatory.
_TOOL_PARAMS: Dict[WorkerId, Dict[str, Any]] = {
    WorkerId.WORKER1: {
        "properties": [
            "question",
            "target_phrase",
            "target_classes",
            "detail_level",
            "return_visual_evidence",
        ],
        "required": [],
    },
    WorkerId.WORKER2: {
        "properties": [
            "question",
            "target_classes",
            "detail_level",
            "image_assignment",
            "return_visual_evidence",
        ],
        "required": [],
    },
    WorkerId.WORKER3: {
        "properties": [
            "question",
            "change_focus",
            "detail_level",
            "image_assignment",
            "return_change_map",
            "return_visual_evidence",
        ],
        "required": [],
    },
}


def _contract_sentence(spec: WorkerSpec) -> str:
    bits: List[str] = []
    if spec.min_images == spec.max_images:
        bits.append(f"exactly {spec.min_images} image(s)")
    else:
        bits.append(f"{spec.min_images}-{spec.max_images} images")
    if spec.requires_sar:
        bits.append("at least one SAR image")
    if spec.requires_optical:
        bits.append("at least one optical/multispectral image")
    if spec.requires_co_registration:
        bits.append("co-registered when two images are supplied")
    if spec.requires_same_location:
        bits.append("same geographic area when two images are supplied")
    return "; ".join(bits)


def build_tool_schemas() -> List[Dict[str, Any]]:
    """
    OpenAI-style tool list, passed straight to
    `tokenizer.apply_chat_template(..., tools=build_tool_schemas())`.

    One tool per worker; the sub-task is a required `task_type` enum on the call,
    so the model picks worker and task in a single structured emission.
    """
    tools: List[Dict[str, Any]] = []
    for worker_id, spec in WORKER_REGISTRY.items():
        exposed = _TOOL_PARAMS[worker_id]
        props: Dict[str, Any] = {
            "task_type": {
                "type": "string",
                "enum": sorted(t.value for t in spec.tasks),
                "description": "Which of this worker's sub-tasks to run.",
            }
        }
        for name in exposed["properties"]:
            props[name] = _PARAM_DEFS[name]

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"call_{worker_id.value}",
                    "description": (
                        f"{spec.display_name}. {spec.description} "
                        f"Fine-tuned on {spec.tuned_on}. "
                        f"INPUT CONTRACT: {_contract_sentence(spec)}."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": ["task_type"] + list(exposed["required"]),
                    },
                },
            }
        )

    # A fourth pseudo-tool so refusal is a first-class, structured action rather
    # than free text the parser has to guess at.
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "reject_request",
                "description": (
                    "Call this INSTEAD of a worker when the supplied images cannot "
                    "support the requested task (wrong image count, missing SAR or "
                    "optical modality, images not co-registered or not the same area, "
                    "unsupported file format, or a required parameter is unavailable)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "enum": [c.value for c in ValidationCode if c != ValidationCode.OK],
                            "description": "Machine-readable failure reason.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One sentence the end user can act on.",
                        },
                    },
                    "required": ["code", "reason"],
                },
            },
        }
    )
    return tools


def registry_prompt_block() -> str:
    """Human-readable contract table injected into the BOSS system prompt."""
    lines: List[str] = []
    for spec in WORKER_REGISTRY.values():
        lines.append(
            f"- {spec.worker_id.value} ({spec.display_name}, tuned on {spec.tuned_on})\n"
            f"    tasks: {', '.join(sorted(t.value for t in spec.tasks))}\n"
            f"    input contract: {_contract_sentence(spec)}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Constrained-generation schema
# --------------------------------------------------------------------------- #

BOSS_DECISION_JSON_SCHEMA: Dict[str, Any] = BossDecision.model_json_schema()


# --------------------------------------------------------------------------- #
# Deterministic validation - the hard guard behind the model
# --------------------------------------------------------------------------- #


def _format_ok(bundle: InputBundle) -> Optional[str]:
    for img in bundle.images:
        if img.format in GEO_FORMATS:
            continue
        if img.format in BENCHMARK_FORMATS and bundle.benchmark_mode:
            continue
        return (
            f"image '{img.image_id}' has format '{img.format.value}', which is not "
            "accepted (GeoTIFF/TIFF always; PNG/JPEG only for benchmark inputs)"
        )
    return None


def _missing_params(task: TaskType, params: TaskParams) -> List[str]:
    missing: List[str] = []
    for name in REQUIRED_PARAMS.get(task, []):
        value = getattr(params, name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    return missing


def check_compatibility(
    task: TaskType, worker: WorkerId, bundle: InputBundle, params: TaskParams
) -> Validation:
    """
    Pure, model-free compatibility check. Returns PASS or the first violated rule.

    `None` (unknown) for co_registered / same_location is *not* a failure - the
    caller records an assumption instead. Only an explicit `False` fails.
    """
    if worker == WorkerId.NONE:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.UNSUPPORTED_TASK_FOR_WORKER,
            reason="no worker was selected",
        )

    spec = WORKER_REGISTRY.get(worker)
    if spec is None:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.UNSUPPORTED_TASK_FOR_WORKER,
            reason=f"unknown worker '{worker}'",
        )

    if not spec.accepts(task):
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.UNSUPPORTED_TASK_FOR_WORKER,
            reason=f"{worker.value} does not implement task '{task.value}'",
        )

    n = bundle.image_count
    if n == 0:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.NO_IMAGES,
            reason="no image was supplied; every task requires at least one image",
        )

    if n < spec.min_images or n > spec.max_images:
        expected = (
            f"exactly {spec.min_images}"
            if spec.min_images == spec.max_images
            else f"{spec.min_images}-{spec.max_images}"
        )
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.IMAGE_COUNT_MISMATCH,
            reason=(
                f"task '{task.value}' needs {expected} image(s) but {n} "
                f"{'was' if n == 1 else 'were'} supplied"
            ),
        )

    fmt_problem = _format_ok(bundle)
    if fmt_problem:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.UNSUPPORTED_FORMAT,
            reason=fmt_problem,
        )

    mods = set(bundle.modalities)
    if spec.allowed_modalities and not mods.issubset(spec.allowed_modalities):
        bad = sorted(m.value for m in mods - spec.allowed_modalities)
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.MODALITY_MISMATCH,
            reason=f"{worker.value} cannot accept modality/modalities {bad}",
        )

    if spec.requires_sar and Modality.SAR not in mods:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.MODALITY_MISMATCH,
            reason=(
                f"{worker.value} requires a SAR image; supplied modalities are "
                f"{sorted(m.value for m in mods)}"
            ),
        )

    if spec.requires_optical and not (mods & OPTICAL_LIKE):
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.MODALITY_MISMATCH,
            reason=(
                f"{worker.value} requires an optical/multispectral image; supplied "
                f"modalities are {sorted(m.value for m in mods)}"
            ),
        )

    # Pair-level gates only bite once there is actually a pair.
    if n >= 2:
        if spec.requires_same_location and bundle.same_location is False:
            return Validation(
                status=ValidationStatus.FAIL,
                code=ValidationCode.NOT_SAME_LOCATION,
                reason="the two images do not cover the same geographic area",
            )
        if spec.requires_co_registration and bundle.co_registered is False:
            return Validation(
                status=ValidationStatus.FAIL,
                code=ValidationCode.NOT_CO_REGISTERED,
                reason=(
                    "the image pair is not co-registered; pair analysis requires "
                    "spatially aligned inputs"
                ),
            )
        if spec.requires_distinct_timestamps:
            stamps = [i.timestamp for i in bundle.images if i.timestamp]
            if len(stamps) == 2 and stamps[0] == stamps[1]:
                return Validation(
                    status=ValidationStatus.FAIL,
                    code=ValidationCode.IMAGE_COUNT_MISMATCH,
                    reason="change analysis needs two different acquisition dates",
                )

    missing = _missing_params(task, params)
    if missing:
        return Validation(
            status=ValidationStatus.FAIL,
            code=ValidationCode.MISSING_REQUIRED_PARAM,
            reason=f"task '{task.value}' requires parameter(s): {', '.join(missing)}",
        )

    return Validation(status=ValidationStatus.PASS, code=ValidationCode.OK, reason="")


def infer_image_assignment(worker: WorkerId, bundle: InputBundle) -> Dict[str, str]:
    """Fill in role -> image_id when the model left it empty."""
    imgs = bundle.images
    if worker == WorkerId.WORKER2 and len(imgs) == 2:
        sar = next((i for i in imgs if i.modality == Modality.SAR), None)
        opt = next((i for i in imgs if i.modality in OPTICAL_LIKE), None)
        if sar and opt:
            return {"optical": opt.image_id, "sar": sar.image_id}
    if worker == WorkerId.WORKER3 and len(imgs) == 2:
        a, b = imgs
        # Prefer explicit role hints, then timestamps, then upload order.
        hints = {(i.role_hint or "").lower(): i for i in imgs}
        if "pre" in hints and "post" in hints:
            return {"pre": hints["pre"].image_id, "post": hints["post"].image_id}
        if a.timestamp and b.timestamp and a.timestamp > b.timestamp:
            a, b = b, a
        return {"pre": a.image_id, "post": b.image_id}
    return {}


def validate_decision(decision: BossDecision, bundle: InputBundle) -> BossDecision:
    """
    Post-process a raw model decision:

      1. re-run compatibility deterministically - a hard failure always wins over
         the model's optimism, so an incompatible worker is never executed;
      2. force target_worker to NONE whenever validation fails;
      3. auto-fill image_assignment;
      4. record an assumption when co-registration / location is merely unknown.

    A model-emitted FAIL is preserved even when the deterministic check passes:
    the model may have caught something semantic the rules cannot see.
    """
    result = decision.model_copy(deep=True)

    # An explicit refusal is already terminal. Re-running compatibility against
    # WorkerId.NONE would only replace the specific reason the router gave with a
    # generic "no worker was selected", which is worse for the user and for the
    # audit trail.
    if (
        result.target_worker == WorkerId.NONE
        or result.validation.status == ValidationStatus.FAIL
    ):
        result.target_worker = WorkerId.NONE
        return result

    verdict = check_compatibility(
        result.task_type, result.target_worker, bundle, result.params
    )

    if verdict.status == ValidationStatus.FAIL:
        result.validation = verdict
        result.target_worker = WorkerId.NONE
        if verdict.reason and verdict.reason not in result.audit_summary:
            result.audit_summary = (
                f"Rejected: {verdict.reason}. (Router had proposed "
                f"'{decision.task_type.value}' on {decision.target_worker.value}.)"
            )
        return result

    if not result.image_assignment:
        result.image_assignment = infer_image_assignment(result.target_worker, bundle)

    spec = WORKER_REGISTRY[result.target_worker]
    if bundle.image_count >= 2:
        if spec.requires_co_registration and bundle.co_registered is None:
            result.assumptions.append(
                "co-registration was not verified by the uploader; assumed aligned"
            )
        if spec.requires_same_location and bundle.same_location is None:
            result.assumptions.append(
                "same-area status was not verified by the uploader; assumed same AOI"
            )
    return result


# --------------------------------------------------------------------------- #
# Constrained-generation schema for the tool call THE BOSS emits
# --------------------------------------------------------------------------- #
#
# Why a hand-written schema instead of BossDecision.model_json_schema():
#   * pydantic emits $ref/$defs, which several constrained-decoding backends
#     handle poorly; this one is fully dereferenced.
#   * it mirrors Qwen's native tool-call shape ({"name", "arguments"}) so the
#     model stays on the format it was trained on, with three extra audit fields
#     the problem statement requires (audit_summary / confidence / assumptions).
#   * image_assignment uses a CLOSED set of role keys instead of an open
#     additionalProperties map, which every backend can enforce.

ROLE_KEYS: List[str] = ["optical", "sar", "pre", "post", "image"]

_ALL_TASK_VALUES: List[str] = [t.value for t in TaskType]
_TOOL_NAMES: List[str] = [f"call_{w.value}" for w in WORKER_REGISTRY] + ["reject_request"]
_REJECT_CODES: List[str] = [c.value for c in ValidationCode if c != ValidationCode.OK]

BOSS_TOOL_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "BossToolCall",
    "properties": {
        "name": {
            "type": "string",
            "enum": _TOOL_NAMES,
            "description": "The tool to invoke.",
        },
        "arguments": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": _ALL_TASK_VALUES,
                    "description": (
                        "The task the user asked for. Always present - on "
                        "reject_request it records what was requested but refused."
                    ),
                },
                "question": {"type": "string"},
                "target_phrase": {"type": "string"},
                "target_classes": {"type": "array", "items": {"type": "string"}},
                "change_focus": {"type": "string"},
                "detail_level": {"type": "string", "enum": ["brief", "detailed"]},
                "return_change_map": {"type": "boolean"},
                "return_visual_evidence": {"type": "boolean"},
                "image_assignment": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in ROLE_KEYS},
                    "additionalProperties": False,
                },
                "code": {
                    "type": "string",
                    "enum": _REJECT_CODES,
                    "description": "reject_request only.",
                },
                "reason": {"type": "string", "description": "reject_request only."},
            },
            "required": ["task_type"],
        },
        "audit_summary": {"type": "string"},
        "confidence": {"type": "number"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "arguments", "audit_summary", "confidence"],
}

REJECT_TOOL_NAME = "reject_request"

# Which TaskParams fields each task is allowed to carry through to the worker.
# Anything else the model emits is dropped, so workers get a clean payload.
_ALLOWED_PARAMS: Dict[TaskType, Set[str]] = {
    TaskType.SINGLE_VQA: {"question", "target_classes", "return_visual_evidence"},
    TaskType.CAPTIONING: {"detail_level", "return_visual_evidence"},
    TaskType.GROUNDING: {"target_phrase", "return_visual_evidence"},
    TaskType.CROSS_MODAL_FUSION: {
        "question",
        "target_classes",
        "detail_level",
        "return_visual_evidence",
    },
    TaskType.CHANGE_VQA: {
        "question",
        "change_focus",
        "return_change_map",
        "return_visual_evidence",
    },
    TaskType.CHANGE_DESCRIPTION: {
        "change_focus",
        "detail_level",
        "return_change_map",
        "return_visual_evidence",
    },
}


class MalformedDecision(ValueError):
    """Raised when a model emission cannot be mapped onto the contract at all."""


def extract_first_json_object(text: str) -> str:
    """
    Pull the first balanced top-level JSON object out of free text.

    Needed only on the unconstrained fallback path; constrained decoding emits a
    bare object. String-aware, so braces inside strings do not confuse it.
    """
    start = text.find("{")
    if start == -1:
        raise MalformedDecision("no JSON object found in model output")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise MalformedDecision("unterminated JSON object in model output")


def _worker_from_tool_name(name: str) -> WorkerId:
    if name == REJECT_TOOL_NAME:
        return WorkerId.NONE
    key = name[len("call_"):] if name.startswith("call_") else name
    try:
        worker = WorkerId(key)
    except ValueError as exc:
        raise MalformedDecision(f"unknown tool name '{name}'") from exc
    if worker not in WORKER_REGISTRY:
        raise MalformedDecision(f"tool '{name}' is not a registered worker")
    return worker


def _build_params(task: TaskType, args: Dict[str, Any]) -> TaskParams:
    allowed = _ALLOWED_PARAMS.get(task, set())
    payload = {k: v for k, v in args.items() if k in allowed and v is not None}
    try:
        return TaskParams(**payload)
    except Exception:
        # A single bad field must not sink the whole decision; validation will
        # then report any genuinely missing required parameter.
        return TaskParams()


def parse_tool_call_envelope(payload: Dict[str, Any]) -> BossDecision:
    """
    Map a raw tool-call envelope onto a BossDecision. Structure only: no
    compatibility checking happens here, that is validate_decision's job.
    """
    if not isinstance(payload, dict):
        raise MalformedDecision("model output is not a JSON object")

    name = payload.get("name")
    if not isinstance(name, str):
        raise MalformedDecision("missing 'name' in tool call")
    worker = _worker_from_tool_name(name)

    args = payload.get("arguments")
    if not isinstance(args, dict):
        raise MalformedDecision("missing or non-object 'arguments' in tool call")

    raw_task = args.get("task_type")
    try:
        task = TaskType(raw_task)
    except ValueError as exc:
        raise MalformedDecision(f"invalid task_type '{raw_task}'") from exc

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    assumptions = payload.get("assumptions") or []
    if not isinstance(assumptions, list):
        assumptions = []
    assumptions = [str(a) for a in assumptions]

    audit = str(payload.get("audit_summary") or "")

    if worker == WorkerId.NONE:
        try:
            code = ValidationCode(args.get("code"))
        except ValueError:
            code = ValidationCode.AMBIGUOUS_QUERY
        reason = str(args.get("reason") or "the router refused the request")
        return BossDecision(
            task_type=task,
            target_worker=WorkerId.NONE,
            params=TaskParams(),
            validation=Validation(
                status=ValidationStatus.FAIL, code=code, reason=reason
            ),
            audit_summary=audit or f"Rejected '{task.value}': {reason}",
            confidence=confidence,
            assumptions=assumptions,
        )

    assignment = payload.get("arguments", {}).get("image_assignment") or {}
    if not isinstance(assignment, dict):
        assignment = {}
    assignment = {str(k): str(v) for k, v in assignment.items() if v}

    return BossDecision(
        task_type=task,
        target_worker=worker,
        params=_build_params(task, args),
        validation=Validation(status=ValidationStatus.PASS, code=ValidationCode.OK),
        audit_summary=audit,
        confidence=confidence,
        assumptions=assumptions,
        image_assignment=assignment,
    )


def rejection_decision(
    code: ValidationCode,
    reason: str,
    task: TaskType = TaskType.SINGLE_VQA,
    audit_summary: str = "",
) -> BossDecision:
    """Build a safe, well-formed refusal, used when model output is unusable."""
    return BossDecision(
        task_type=task,
        target_worker=WorkerId.NONE,
        params=TaskParams(),
        validation=Validation(status=ValidationStatus.FAIL, code=code, reason=reason),
        audit_summary=audit_summary or f"Rejected before worker dispatch: {reason}",
        confidence=0.0,
    )
