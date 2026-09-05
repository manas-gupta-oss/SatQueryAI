"""
SatQueryAI - LangGraph state schema and BOSS routing contract.

This module is the single source of truth for everything that flows through the
graph. It is deliberately free of model / worker / backend implementation
details: BOSS only ever sees *metadata* about images, never pixels.

Layering:
    - Enums + ImageMeta + InputBundle : what the backend hands the graph
    - BossDecision                    : what THE BOSS emits (constrained JSON)
    - WorkerOutput                    : what a worker node hands back
    - GraphState                      : the LangGraph TypedDict tying it together
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums - the closed vocabularies the router is allowed to emit
# --------------------------------------------------------------------------- #


class TaskType(str, Enum):
    """The six tasks mandated by the problem statement."""

    SINGLE_VQA = "single_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CROSS_MODAL_FUSION = "cross_modal_fusion"
    CHANGE_VQA = "change_vqa"
    CHANGE_DESCRIPTION = "change_description"


class WorkerId(str, Enum):
    """Specialist registry keys. BOSS knows these names + input contracts only."""

    WORKER1 = "worker1"  # single-image (VRSBench-tuned)
    WORKER2 = "worker2"  # cross-modal optical+SAR (BigEarthNet.txt-tuned)
    WORKER3 = "worker3"  # bi-temporal change (CDVQA-tuned)
    NONE = "none"        # validation failed -> no worker is executed


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    UNSUPPORTED = "unsupported"


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class ValidationCode(str, Enum):
    """Machine-readable reasons for a rejected query/input combination."""

    OK = "ok"
    NO_IMAGES = "no_images"
    IMAGE_COUNT_MISMATCH = "image_count_mismatch"        # e.g. change VQA with 1 image
    MODALITY_MISMATCH = "modality_mismatch"              # e.g. fusion without a SAR image
    NOT_CO_REGISTERED = "not_co_registered"              # pair analysis on unaligned inputs
    NOT_SAME_LOCATION = "not_same_location"              # pair from different AOIs
    UNSUPPORTED_FORMAT = "unsupported_format"            # e.g. PNG outside benchmark mode
    MISSING_REQUIRED_PARAM = "missing_required_param"    # e.g. grounding with no target phrase
    UNSUPPORTED_TASK_FOR_WORKER = "unsupported_task_for_worker"
    AMBIGUOUS_QUERY = "ambiguous_query"                  # resolved via assumption, not fatal


# --------------------------------------------------------------------------- #
# Inputs - produced by backend/routes/upload.py, consumed by BOSS
# --------------------------------------------------------------------------- #


class ImageMeta(BaseModel):
    """Metadata for one uploaded image. BOSS reasons over this, never the pixels."""

    image_id: str = Field(..., description="Stable id returned by /api/upload")
    path: str = Field("", description="Server-side path or URI the worker will read")
    modality: Modality = Modality.UNKNOWN
    format: ImageFormat = ImageFormat.UNSUPPORTED

    # Optional geospatial / temporal metadata used for compatibility checks.
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    crs: Optional[str] = Field(None, description="e.g. 'EPSG:32643'; None if not georeferenced")
    georeferenced: bool = False
    timestamp: Optional[str] = Field(None, description="ISO-8601 acquisition time if known")
    sensor: Optional[str] = Field(None, description="e.g. 'Sentinel-2', 'Cartosat-2S', 'RISAT'")

    # Free-form hint from the uploader UI ("pre", "post", "optical", "sar", ...).
    role_hint: Optional[str] = None


class InputBundle(BaseModel):
    """The full image-side context for one query."""

    images: List[ImageMeta] = Field(default_factory=list)
    co_registered: Optional[bool] = Field(
        None, description="True if the pair is pixel-aligned. None = unknown/unchecked."
    )
    same_location: Optional[bool] = Field(
        None, description="True if images cover the same AOI. None = unknown/unchecked."
    )
    benchmark_mode: bool = Field(
        False,
        description=(
            "True when inputs come from a public benchmark (VRSBench/RSVQA/CDVQA); "
            "only then are PNG/JPEG accepted."
        ),
    )

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def modalities(self) -> List[Modality]:
        return [img.modality for img in self.images]

    def to_prompt_block(self) -> str:
        """Compact, deterministic rendering of the metadata for the BOSS prompt."""
        lines = [
            f"image_count: {self.image_count}",
            f"co_registered: {self.co_registered}",
            f"same_location: {self.same_location}",
            f"benchmark_mode: {self.benchmark_mode}",
        ]
        for i, img in enumerate(self.images):
            lines.append(
                f"image[{i}]: id={img.image_id}, modality={img.modality.value}, "
                f"format={img.format.value}, georeferenced={img.georeferenced}, "
                f"timestamp={img.timestamp}, sensor={img.sensor}, "
                f"role_hint={img.role_hint}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# BOSS output - this is the constrained-JSON contract
# --------------------------------------------------------------------------- #


class TaskParams(BaseModel):
    """
    Task-specific arguments. Every field is optional; the registry declares which
    ones are *required* per task, and validation enforces that.

    Nothing here may configure model internals (checkpoints, decoding, LoRA
    paths) - the problem statement only permits task parameters.
    """

    question: Optional[str] = Field(
        None, description="Normalised question text for VQA tasks."
    )
    target_phrase: Optional[str] = Field(
        None, description="Referring expression to localise, for grounding."
    )
    target_classes: Optional[List[str]] = Field(
        None, description="Land-cover/object classes of interest, e.g. ['built-up', 'water']."
    )
    change_focus: Optional[str] = Field(
        None, description="What change to reason about, e.g. 'built-up area'."
    )
    detail_level: Optional[str] = Field(
        None, description="'brief' | 'detailed' for captioning / change description."
    )
    return_change_map: bool = Field(
        False, description="Ask worker3 for a spatial change map alongside text."
    )
    return_visual_evidence: bool = Field(
        True, description="Ask the worker for boxes/masks/overlays where supported."
    )


class Validation(BaseModel):
    status: ValidationStatus
    code: ValidationCode = ValidationCode.OK
    reason: str = Field(
        "", description="Human-readable explanation; required when status == fail."
    )


class BossDecision(BaseModel):
    """
    The single structured object THE BOSS must emit. Generation is constrained to
    this schema (outlines / guided_json), so it always parses.
    """

    task_type: TaskType
    target_worker: WorkerId
    params: TaskParams = Field(default_factory=TaskParams)
    validation: Validation
    audit_summary: str = Field(
        ...,
        description=(
            "One or two sentences: task chosen, worker chosen, why, and any assumption "
            "made. This is the graded execution trace."
        ),
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Router's confidence in the task/worker choice."
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit assumptions made to resolve an ambiguous query.",
    )
    image_assignment: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Semantic role -> image_id, e.g. {'optical': 'img_0', 'sar': 'img_1'} or "
            "{'pre': 'img_0', 'post': 'img_1'}. Empty for single-image tasks."
        ),
    )

    @property
    def is_executable(self) -> bool:
        return (
            self.validation.status == ValidationStatus.PASS
            and self.target_worker != WorkerId.NONE
        )


# --------------------------------------------------------------------------- #
# Worker output - the contract workers must satisfy (they own the internals)
# --------------------------------------------------------------------------- #


class WorkerOutput(BaseModel):
    worker: WorkerId
    task_type: TaskType
    answer: str = Field("", description="Primary textual result.")
    confidence: Optional[float] = None

    # Visual evidence. Boxes are normalised [0,1] xyxy.
    boxes: List[List[float]] = Field(default_factory=list)
    box_labels: List[str] = Field(default_factory=list)
    mask_path: Optional[str] = Field(None, description="Change map / segmentation overlay.")
    overlay_path: Optional[str] = None

    extras: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ExecutionStep(BaseModel):
    """One line of the auditable execution trace shown in the UI / report."""

    node: str
    detail: str
    latency_ms: Optional[float] = None


def append_trace(
    left: Optional[List[ExecutionStep]], right: Optional[List[ExecutionStep]]
) -> List[ExecutionStep]:
    """LangGraph reducer: trace entries accumulate instead of overwriting."""
    return (left or []) + (right or [])


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #


class GraphState(TypedDict, total=False):
    """What flows through the LangGraph pipeline."""

    # --- inputs ---
    query: str
    inputs: InputBundle
    request_id: str

    # --- BOSS ---
    boss_decision: Optional[BossDecision]
    boss_raw_output: Optional[str]        # unparsed model text, kept for audit

    # --- worker ---
    worker_output: Optional[WorkerOutput]

    # --- terminal ---
    final_response: Optional[Dict[str, Any]]
    error: Optional[str]

    # --- audit ---
    trace: Annotated[List[ExecutionStep], append_trace]


def initial_state(query: str, inputs: InputBundle, request_id: str = "") -> GraphState:
    return GraphState(
        query=query,
        inputs=inputs,
        request_id=request_id,
        boss_decision=None,
        boss_raw_output=None,
        worker_output=None,
        final_response=None,
        error=None,
        trace=[],
    )
