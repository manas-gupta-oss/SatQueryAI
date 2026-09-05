"""
THE BOSS - the routing / function-calling node of SatQueryAI.

Qwen2.5-3B-Instruct, 4-bit quantised, used *purely* as a task router. It reads
the natural-language query plus image METADATA and emits one structured tool
call naming the specialist worker to run. It never touches pixels, never loads
an image, and knows nothing about worker internals.

Pipeline inside decide():

    query + InputBundle
        -> Qwen chat template (with native tools=[...] function definitions)
        -> constrained JSON generation (token-level schema mask)
        -> parse_tool_call_envelope()   -> BossDecision          [structure]
        -> validate_decision()          -> BossDecision          [compatibility]

CONSTRAINED GENERATION - implementation choice
----------------------------------------------
Primary backend is **lm-format-enforcer**. It is the only option that plugs
directly into plain `transformers.generate()` via the documented
`prefix_allowed_tokens_fn` hook, so it composes with a bitsandbytes 4-bit model
with no custom generation loop and no vLLM/server dependency:

    pip install lm-format-enforcer

    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn)

`outlines` is supported as a secondary backend through its logits-processor
integration, but its public API has churned across releases, so lm-format-
enforcer is the tested path.

`guided_json` was NOT used: it is a vLLM / OpenAI-server parameter and has no
meaning in a local `transformers` generate() call. Claiming otherwise would be
fiction.

If neither library is importable, generation falls back to an unconstrained
decode plus balanced-brace JSON extraction. That path is **not** schema-
guaranteed; it is for debugging only, it logs a warning, and any unparseable
output becomes a structured rejection rather than a crash or a guessed route.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from orchestration.state import (
    BossDecision,
    ExecutionStep,
    GraphState,
    ImageFormat,
    ImageMeta,
    InputBundle,
    Modality,
    ValidationCode,
    ValidationStatus,
)
from orchestration.tool_schema import (
    BOSS_TOOL_CALL_SCHEMA,
    MalformedDecision,
    build_tool_schemas,
    extract_first_json_object,
    parse_tool_call_envelope,
    registry_prompt_block,
    rejection_decision,
    validate_decision,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("configs", "boss_config.yaml")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_FALLBACK_CONFIG: Dict[str, Any] = {
    "model": {"name": "Qwen/Qwen2.5-3B-Instruct", "device_map": "auto"},
    "quantization": {"enabled": True},
    "generation": {"do_sample": False, "max_new_tokens": 320},
    "constrained_generation": {
        "backend": "auto",
        "allow_unconstrained_fallback": True,
        "use_tool_call_prefill": True,
    },
    "prompt": {"include_few_shots": True, "max_few_shots": 9,
               "include_registry_block": True},
    "routing": {"low_confidence_threshold": 0.35},
}


def load_boss_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read configs/boss_config.yaml. Falls back to safe defaults if absent."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml  # PyYAML; imported lazily so routing tests do not need it
    except ImportError:
        logger.warning("PyYAML not installed - using built-in BOSS defaults")
        return json.loads(json.dumps(_FALLBACK_CONFIG))
    if not os.path.exists(path):
        logger.warning("boss config not found at %s - using defaults", path)
        return json.loads(json.dumps(_FALLBACK_CONFIG))
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

_ROUTING_RULES = """\
ROUTING RULES (apply in order, they are not negotiable):

R1. Exactly ONE optical or multispectral image
      - a question about content            -> call_worker1 / single_vqa
      - "describe" / "what is in this"      -> call_worker1 / captioning
      - "highlight" / "locate" / "where is" -> call_worker1 / grounding
        (grounding REQUIRES target_phrase: the exact object phrase from the query)

R2. Exactly ONE SAR image
      -> call_worker2 / single_vqa.
      worker1 is tuned on optical imagery only and must never receive SAR.
      worker2 does not caption; a captioning request on a lone SAR image is
      answered as single_vqa with the request restated as the question.

R3. TWO images, one optical/multispectral + one SAR, same area
      -> call_worker2 / cross_modal_fusion.
      Set image_assignment {"optical": <id>, "sar": <id>}.
      Put requested land-cover or object classes in target_classes.

R4. TWO images of the same area at DIFFERENT times
      - a specific question ("has X increased?", "how many Y were added?")
          -> call_worker3 / change_vqa, question set, change_focus = the subject
      - an open request ("what changed?", "describe the differences")
          -> call_worker3 / change_description
      Set image_assignment {"pre": <earlier id>, "post": <later id>}.
      Set return_change_map true when the query asks WHERE the change occurred.

R5. Captioning is worker1 only. Grounding is worker1 only.
    cross_modal_fusion is worker2 only. Both change tasks are worker3 only.

VALIDATION RULES - call reject_request instead of a worker when:

V1. A change / bi-temporal task is requested but image_count != 2
      -> code image_count_mismatch
V2. cross_modal_fusion is requested but no SAR image is present, or a worker2
    task has no SAR image at all
      -> code modality_mismatch
V3. co_registered is false for any two-image task
      -> code not_co_registered
V4. same_location is false for any two-image task
      -> code not_same_location
V5. Any image format is png or jpeg while benchmark_mode is false
      -> code unsupported_format
V6. Grounding is requested but the query names no object to localise
      -> code missing_required_param
V7. image_count is 0
      -> code no_images

IMPORTANT: co_registered or same_location of None means UNKNOWN, not false.
Unknown is acceptable - proceed with the worker and record the assumption in
the assumptions list and in audit_summary. Only an explicit false is a failure.

audit_summary must state, in one or two plain sentences: the task selected, the
worker selected, the evidence in the metadata that justifies it, and any
assumption made. It is shown to the user as the execution trace."""


def build_system_prompt(config: Optional[Dict[str, Any]] = None) -> str:
    config = config or {}
    prompt_cfg = config.get("prompt", {}) or {}

    parts: List[str] = [
        "You are THE BOSS, the routing controller of SatQuery AI, an agentic "
        "remote-sensing vision-language system.",
        "",
        "You do NOT analyse imagery and you never see pixels. You read the user "
        "query and the IMAGE METADATA block, then select exactly one specialist "
        "worker to execute, or refuse the request. Specialist workers do all "
        "visual analysis.",
        "",
    ]
    if prompt_cfg.get("include_registry_block", True):
        parts += ["AVAILABLE WORKERS AND THEIR INPUT CONTRACTS:", registry_prompt_block(), ""]
    parts += [
        _ROUTING_RULES,
        "",
        "OUTPUT FORMAT: reply with exactly one JSON object inside <tool_call> "
        "tags and nothing else - no prose, no explanation, no second call. "
        "The object has these keys:",
        '  "name": one of call_worker1, call_worker2, call_worker3, reject_request',
        '  "arguments": always includes "task_type"; the task-specific parameters '
        "for a worker call, or \"code\" and \"reason\" for reject_request. On a "
        "rejection, task_type records the task the user ASKED for.",
        '  "audit_summary": one or two sentences of justification',
        '  "confidence": a number between 0 and 1',
        '  "assumptions": a list of strings, empty when nothing was assumed',
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Few-shot examples - the 5 representative queries plus 4 edge/failure cases
# --------------------------------------------------------------------------- #


def _img(
    image_id: str,
    modality: Modality,
    fmt: ImageFormat = ImageFormat.GEOTIFF,
    timestamp: Optional[str] = None,
    sensor: Optional[str] = None,
    role_hint: Optional[str] = None,
) -> ImageMeta:
    return ImageMeta(
        image_id=image_id,
        path=f"/data/{image_id}",
        modality=modality,
        format=fmt,
        georeferenced=fmt in (ImageFormat.GEOTIFF,),
        timestamp=timestamp,
        sensor=sensor,
        role_hint=role_hint,
    )


FEW_SHOTS: List[Tuple[str, InputBundle, Dict[str, Any]]] = [
    # 1 - representative query: land-cover / object description (single optical)
    (
        "Describe the land-cover and major objects visible in this image.",
        InputBundle(images=[_img("img_0", Modality.OPTICAL, sensor="Cartosat-2S")]),
        {
            "name": "call_worker1",
            "arguments": {"task_type": "captioning", "detail_level": "detailed"},
            "audit_summary": (
                "Open description request over a single optical GeoTIFF, so the task "
                "is captioning and worker1 (VRSBench-tuned single-image specialist) "
                "was selected."
            ),
            "confidence": 0.95,
            "assumptions": [],
        },
    ),
    # 2 - representative query: text-guided grounding
    (
        "Highlight the water body referred to in the query.",
        InputBundle(images=[_img("img_0", Modality.MULTISPECTRAL, sensor="Sentinel-2")]),
        {
            "name": "call_worker1",
            "arguments": {
                "task_type": "grounding",
                "target_phrase": "the water body",
                "return_visual_evidence": True,
            },
            "audit_summary": (
                "The query asks to localise a named object in one multispectral image, "
                "so text-guided grounding on worker1 was selected with target_phrase "
                "'the water body'."
            ),
            "confidence": 0.93,
            "assumptions": [],
        },
    ),
    # 3 - representative query: bi-temporal change, what AND where
    (
        "What changed between these two dates, and where did the change occur?",
        InputBundle(
            images=[
                _img("img_0", Modality.OPTICAL, timestamp="2021-03-14", role_hint="pre"),
                _img("img_1", Modality.OPTICAL, timestamp="2024-01-09", role_hint="post"),
            ],
            co_registered=True,
            same_location=True,
        ),
        {
            "name": "call_worker3",
            "arguments": {
                "task_type": "change_vqa",
                "question": "What changed between the two acquisition dates and where?",
                "return_change_map": True,
                "image_assignment": {"pre": "img_0", "post": "img_1"},
            },
            "audit_summary": (
                "Two co-registered optical images of the same area from 2021-03-14 and "
                "2024-01-09 with an explicit question, so change_vqa on worker3 was "
                "selected; a change map was requested because the query asks where."
            ),
            "confidence": 0.94,
            "assumptions": [],
        },
    ),
    # 4 - representative query: optical + SAR fusion
    (
        "Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        InputBundle(
            images=[
                _img("img_0", Modality.OPTICAL, sensor="Cartosat-2S"),
                _img("img_1", Modality.SAR, sensor="RISAT"),
            ],
            co_registered=True,
            same_location=True,
        ),
        {
            "name": "call_worker2",
            "arguments": {
                "task_type": "cross_modal_fusion",
                "target_classes": ["built-up", "water"],
                "image_assignment": {"optical": "img_0", "sar": "img_1"},
                "return_visual_evidence": True,
            },
            "audit_summary": (
                "A co-registered Cartosat-2S optical and RISAT SAR pair of the same "
                "area with an explicit joint-analysis request, so cross_modal_fusion "
                "on worker2 was selected for built-up and water classes."
            ),
            "confidence": 0.96,
            "assumptions": [],
        },
    ),
    # 5 - representative query: directional change question
    (
        "Has the built-up area increased, decreased, or remained unchanged?",
        InputBundle(
            images=[
                _img("img_0", Modality.OPTICAL, timestamp="2019-11-02"),
                _img("img_1", Modality.OPTICAL, timestamp="2023-11-20"),
            ],
            co_registered=True,
            same_location=True,
        ),
        {
            "name": "call_worker3",
            "arguments": {
                "task_type": "change_vqa",
                "question": "Has the built-up area increased, decreased, or remained unchanged?",
                "change_focus": "built-up area",
                "image_assignment": {"pre": "img_0", "post": "img_1"},
            },
            "audit_summary": (
                "A specific directional question about built-up area over a "
                "co-registered bi-temporal optical pair, so change_vqa on worker3 was "
                "selected with change_focus 'built-up area'."
            ),
            "confidence": 0.95,
            "assumptions": [],
        },
    ),
    # 6 - edge: single SAR image must go to worker2, not worker1
    (
        "How many ships can be seen in this radar scene?",
        InputBundle(images=[_img("img_0", Modality.SAR, sensor="RISAT")]),
        {
            "name": "call_worker2",
            "arguments": {
                "task_type": "single_vqa",
                "question": "How many ships are visible in this SAR scene?",
                "return_visual_evidence": True,
            },
            "audit_summary": (
                "A single SAR image was supplied, so worker2 handles it: worker1 is "
                "tuned on optical imagery and does not accept SAR. Task is single_vqa."
            ),
            "confidence": 0.88,
            "assumptions": [],
        },
    ),
    # 7 - failure: change task with only one image
    (
        "What changed between these two dates?",
        InputBundle(images=[_img("img_0", Modality.OPTICAL, timestamp="2022-06-01")]),
        {
            "name": "reject_request",
            "arguments": {
                "task_type": "change_description",
                "code": "image_count_mismatch",
                "reason": (
                    "Change analysis needs two images of the same area from different "
                    "dates, but only one image was supplied."
                ),
            },
            "audit_summary": (
                "Rejected change_description: bi-temporal analysis requires 2 images "
                "and image_count is 1, so no worker was invoked."
            ),
            "confidence": 0.97,
            "assumptions": [],
        },
    ),
    # 8 - failure: pair is explicitly not co-registered
    (
        "Use the optical and SAR images together to map flooded areas.",
        InputBundle(
            images=[
                _img("img_0", Modality.OPTICAL),
                _img("img_1", Modality.SAR),
            ],
            co_registered=False,
            same_location=True,
        ),
        {
            "name": "reject_request",
            "arguments": {
                "task_type": "cross_modal_fusion",
                "code": "not_co_registered",
                "reason": (
                    "The optical and SAR images are flagged as not co-registered; "
                    "joint pixel-level analysis needs spatially aligned inputs."
                ),
            },
            "audit_summary": (
                "Rejected cross_modal_fusion: co_registered is false, so worker2 "
                "cannot fuse the pair reliably."
            ),
            "confidence": 0.96,
            "assumptions": [],
        },
    ),
    # 9 - ambiguity resolved by assumption, not refusal
    (
        "What changed?",
        InputBundle(
            images=[
                _img("img_0", Modality.OPTICAL, timestamp="2020-02-10"),
                _img("img_1", Modality.OPTICAL, timestamp="2022-02-18"),
            ],
            co_registered=None,
            same_location=None,
        ),
        {
            "name": "call_worker3",
            "arguments": {
                "task_type": "change_description",
                "image_assignment": {"pre": "img_0", "post": "img_1"},
                "detail_level": "detailed",
            },
            "audit_summary": (
                "An open 'what changed' query over two optical images of the same area "
                "from different dates, so change_description on worker3 was selected; "
                "co-registration was not verified by the uploader and is assumed."
            ),
            "confidence": 0.72,
            "assumptions": [
                "co-registration and same-area status are unknown; assumed to hold "
                "because both images were uploaded as a temporal pair"
            ],
        },
    ),
]


def format_user_turn(query: str, bundle: InputBundle) -> str:
    """The exact user-message layout used for both few-shots and live queries."""
    return (
        f"USER QUERY:\n{query}\n\n"
        f"IMAGE METADATA:\n{bundle.to_prompt_block()}"
    )


def build_messages(
    query: str, bundle: InputBundle, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    config = config or {}
    prompt_cfg = config.get("prompt", {}) or {}
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(config)}
    ]
    if prompt_cfg.get("include_few_shots", True):
        limit = int(prompt_cfg.get("max_few_shots", len(FEW_SHOTS)))
        for shot_query, shot_bundle, envelope in FEW_SHOTS[:limit]:
            messages.append({"role": "user", "content": format_user_turn(shot_query, shot_bundle)})
            messages.append(
                {
                    "role": "assistant",
                    "content": "<tool_call>\n"
                    + json.dumps(envelope, ensure_ascii=False)
                    + "\n</tool_call>",
                }
            )
    messages.append({"role": "user", "content": format_user_turn(query, bundle)})
    return messages


# --------------------------------------------------------------------------- #
# Pure decision path - no model, no torch. Importable by the routing tests.
# --------------------------------------------------------------------------- #


def decision_from_raw(raw_output: str, bundle: InputBundle) -> BossDecision:
    """
    Turn a raw model emission into a validated BossDecision.

    Every failure mode here resolves to a well-formed rejection: a malformed
    emission must never produce a guessed route, and must never crash the graph.
    """
    text = (raw_output or "").strip()
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    text = text.replace("</tool_call>", "").strip()

    try:
        payload = json.loads(extract_first_json_object(text))
    except (MalformedDecision, json.JSONDecodeError) as exc:
        logger.warning("BOSS emitted unparseable output: %s", exc)
        return rejection_decision(
            ValidationCode.AMBIGUOUS_QUERY,
            f"the router produced an unreadable decision ({exc}); no worker was invoked",
        )

    try:
        decision = parse_tool_call_envelope(payload)
    except MalformedDecision as exc:
        logger.warning("BOSS emitted an off-contract tool call: %s", exc)
        return rejection_decision(
            ValidationCode.AMBIGUOUS_QUERY,
            f"the router produced an invalid tool call ({exc}); no worker was invoked",
        )

    # The deterministic gate. This is what makes an incompatible worker call
    # structurally impossible, whatever the 3B model believed.
    return validate_decision(decision, bundle)


# --------------------------------------------------------------------------- #
# Model-backed router
# --------------------------------------------------------------------------- #


class BossRouter:
    """
    Lazily-loaded Qwen2.5-3B router. Heavy imports (torch, transformers) happen
    inside load(), so this module stays importable for routing tests on a
    machine with no GPU and no model weights.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None):
        self.config = config if config is not None else load_boss_config(config_path)
        self.model = None
        self.tokenizer = None
        self._prefix_fn_builder = None
        self._backend_used = "none"

    # -- loading ----------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mcfg = self.config.get("model", {}) or {}
        qcfg = self.config.get("quantization", {}) or {}
        name = mcfg.get("name", "Qwen/Qwen2.5-3B-Instruct")

        kwargs: Dict[str, Any] = {
            "device_map": mcfg.get("device_map", "auto"),
            "trust_remote_code": bool(mcfg.get("trust_remote_code", False)),
        }
        for key in ("revision", "cache_dir"):
            if mcfg.get(key):
                kwargs[key] = mcfg[key]
        if mcfg.get("local_files_only"):
            kwargs["local_files_only"] = True
        if mcfg.get("attn_implementation"):
            kwargs["attn_implementation"] = mcfg["attn_implementation"]

        if qcfg.get("enabled", True):
            from transformers import BitsAndBytesConfig

            compute = getattr(
                torch, str(qcfg.get("bnb_4bit_compute_dtype", "bfloat16")), torch.float16
            )
            if compute is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.info("bfloat16 unsupported on this GPU - using float16")
                compute = torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
                bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_compute_dtype=compute,
            )
        else:
            kwargs["torch_dtype"] = "auto"

        logger.info("loading BOSS router: %s (4-bit=%s)", name, qcfg.get("enabled", True))
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=kwargs["trust_remote_code"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model.eval()
        self._init_constrained_backend()

    def _init_constrained_backend(self) -> None:
        """Pick the constrained-decoding backend. See the module docstring."""
        wanted = (self.config.get("constrained_generation", {}) or {}).get("backend", "auto")

        if wanted in ("auto", "lm-format-enforcer"):
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )

                parser = JsonSchemaParser(BOSS_TOOL_CALL_SCHEMA)
                self._prefix_fn_builder = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )
                self._backend_used = "lm-format-enforcer"
                logger.info("constrained generation: lm-format-enforcer")
                return
            except ImportError:
                if wanted == "lm-format-enforcer":
                    raise
            except Exception as exc:  # schema rejected by the parser
                logger.error("lm-format-enforcer could not build a parser: %s", exc)
                if wanted == "lm-format-enforcer":
                    raise

        if wanted in ("auto", "outlines"):
            try:
                from outlines.processors import JSONLogitsProcessor  # type: ignore

                self._outlines_processor = JSONLogitsProcessor(
                    BOSS_TOOL_CALL_SCHEMA, self.tokenizer
                )
                self._backend_used = "outlines"
                logger.info("constrained generation: outlines")
                return
            except Exception as exc:
                if wanted == "outlines":
                    raise RuntimeError(f"outlines backend unavailable: {exc}") from exc

        cg = self.config.get("constrained_generation", {}) or {}
        if not cg.get("allow_unconstrained_fallback", True):
            raise RuntimeError(
                "no constrained-generation backend available. Install one with: "
                "pip install lm-format-enforcer"
            )
        self._backend_used = "none"
        logger.warning(
            "NO constrained-generation backend installed - falling back to "
            "unconstrained decoding with JSON extraction. Output is not "
            "schema-guaranteed. Install lm-format-enforcer for the tested path."
        )

    @property
    def backend(self) -> str:
        return self._backend_used

    # -- generation -------------------------------------------------------- #

    def _build_prompt(self, query: str, bundle: InputBundle) -> str:
        messages = build_messages(query, bundle, self.config)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=build_tool_schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )
        cg = self.config.get("constrained_generation", {}) or {}
        if cg.get("use_tool_call_prefill", True):
            # Prefill Qwen's tool-call opening tag so the model stays on its
            # trained format while the constrained decoder emits only the JSON.
            prompt += "<tool_call>\n"
        return prompt

    def generate_raw(self, query: str, bundle: InputBundle) -> str:
        import torch

        self.load()
        prompt = self._build_prompt(query, bundle)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gcfg = self.config.get("generation", {}) or {}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(gcfg.get("max_new_tokens", 320)),
            "do_sample": bool(gcfg.get("do_sample", False)),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs.update(
                temperature=float(gcfg.get("temperature", 0.1)),
                top_p=float(gcfg.get("top_p", 0.9)),
                top_k=int(gcfg.get("top_k", 20)),
            )
        if gcfg.get("seed") is not None:
            torch.manual_seed(int(gcfg["seed"]))

        if self._backend_used == "lm-format-enforcer":
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_fn_builder
        elif self._backend_used == "outlines":
            from transformers import LogitsProcessorList

            gen_kwargs["logits_processor"] = LogitsProcessorList([self._outlines_processor])

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def decide(self, query: str, bundle: InputBundle) -> Tuple[BossDecision, str]:
        """Full routing pass. Returns the validated decision and the raw emission."""
        try:
            raw = self.generate_raw(query, bundle)
        except Exception as exc:  # OOM, missing weights, backend failure
            logger.exception("BOSS generation failed")
            return (
                rejection_decision(
                    ValidationCode.AMBIGUOUS_QUERY,
                    f"the router could not produce a decision ({type(exc).__name__}: {exc})",
                ),
                "",
            )
        return decision_from_raw(raw, bundle), raw


# --------------------------------------------------------------------------- #
# Singleton + LangGraph node
# --------------------------------------------------------------------------- #

_BOSS_SINGLETON: Optional[BossRouter] = None


def get_boss(config_path: Optional[str] = None, preload: bool = False) -> BossRouter:
    """Process-wide BOSS instance. The backend loads it once at startup."""
    global _BOSS_SINGLETON
    if _BOSS_SINGLETON is None:
        _BOSS_SINGLETON = BossRouter(config_path=config_path)
    if preload:
        _BOSS_SINGLETON.load()
    return _BOSS_SINGLETON


def reset_boss() -> None:
    """Drop the singleton. Used by tests and by config hot-reload."""
    global _BOSS_SINGLETON
    _BOSS_SINGLETON = None


def _coerce_bundle(raw: Any) -> InputBundle:
    if isinstance(raw, InputBundle):
        return raw
    if isinstance(raw, dict):
        return InputBundle(**raw)
    return InputBundle()


def boss_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node. Reads query + inputs, writes boss_decision, boss_raw_output
    and one trace entry. Never raises: an unusable model output becomes a
    structured rejection that the router sends down the reject path.
    """
    started = time.perf_counter()
    query = state.get("query", "") or ""
    bundle = _coerce_bundle(state.get("inputs"))

    boss = get_boss()
    decision, raw = boss.decide(query, bundle)

    elapsed = (time.perf_counter() - started) * 1000.0
    if decision.validation.status == ValidationStatus.PASS:
        detail = (
            f"task={decision.task_type.value} worker={decision.target_worker.value} "
            f"confidence={decision.confidence:.2f}"
        )
    else:
        detail = (
            f"rejected task={decision.task_type.value} "
            f"code={decision.validation.code.value}: {decision.validation.reason}"
        )

    return {
        "boss_decision": decision,
        "boss_raw_output": raw,
        "trace": [ExecutionStep(node="boss", detail=detail, latency_ms=elapsed)],
    }


def make_boss_node(decide_fn):
    """
    Build a boss node backed by any callable (query, bundle) -> BossDecision.

    Lets the graph be exercised end-to-end with a scripted router in tests and in
    CI, without downloading 3B weights. The production graph uses boss_node.
    """

    def _node(state: GraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        bundle = _coerce_bundle(state.get("inputs"))
        decision = decide_fn(state.get("query", ""), bundle)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "boss_decision": decision,
            "boss_raw_output": "",
            "trace": [
                ExecutionStep(
                    node="boss",
                    detail=(
                        f"task={decision.task_type.value} "
                        f"worker={decision.target_worker.value} "
                        f"status={decision.validation.status.value}"
                    ),
                    latency_ms=elapsed,
                )
            ],
        }

    return _node
