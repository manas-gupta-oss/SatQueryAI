"""Shared machinery for QLoRA fine-tuning Qwen2.5-VL-3B on VRSBench under a 6 GB budget.

Everything numerically load-bearing lives here so `warmup.py`, `train.py` and
`compare.py` cannot drift apart:

* target compaction   -- the change that makes 6 GB viable at all
* completion-only labels with an explicit zero-supervised-token guard
* NaN detection helpers used by every stage
"""

from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "vrsbench"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "Describe the image in detail."
SEED = 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Sized for 6 GB. See `docs` in README-FINETUNE.md for the budget table."""

    model_id: str = MODEL_ID
    train_jsonl: Path = DATA_DIR / "train.jsonl"
    val_jsonl: Path = DATA_DIR / "val.jsonl"
    out_dir: Path = PROJECT_ROOT / "adapters" / "qwen25vl-vrsbench-qlora"

    # 512x512 -> 504x504 -> 324 vision tokens. Native res; satellite detail matters.
    max_pixels: int = 324 * 28 * 28
    min_pixels: int = 144 * 28 * 28

    # p95 of the compacted target is ~362 tok, +324 vision +30 prompt = ~716.
    # 1024 covers essentially the whole dataset, so nothing gets starved to
    # zero supervised tokens (the 0/0 -> NaN trap).
    max_len: int = 1024

    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-4
    warmup_steps: int = 20
    max_steps: int = 400          # ~1600 samples; enough for format adherence
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    eval_samples: int = 32

    # NaN policy
    nan_check_every: int = 25     # full adapter-weight audit cadence
    max_consecutive_skips: int = 12


CFG = Config()


# --------------------------------------------------------------------------
# memory helpers
# --------------------------------------------------------------------------

def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def vram_now_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def gpu_report() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible.")
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "capability": f"sm_{p.major}{p.minor}",
        "vram_gb": p.total_memory / 1024**3,
        "bf16": torch.cuda.is_bf16_supported(),
    }


# --------------------------------------------------------------------------
# target compaction
# --------------------------------------------------------------------------

# obj_corner is the same box as obj_coord, written as 4 polygon corners instead
# of xyxy -- 8 floats of pure redundancy. Dropping it plus the always-derivable
# flags and switching off indent=4 pretty-printing takes the p95 target from
# 1297 tokens to 362, which is what lets this fit in 6 GB at all.
DROP_OBJECT_KEYS = ("obj_corner", "flag", "is_unique", "obj_rel_position", "obj_rel_size")


def compact_target(answer: str) -> str:
    """VRSBench answer JSON -> minimal, information-equivalent, compact JSON.

    Falls back to the raw string if it does not parse (it always does for this
    dataset, but a silent crash mid-epoch is worse than a slightly long sample).
    """
    try:
        d = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return answer.strip()

    d.pop("image", None)
    for obj in d.get("objects") or []:
        if isinstance(obj, dict):
            for k in DROP_OBJECT_KEYS:
                obj.pop(k, None)
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def load_split(jsonl_path: str | Path, compact: bool = True) -> list[dict]:
    """Load a VRSBench jsonl. Image paths in the file are absolute and local."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {jsonl_path}")

    rows: list[dict] = []
    dropped = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            img = Path(r["image"])
            if not img.exists():                       # tolerate a moved dataset
                img = jsonl_path.parent / f"Images_{jsonl_path.stem}" / Path(r["image"]).name
            if not img.exists():
                dropped += 1
                continue

            answer = (r.get("answer") or "").strip()
            if not answer:
                dropped += 1
                continue

            rows.append({
                "image": str(img),
                "question": (r.get("question") or PROMPT).strip(),
                "answer": compact_target(answer) if compact else answer,
            })

    if dropped:
        print(f"  [{jsonl_path.name}] dropped {dropped} unusable rows")
    return rows


# --------------------------------------------------------------------------
# processor + collator
# --------------------------------------------------------------------------

def build_processor(cfg: Config = CFG):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        padding_side="right",
    )
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    return proc


class Collator:
    """Builds model kwargs with completion-only labels.

    Fixes three defects in the original `labels = input_ids.clone()` approach:
      1. pad tokens were supervised
      2. `<|image_pad|>` tokens were supervised -- the model was being asked to
         predict image placeholders, which yields large meaningless gradients
      3. the prompt was supervised

    And it *reports* `empty` -- rows where truncation left zero supervised
    tokens. Feeding one of those to cross-entropy computes 0/0 = NaN, which is
    the most likely cause of the original "NaN before iteration 10".
    """

    def __init__(self, processor, max_len: int = CFG.max_len):
        self.p = processor
        self.tok = processor.tokenizer
        self.max_len = max_len
        self.assistant_header = self.tok(
            "<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

        self.vision_ids: set[int] = set()
        for t in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                  "<|vision_end|>", "<|vision_pad|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self.vision_ids.add(i)

    @staticmethod
    def _rfind_sub(seq: list[int], sub: list[int]) -> int:
        """Index just past the LAST occurrence of `sub` in `seq`; -1 if absent."""
        n, m = len(seq), len(sub)
        for i in range(n - m, -1, -1):
            if seq[i:i + m] == sub:
                return i + m
        return -1

    def __call__(self, rows: list[dict]) -> tuple[dict, dict]:
        texts, images = [], []
        for r in rows:
            msgs = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": r["question"]},
            ]}]
            prompt = self.p.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            texts.append(prompt + r["answer"] + "<|im_end|>\n")
            with Image.open(r["image"]) as im:      # close the handle eagerly
                images.append(im.convert("RGB"))

        try:
            batch = self.p(
                text=texts, images=images, return_tensors="pt",
                padding=True, truncation=True, max_length=self.max_len,
            )
        except (TypeError, ValueError):
            # Some processor versions reject truncation kwargs. Cut from the
            # right afterwards -- safe, because the ~324 vision tokens sit at
            # the front and survive intact.
            batch = self.p(text=texts, images=images, return_tensors="pt", padding=True)
            for k in ("input_ids", "attention_mask"):
                if k in batch and batch[k].shape[1] > self.max_len:
                    batch[k] = batch[k][:, : self.max_len]

        ids = batch["input_ids"]
        attn = batch.get("attention_mask")
        labels = ids.clone()

        supervised: list[int] = []
        for b in range(ids.size(0)):
            cut = self._rfind_sub(ids[b].tolist(), self.assistant_header)
            if cut < 0:                              # header truncated away
                cut = ids.size(1)
            labels[b, :cut] = -100
            if attn is not None:
                labels[b][attn[b] == 0] = -100       # padding
            for vid in self.vision_ids:
                labels[b][ids[b] == vid] = -100      # image placeholders
            labels[b][ids[b] == self.tok.pad_token_id] = -100
            supervised.append(int((labels[b] != -100).sum()))

        batch["labels"] = labels
        stats = {
            "seq_len": int(ids.shape[1]),
            "supervised": supervised,
            "empty": sum(1 for s in supervised if s == 0),
            "n_vision_tok": int(sum((ids == i).sum() for i in self.vision_ids)),
        }
        return dict(batch), stats


def to_device(batch: dict, device: str = "cuda") -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

@dataclass
class Mode:
    quant: str | None
    compute: torch.dtype
    autocast: torch.dtype | None
    vis_fp32: bool = False
    skip_vis_quant: bool = False

    @property
    def needs_scaler(self) -> bool:
        return self.autocast == torch.float16


def available_modes() -> dict[str, Mode]:
    """Modes that can actually load inside 6 GB, best-first.

    Unquantized fp16/bf16 needs 7.5 GB of weights alone, so it is excluded --
    on this card NF4 is the only base-weight format that fits.
    """
    bf16_ok = torch.cuda.is_bf16_supported()
    modes: dict[str, Mode] = {}
    if bf16_ok:
        # sm_89 (Ada) has native bf16 -- this sidesteps the fp16 ViT-overflow
        # failure mode entirely. It is the expected winner on an RTX 4050.
        modes["nf4_bf16"] = Mode("nf4", torch.bfloat16, torch.bfloat16)
    modes["nf4_fp16_visfp32"] = Mode("nf4", torch.float16, torch.float16,
                                     vis_fp32=True, skip_vis_quant=True)
    modes["nf4_fp16"] = Mode("nf4", torch.float16, torch.float16)
    modes["nf4_fp32"] = Mode("nf4", torch.float32, None)
    return modes


MODE_PREFERENCE = ["nf4_bf16", "nf4_fp32", "nf4_fp16_visfp32", "nf4_fp16"]


def autocast_ctx(mode: Mode):
    if mode.autocast is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=mode.autocast)


def _get_visual(model):
    m = model
    while hasattr(m, "base_model"):                  # unwrap PEFT
        m = m.base_model
    while hasattr(m, "model") and not hasattr(m, "visual"):
        m = m.model
    if hasattr(m, "visual"):
        return m.visual
    raise AttributeError("could not locate the vision tower")


def _force_visual_fp32(model, out_dtype: torch.dtype):
    """Run the ViT in true fp32 with autocast off, then cast its output back.

    Only relevant on GPUs without bf16; kept so the same code can run on a T4.
    """
    vis = _get_visual(model)
    vis.to(torch.float32)
    orig = vis.forward

    def fwd(*a, **k):
        a = [x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in a]
        k = {kk: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for kk, v in k.items()}
        with torch.autocast("cuda", enabled=False):
            out = orig(*a, **k)
        return out.to(out_dtype) if torch.is_tensor(out) else out

    vis.forward = fwd
    return model


def lora_target_names(model, include_vision: bool = False) -> list[str]:
    """Explicit module paths so LoRA lands on the language model only.

    Qwen2.5-VL's ViT reuses the names gate_proj/up_proj/down_proj, so PEFT's
    default suffix matching would silently attach adapters to the vision tower.
    """
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"}
    names = []
    for n, m in model.named_modules():
        if type(m).__name__ not in ("Linear", "Linear4bit", "Linear8bitLt"):
            continue
        if "lm_head" in n:
            continue
        if not include_vision and "visual" in n:
            continue
        if n.split(".")[-1] in suffixes:
            names.append(n)
    return names


def build_model(mode: Mode, cfg: Config = CFG, lora: bool = True,
                grad_ckpt: bool = True, verbose: bool = True):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kw: dict[str, Any] = dict(device_map={"": 0}, attn_implementation="sdpa")

    if mode.quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=mode.compute,
            llm_int8_skip_modules=(
                ["visual", "lm_head", "merger"] if mode.skip_vis_quant else ["lm_head"]
            ),
        )
    kw["dtype"] = mode.compute

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_id, **kw)
    except TypeError:                                # older transformers
        kw["torch_dtype"] = kw.pop("dtype")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_id, **kw)

    model.config.use_cache = False
    if mode.vis_fp32:
        _force_visual_fp32(model, out_dtype=mode.compute)

    if lora:
        from peft import LoraConfig, get_peft_model

        targets = lora_target_names(model)
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
        ))
        # Trainable params in fp32. This is what prevents AdamW from doing its
        # m/(sqrt(v)+eps) arithmetic in fp16, where eps=1e-8 underflows below
        # fp16's smallest normal (~6.1e-5) and the update divides by ~0.
        n_train = 0
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
                n_train += p.numel()
        if verbose:
            print(f"  LoRA on {len(targets)} modules | {n_train/1e6:.1f}M trainable (fp32)")

    if grad_ckpt:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if verbose:
        print(f"  resident: {vram_now_gb():.2f} GB")
    return model


# --------------------------------------------------------------------------
# NaN guards
# --------------------------------------------------------------------------

class NaNGuard:
    """Central non-finite detector. Every stage funnels through this.

    Distinguishes *recoverable* events (one bad batch -> skip it) from
    *terminal* ones (adapter weights corrupted -> stop before the checkpoint
    is poisoned), which is precisely the distinction the original run lacked.
    """

    def __init__(self, max_consecutive_skips: int = CFG.max_consecutive_skips):
        self.max_consecutive_skips = max_consecutive_skips
        self.skipped_empty = 0
        self.skipped_nan_loss = 0
        self.skipped_nan_grad = 0
        self.consecutive_skips = 0
        self.events: list[dict] = []

    def record_skip(self, kind: str, step: int, detail: str = "") -> None:
        setattr(self, f"skipped_{kind}", getattr(self, f"skipped_{kind}") + 1)
        self.consecutive_skips += 1
        self.events.append({"kind": kind, "step": step, "detail": detail})

    def record_ok(self) -> None:
        self.consecutive_skips = 0

    @property
    def stalled(self) -> bool:
        return self.consecutive_skips >= self.max_consecutive_skips

    @staticmethod
    def check_loss(loss: torch.Tensor) -> bool:
        return bool(torch.isfinite(loss).all())

    @staticmethod
    def nonfinite_params(model) -> list[str]:
        return [n for n, p in model.named_parameters()
                if p.requires_grad and not torch.isfinite(p.data).all()]

    @staticmethod
    def nonfinite_grads(model) -> list[str]:
        return [n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]

    def summary(self) -> str:
        return (f"skipped: {self.skipped_empty} empty-label, "
                f"{self.skipped_nan_loss} non-finite-loss, "
                f"{self.skipped_nan_grad} non-finite-grad")


class NonFiniteWeights(RuntimeError):
    """Raised when adapter weights themselves go non-finite -- terminal."""


class TrainingStalled(RuntimeError):
    """Raised when too many consecutive batches are skipped -- terminal."""


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
