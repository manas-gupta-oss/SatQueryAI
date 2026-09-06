"""Shared machinery for the bi-temporal (change-detection) agent.

Deliberately self-contained -- it does NOT import from `finetune/`, so work
here cannot disturb the VRSBench agent's code or adapters.

The one structural difference from the single-image agent is that every sample
carries TWO images, so the collator emits two `<|image_pad|>` blocks per row and
the pixel budget is tuned for LEVIR-CC's 256x256 tiles.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "bitemporal"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    model_id: str = MODEL_ID
    train_jsonl: Path = DATA_DIR / "train.jsonl"
    val_jsonl: Path = DATA_DIR / "val.jsonl"
    test_jsonl: Path = DATA_DIR / "test.jsonl"
    out_dir: Path = PROJECT_ROOT / "adapters" / "qwen25vl-levircc-qlora"

    # LEVIR-CC tiles are 256x256 = 65,536 px. Keep min_pixels BELOW that or the
    # processor upscales them and we pay for tokens carrying no extra detail.
    # 65,536 px -> smart_resize to 252x252 -> (252/14)^2 / 4 = 81 tokens/image,
    # so 162 for the pair.
    min_pixels: int = 64 * 28 * 28          # 50,176
    max_pixels: int = 144 * 28 * 28         # 112,896

    # Measured on the prepared data: p95 total ~377 tokens, worst case ~400.
    # 640 leaves headroom without paying for it in the logit tensor.
    max_len: int = 640

    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-4
    warmup_steps: int = 20
    max_steps: int = 600
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    log_every: int = 25
    eval_every: int = 100
    save_every: int = 100
    eval_samples: int = 48

    nan_check_every: int = 10
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


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def load_split(jsonl_path: str | Path) -> list[dict]:
    """Load a prepared bi-temporal jsonl (see bitemporal/prepare_data.py)."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Missing {jsonl_path}. Run: python -m bitemporal.prepare_data")

    rows, dropped = [], 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (Path(r["image_a"]).exists() and Path(r["image_b"]).exists()):
                dropped += 1
                continue
            if not (r.get("answer") or "").strip():
                dropped += 1
                continue
            rows.append(r)
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
    """Two images per sample, completion-only labels.

    Masks to -100: the prompt, padding, and both `<|image_pad|>` blocks. Reports
    `empty` -- rows where truncation left zero supervised tokens, which would
    make cross-entropy compute 0/0 = NaN.
    """

    def __init__(self, processor, max_len: int = CFG.max_len):
        self.p = processor
        self.tok = processor.tokenizer
        self.max_len = max_len
        self.assistant_header = self.tok(
            "<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

        self.vision_ids: set[int] = set()
        for t in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                  "<|vision_end|>", "<|vision_pad|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self.vision_ids.add(i)

    @staticmethod
    def _rfind_sub(seq: list[int], sub: list[int]) -> int:
        n, m = len(seq), len(sub)
        for i in range(n - m, -1, -1):
            if seq[i:i + m] == sub:
                return i + m
        return -1

    def __call__(self, rows: list[dict]) -> tuple[dict, dict]:
        texts: list[str] = []
        images: list[Image.Image] = []       # flat, in the order the tokens appear

        for r in rows:
            msgs = [{"role": "user", "content": [
                {"type": "image"},           # earlier
                {"type": "image"},           # later
                {"type": "text", "text": r["question"]},
            ]}]
            prompt = self.p.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            texts.append(prompt + r["answer"] + "<|im_end|>\n")
            for key in ("image_a", "image_b"):
                with Image.open(r[key]) as im:
                    images.append(im.convert("RGB"))

        try:
            batch = self.p(
                text=texts, images=images, return_tensors="pt",
                padding=True, truncation=True, max_length=self.max_len,
            )
        except (TypeError, ValueError):
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
            if cut < 0:
                cut = ids.size(1)
            labels[b, :cut] = -100
            if attn is not None:
                labels[b][attn[b] == 0] = -100
            for vid in self.vision_ids:
                labels[b][ids[b] == vid] = -100
            labels[b][ids[b] == self.tok.pad_token_id] = -100
            supervised.append(int((labels[b] != -100).sum()))

        batch["labels"] = labels
        stats = {
            "seq_len": int(ids.shape[1]),
            "supervised": supervised,
            "empty": sum(1 for s in supervised if s == 0),
            "n_vision_tok": int(sum((ids == i).sum() for i in self.vision_ids)),
            "n_images": len(images),
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
    """Only modes that fit in 6 GB. Unquantized weights alone are 7.5 GB."""
    modes: dict[str, Mode] = {}
    if torch.cuda.is_bf16_supported():
        # sm_89 (Ada) has native bf16. Measured on this model, the ViT emits
        # activations near 39,680 -- 60% of fp16's 65,504 ceiling -- so bf16
        # removes the overflow risk entirely rather than merely surviving it.
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
    while hasattr(m, "base_model"):
        m = m.base_model
    while hasattr(m, "model") and not hasattr(m, "visual"):
        m = m.model
    if hasattr(m, "visual"):
        return m.visual
    raise AttributeError("could not locate the vision tower")


def _force_visual_fp32(model, out_dtype: torch.dtype):
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
    """LM only. Qwen2.5-VL's ViT reuses gate_proj/up_proj/down_proj, so PEFT's
    suffix matching would otherwise attach adapters to the vision tower too."""
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
                ["visual", "lm_head", "merger"] if mode.skip_vis_quant else ["lm_head"]),
        )
    kw["dtype"] = mode.compute

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_id, **kw)
    except TypeError:
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
        # Trainable params in fp32: this is what keeps AdamW out of fp16
        # arithmetic, where eps=1e-8 underflows below fp16's smallest normal
        # (~6.1e-5) and m/(sqrt(v)+eps) divides by ~0.
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
            gradient_checkpointing_kwargs={"use_reentrant": False})

    if verbose:
        print(f"  resident: {vram_now_gb():.2f} GB")
    return model


# --------------------------------------------------------------------------
# NaN guards
# --------------------------------------------------------------------------

class NaNGuard:
    """Separates recoverable events (skip one micro-batch) from terminal ones
    (adapter corrupted -> stop before the checkpoint is written)."""

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
    """Adapter weights themselves went non-finite -- terminal."""


class TrainingStalled(RuntimeError):
    """Too many consecutive unusable batches -- terminal."""
