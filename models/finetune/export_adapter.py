"""Export a training checkpoint as a lean, repo-committable inference adapter.

    python -m finetune.export_adapter                    # defaults to the 600-step run
    python -m finetune.export_adapter --src ... --dst ...

Training keeps LoRA parameters in fp32 (that is what stops AdamW doing fp16
arithmetic, which is what corrupted the earlier runs). For *inference* that
precision is pointless -- the base model is 4-bit NF4 -- so this recasts the
adapter to bf16.

The practical reason: fp32 puts `adapter_model.safetensors` at ~115 MB, and
GitHub hard-rejects any file over 100 MB. bf16 halves it to ~58 MB, which
commits normally with no Git LFS and no external hosting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "adapters" / "qwen25vl-vrsbench-qlora-leg2" / "step-300"
DEFAULT_DST = PROJECT_ROOT / "adapters" / "report-adapter"

# Everything the processor needs at inference time, plus the PEFT config.
KEEP = [
    "adapter_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",   # transformers 4.x name
    "processor_config.json",      # transformers 5.x name (Unsloth writes this)
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "train_meta.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    weights = src / "adapter_model.safetensors"
    if not weights.exists():
        print(f"no adapter weights at {weights}")
        return 2

    dtype = getattr(torch, args.dtype)
    dst.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(weights))
    out, n_bad = {}, []
    for name, t in tensors.items():
        if t.is_floating_point():
            if not torch.isfinite(t).all():
                n_bad.append(name)
            out[name] = t.to(dtype)
        else:
            out[name] = t

    # Never export something that would generate '!' forever.
    if n_bad:
        print(f"REFUSING TO EXPORT: {len(n_bad)} non-finite tensors, e.g. {n_bad[:3]}")
        return 1

    save_file(out, str(dst / "adapter_model.safetensors"),
              metadata={"format": "pt"})

    for fn in KEEP:
        s = src / fn
        if s.exists():
            shutil.copy2(s, dst / fn)

    meta_path = dst / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["exported_dtype"] = args.dtype
    try:
        origin = str(src.relative_to(PROJECT_ROOT))
    except ValueError:                               # src outside the project tree
        origin = str(src)
    meta["exported_from"] = origin.replace("\\", "/")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    before = weights.stat().st_size / 1024**2
    after = (dst / "adapter_model.safetensors").stat().st_size / 1024**2
    total = sum(p.stat().st_size for p in dst.iterdir() if p.is_file()) / 1024**2
    print(f"tensors        : {len(out)} ({len(out)//2} LoRA pairs)")
    print(f"adapter weights: {before:.1f} MB -> {after:.1f} MB  ({args.dtype})")
    print(f"export total   : {total:.1f} MB   at {dst}")
    print(f"github 100 MB limit: {'OK' if after < 100 else 'STILL TOO LARGE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
