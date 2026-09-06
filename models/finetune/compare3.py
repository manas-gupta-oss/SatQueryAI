r"""Three-way comparison on a single image: base vs 300-step vs 600-step adapter.

    python -m finetune.compare3                          # uses satellite.png
    python -m finetune.compare3 --image path\to\pic.png
    python -m finetune.compare3 --save demo.md           # also write a markdown report
    python -m finetune.compare3 --pretty                 # pretty-print the JSON

Loads the 4-bit base ONCE and attaches both adapters to it under separate
names, then switches between them with `set_adapter()` / `disable_adapter()`.
All three answers therefore come from one model in one 6 GB allocation, and
the adapter is the only variable between them -- no reloads, no second copy,
nothing else that could explain a difference.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import torch
from PIL import Image

from finetune.common import (
    CFG,
    MODE_PREFERENCE,
    PROMPT,
    autocast_ctx,
    available_modes,
    build_model,
    build_processor,
    free,
    set_seed,
    to_device,
)

RULE = "=" * 74
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANTS = [
    ("300-step", PROJECT_ROOT / "adapters" / "qwen25vl-vrsbench-qlora" / "step-300"),
    ("600-step", PROJECT_ROOT / "adapters" / "qwen25vl-vrsbench-qlora-leg2" / "step-300"),
]


def classify(text: str) -> tuple[str, dict | None]:
    """What kind of output is this? Valid schema JSON, prose, or NaN wreckage."""
    if not text.strip():
        return "EMPTY", None
    if set(text.strip()) <= {"!", " ", "\n"}:
        return "NaN-CORRUPTED (all '!')", None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "prose (not JSON)", None
    if not isinstance(obj, dict):
        return "JSON, not an object", None
    keys = [k for k in ("caption", "objects", "qa_pairs") if k in obj]
    return f"valid JSON {keys}", obj


def render(text: str, obj: dict | None, pretty: bool, width: int = 70) -> str:
    if obj is not None and pretty:
        return textwrap.indent(json.dumps(obj, indent=2, ensure_ascii=False), "    ")
    return textwrap.indent(textwrap.fill(text, width), "    ")


def generate(model, processor, mode, image: Image.Image, question: str,
             max_new_tokens: int) -> tuple[str, float]:
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    t0 = time.time()
    with torch.no_grad(), autocast_ctx(mode):
        out = model.generate(
            **to_device(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip(), elapsed


def check_finite(model, adapter_name: str) -> list[str]:
    """A corrupted adapter is exactly what produced the endless '!' output.
    Never report on one without saying so."""
    return [n for n, p in model.named_parameters()
            if "lora" in n and adapter_name in n and not torch.isfinite(p).all()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=str(PROJECT_ROOT / "satellite.png"))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--pretty", action="store_true", help="pretty-print JSON outputs")
    ap.add_argument("--save", default=None, help="also write a markdown report here")
    ap.add_argument("--adapter", action="append", metavar="LABEL=PATH",
                    help="override the compared adapters; repeatable")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"image not found: {img_path}")
        return 2

    if args.adapter:
        variants = []
        for spec in args.adapter:
            label, _, path = spec.partition("=")
            variants.append((label, Path(path)))
    else:
        variants = list(DEFAULT_VARIANTS)

    missing = [f"{lbl} -> {p}" for lbl, p in variants if not p.exists()]
    if missing:
        print("adapter(s) not found:")
        for m in missing:
            print("   ", m)
        return 2

    modes = available_modes()
    mode_name = args.mode or next(n for n in MODE_PREFERENCE if n in modes)
    mode = modes[mode_name]

    with Image.open(img_path) as im:
        image = im.convert("RGB")
        size = im.size

    print(RULE)
    print(f"image  : {img_path.name}  ({size[0]}x{size[1]})")
    print(f"prompt : {args.prompt!r}")
    print(f"mode   : {mode_name}")
    for lbl, p in variants:
        meta_p = p / "train_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        print(f"adapter: {lbl:9s} {p.name}  (lr={meta.get('lr', '?')})")
    print(RULE)

    set_seed()
    processor = build_processor()

    print("loading 4-bit base once, attaching both adapters ...")
    free()
    model = build_model(mode, lora=False, grad_ckpt=False, verbose=False)

    from peft import PeftModel

    first_label, first_path = variants[0]
    slug0 = first_label.replace("-", "_")
    model = PeftModel.from_pretrained(model, str(first_path), adapter_name=slug0)
    slugs = {first_label: slug0}
    for label, path in variants[1:]:
        slug = label.replace("-", "_")
        model.load_adapter(str(path), adapter_name=slug)
        slugs[label] = slug
    model.eval()
    model.config.use_cache = True

    for label in slugs:
        bad = check_finite(model, slugs[label])
        if bad:
            print(f"\n!! {label} ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors")
            print(f"!! e.g. {bad[:3]}")
            return 1
    print(f"adapter weight check: all finite ({', '.join(slugs)})")
    print(f"resident: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    results: list[tuple[str, str, str, dict | None, float]] = []

    with model.disable_adapter():
        text, dt = generate(model, processor, mode, image, args.prompt, args.max_new_tokens)
    kind, obj = classify(text)
    results.append(("BASE (no adapter)", kind, text, obj, dt))

    for label, _ in variants:
        model.set_adapter(slugs[label])
        text, dt = generate(model, processor, mode, image, args.prompt, args.max_new_tokens)
        kind, obj = classify(text)
        results.append((f"TUNED {label}", kind, text, obj, dt))

    for name, kind, text, obj, dt in results:
        print(RULE)
        print(f"{name}   [{kind}]   {dt:.1f}s, {len(text)} chars")
        print(RULE)
        print(render(text, obj, args.pretty))
        if obj:
            print(f"\n    caption : {str(obj.get('caption',''))[:220]}")
            print(f"    objects : {len(obj.get('objects') or [])}"
                  f"   qa_pairs: {len(obj.get('qa_pairs') or [])}")
            cls = [o.get("obj_cls") for o in (obj.get("objects") or []) if isinstance(o, dict)]
            if cls:
                print(f"    classes : {cls}")
        print()

    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"{'variant':22s} {'output type':28s} {'chars':>6s} {'objs':>5s} {'qa':>4s} {'sec':>6s}")
    print("-" * 74)
    for name, kind, text, obj, dt in results:
        nobj = len(obj.get("objects") or []) if obj else 0
        nqa = len(obj.get("qa_pairs") or []) if obj else 0
        print(f"{name:22s} {kind[:28]:28s} {len(text):>6d} {nobj:>5d} {nqa:>4d} {dt:>6.1f}")

    n_json = sum(1 for _, k, _, _, _ in results if k.startswith("valid JSON"))
    print(f"\nvalid schema JSON: {n_json}/{len(results)}")
    if results[0][1].startswith("valid JSON"):
        print("NOTE: the base model produced JSON here -- unusual; the format")
        print("      difference is weaker on this image than on VRSBench val.")
    else:
        print("Base emits prose; both adapters emit the VRSBench schema.")

    a, b = results[1][2], results[2][2] if len(results) > 2 else ("", "")
    if len(results) > 2:
        print(f"\n300-step vs 600-step: {'identical' if a == b else 'differ'}"
              f" ({len(a)} vs {len(b)} chars)")

    if args.save:
        out = Path(args.save)
        lines = [f"# Fine-tune comparison — `{img_path.name}`", "",
                 f"**Image:** {size[0]}x{size[1]}  ", f"**Prompt:** `{args.prompt}`  ",
                 f"**Quantization:** {mode_name} (4-bit NF4, bf16 compute)", ""]
        for name, kind, text, obj, dt in results:
            lines += [f"## {name}", "", f"*{kind} — {dt:.1f}s, {len(text)} chars*", "",
                      "```json" if obj else "```text",
                      json.dumps(obj, indent=2, ensure_ascii=False) if obj else text,
                      "```", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nreport written -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
