"""Minimal side-by-side: base model vs fine-tuned adapter, same image, same prompt.

    python -m finetune.compare --adapter adapters/qwen25vl-vrsbench-qlora/final
    python -m finetune.compare --adapter ... -n 5 --full
    python -m finetune.compare --adapter ... --image data/vrsbench/Images_val/05866_0000.png

Loads the 4-bit base once and toggles the LoRA adapter on and off around it
(`disable_adapter()`), so both answers come from one model in one 6 GB
allocation -- no reload, no second copy, and the only difference between the
two outputs is the adapter itself.

The headline metric is whether the output parses as JSON. The base model
answers "Describe the image in detail." with a prose paragraph; a working
adapter answers with the VRSBench schema. That is the difference to show.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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
    load_split,
    set_seed,
    to_device,
)

RULE = "=" * 72


def generate(model, processor, mode, image_path: str, question: str,
             max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")

    with torch.no_grad(), autocast_ctx(mode):
        out = model.generate(
            **to_device(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def describe(text: str) -> tuple[str, dict | None]:
    """Classify an output: valid schema JSON, or prose."""
    if not text:
        return "EMPTY", None
    if set(text.strip()) <= {"!", " ", "\n"}:
        return "NaN-CORRUPTED (all '!')", None
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return "prose (not JSON)", None
    if not isinstance(d, dict):
        return "JSON but not an object", None
    keys = [k for k in ("caption", "objects", "qa_pairs") if k in d]
    return f"valid JSON, keys={keys}", d


def brief(text: str, full: bool, width: int = 68) -> str:
    if full:
        return textwrap.indent(textwrap.fill(text, width), "    ")
    head = text if len(text) <= 300 else text[:300] + " ..."
    return textwrap.indent(textwrap.fill(head, width), "    ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", default=None)
    ap.add_argument("-n", type=int, default=3, help="validation images to compare")
    ap.add_argument("--image", default=None, help="compare one specific image instead")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--full", action="store_true", help="print untruncated outputs")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter not found: {adapter}")
        return 2

    meta_path = adapter / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    modes = available_modes()
    mode_name = args.mode or meta.get("mode") or next(n for n in MODE_PREFERENCE if n in modes)
    mode = modes[mode_name]

    print(RULE)
    print(f"adapter : {adapter}")
    print(f"mode    : {mode_name}   (trained {meta.get('step', '?')} steps, "
          f"lr={meta.get('lr', '?')})")
    print(RULE)

    set_seed()
    processor = build_processor()

    print("loading base model (4-bit) ...")
    free()
    model = build_model(mode, lora=False, grad_ckpt=False, verbose=False)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True

    # Refuse to report on a corrupted adapter -- this is the exact failure
    # that produced the endless '!' output previously.
    bad = [n for n, p in model.named_parameters()
           if "lora" in n and not torch.isfinite(p).all()]
    if bad:
        print(f"\n!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors")
        print(f"!! e.g. {bad[:3]}")
        print("!! Retrain with a mode that survived `python -m finetune.warmup`.")
        return 1
    print("adapter weight check: all finite\n")

    if args.image:
        rows = [{"image": args.image, "question": PROMPT, "answer": ""}]
    else:
        rows = load_split(CFG.val_jsonl, compact=True)[:args.n]

    base_json = tuned_json = 0
    for i, r in enumerate(rows, 1):
        q = r["question"]
        print(RULE)
        print(f"[{i}/{len(rows)}]  {Path(r['image']).name}   prompt: {q!r}")
        print(RULE)

        with model.disable_adapter():
            base_out = generate(model, processor, mode, r["image"], q, args.max_new_tokens)
        tuned_out = generate(model, processor, mode, r["image"], q, args.max_new_tokens)

        b_kind, _ = describe(base_out)
        t_kind, t_obj = describe(tuned_out)
        base_json += b_kind.startswith("valid JSON")
        tuned_json += t_kind.startswith("valid JSON")

        print(f"\n  BASE  [{b_kind}]")
        print(brief(base_out, args.full))
        print(f"\n  TUNED [{t_kind}]")
        print(brief(tuned_out, args.full))
        if t_obj and "caption" in t_obj:
            print(f"\n  tuned caption : {t_obj['caption'][:200]}")
        if r["answer"]:
            try:
                ref = json.loads(r["answer"])
                print(f"  ground truth  : {str(ref.get('caption', ''))[:200]}")
            except json.JSONDecodeError:
                pass
        print()

    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"  images compared          : {len(rows)}")
    print(f"  base  -> valid schema JSON: {base_json}/{len(rows)}")
    print(f"  tuned -> valid schema JSON: {tuned_json}/{len(rows)}")
    print()
    if tuned_json > base_json:
        print("  The adapter changed the output format from prose to the")
        print("  VRSBench schema. That is the difference to demo.")
    elif tuned_json == base_json == 0:
        print("  Neither produces valid JSON -- the adapter has not taken effect.")
        print("  Train for more steps (--max-steps 300+) or check the LR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
