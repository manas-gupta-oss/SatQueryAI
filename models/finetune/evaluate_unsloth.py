r"""Evaluate an Unsloth-trained VRSBench adapter with the SAME metrics as evaluate.py.

    .venv-unsloth\Scripts\python.exe -m finetune.evaluate_unsloth \
        --adapter adapters/qwen25vl-vrsbench-unsloth/step-600

The Unsloth adapter is written by peft 0.20 / transformers 5.x (regex
target_modules, peft-0.20-only config keys), so peft 0.17.1 in the main .venv
cannot load it. This swaps only the model-loading path and imports the scoring
functions unchanged from finetune.evaluate, so the numbers are directly
comparable to the plain-stack run.
"""

from __future__ import annotations

# Unsloth must be imported before transformers.
from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from finetune.common import CFG, Mode, free  # noqa: E402
from finetune.evaluate import RULE, evaluate_model, load_rows, report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--save", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter not found: {adapter}"); return 2
    meta_p = adapter / "train_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    rows = load_rows(args.n)
    print(RULE)
    print(f"adapter : {adapter}  (step {meta.get('step','?')}, backend "
          f"{meta.get('backend','unsloth')}, eff.batch {meta.get('effective_batch','?')})")
    print(f"eval set: {len(rows)} VRSBench val images")
    print(RULE)

    free()
    model, processor = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    # Same pixel budget as training so vision-token counts match.
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        try:
            ip.size = {"shortest_edge": CFG.min_pixels, "longest_edge": CFG.max_pixels}
        except Exception:                            # noqa: BLE001
            pass

    bad = [n for n, p in model.named_parameters()
           if "lora" in n and not torch.isfinite(p).all()]
    if bad:
        print(f"!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors {bad[:3]}"); return 1
    print(f"adapter weight check: all finite | "
          f"{torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    mode = Mode("nf4", torch.bfloat16, torch.bfloat16)
    results = [evaluate_model(model, processor, mode, rows, args.max_new_tokens,
                              f"UNSLOTH step-{meta.get('step','?')}", args.quiet)]

    print("\n" + RULE); print("RESULTS"); print(RULE)
    for r in results:
        report(r)

    print("\n" + RULE); print("SAMPLES"); print(RULE)
    for s in results[0]["samples"]:
        if s["json"]:
            print(f"{s['image']}  objects pred={s['n_obj_pred']} ref={s['n_obj_ref']}")
            print(f"   cls pred={s['cls_pred']}  ref={s['cls_ref']}")
        else:
            print(f"{s['image']}  [NOT VALID JSON]")
        print(f"   pred: {s['pred']}")
        print(f"   ref : {s['ref']}\n")

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
