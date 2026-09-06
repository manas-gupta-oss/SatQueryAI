r"""Evaluate an Unsloth-trained adapter, using the SAME metrics as evaluate.py.

    .venv-unsloth\Scripts\python.exe -m bitemporal.evaluate_unsloth \
        --adapter adapters/qwen25vl-levircc-unsloth/step-900

Why a separate entry point
--------------------------
The Unsloth run writes its adapter with peft 0.20 / transformers 5.x:

  * `target_modules` is a REGEX string, not the explicit 112-name list
  * `base_model_name_or_path` points at unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit
  * the config carries peft-0.20-only keys (alora_invocation_tokens,
    arrow_config, lora_ga_config, ...) that peft 0.17.1 rejects

So it cannot be loaded by bitemporal/evaluate.py in the pinned .venv. Rather
than duplicate the scoring, this module imports the metric functions from
bitemporal.evaluate unchanged and only swaps the model-loading path -- so the
numbers are directly comparable to the plain-stack result.
"""

from __future__ import annotations

# Unsloth must be imported before transformers.
from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from bitemporal.common import CFG, Mode, free, load_split, set_seed  # noqa: E402
from bitemporal.evaluate import RULE, evaluate_model, report  # noqa: E402

UNSLOTH_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--base", action="store_true",
                    help="also score the base model (slow; it is already known to be 50%%)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter not found: {adapter}")
        return 2
    meta_p = adapter / "train_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    set_seed()
    val = load_split(CFG.val_jsonl)
    ch = [r for r in val if r["changeflag"] == 1]
    un = [r for r in val if r["changeflag"] == 0]
    half = args.n // 2
    rows = ch[:half] + un[:half]          # balanced, so chance is exactly 50%

    print(RULE)
    print(f"adapter : {adapter}  (step {meta.get('step','?')}, backend "
          f"{meta.get('backend','unsloth')})")
    print(f"eval set: {len(rows)} val pairs -- {min(half,len(ch))} changed / "
          f"{min(half,len(un))} unchanged")
    print(RULE)

    free()
    model, processor = FastVisionModel.from_pretrained(
        str(adapter), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    bad = [n for n, p in model.named_parameters()
           if "lora" in n and not torch.isfinite(p).all()]
    if bad:
        print(f"!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors {bad[:3]}")
        return 1
    print(f"adapter weight check: all finite | "
          f"{torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    # Unsloth runs bf16 internally; autocast is a no-op wrapper here.
    mode = Mode("nf4", torch.bfloat16, torch.bfloat16)

    results = []
    print("evaluating TUNED (unsloth) ...")
    results.append(evaluate_model(model, processor, mode, rows,
                                  args.max_new_tokens, "TUNED (unsloth)", args.quiet))
    if args.base:
        print("evaluating BASE ...")
        with model.disable_adapter():
            results.append(evaluate_model(model, processor, mode, rows,
                                          args.max_new_tokens, "BASE", args.quiet))

    print("\n" + RULE)
    print("RESULTS")
    print(RULE)
    for r in results:
        report(r)

    print("\n" + RULE)
    print("SAMPLE PREDICTIONS")
    print(RULE)
    for s in results[0]["samples"]:
        flag = "OK " if s["pred_changed"] is s["truth_changed"] else "MISS"
        print(f"[{flag}] {s['file']}  truth_changed={s['truth_changed']} "
              f"pred={s['pred_changed']}")
        print(f"    pred: {s['pred']}")
        print(f"    ref : {s['ref']}\n")

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
