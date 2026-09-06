r"""Measure whether the bi-temporal agent actually detects change.

    python -m bitemporal.evaluate --adapter adapters/qwen25vl-levircc-qlora/final
    python -m bitemporal.evaluate --adapter ... -n 200 --save eval.json

Why this exists
---------------
Validation loss cannot answer the question that matters here. Half the training
set carries one identical target ("The scene is the same as before."), so a
model that ALWAYS says "no change" would score ~50% accuracy and a very low
loss while being useless.

So this evaluates the two halves SEPARATELY and reports the asymmetric error
that a single accuracy number hides:

    miss rate   -- a real change reported as "no change"   (dangerous)
    false alarm -- an unchanged pair reported as changed   (noisy but safe)

A model collapsed onto "no change" shows near-100% accuracy on unchanged pairs
and near-0% on changed ones. Balanced accuracy catches that; plain accuracy
does not.

Also reported, on the CHANGED subset only:
  * valid-JSON rate and schema conformance
  * class agreement (building / road) against ground truth
  * caption word overlap vs the reference (a rough content check, not BLEU)
  * base-model comparison, so the adapter's contribution is visible
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from bitemporal.common import (
    CFG,
    MODE_PREFERENCE,
    autocast_ctx,
    available_modes,
    build_model,
    build_processor,
    free,
    load_split,
    set_seed,
    to_device,
)

RULE = "=" * 74
STOP = {"the", "a", "an", "is", "are", "of", "in", "on", "and", "to", "at",
        "with", "some", "there", "it", "as", "by", "from", "that", "this"}


def generate(model, processor, mode, row: dict, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": row["question"]},
    ]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs = []
    for k in ("image_a", "image_b"):
        with Image.open(row[k]) as im:
            imgs.append(im.convert("RGB"))
    inputs = processor(text=[prompt], images=imgs, return_tensors="pt")
    with torch.no_grad(), autocast_ctx(mode):
        out = model.generate(**to_device(inputs), max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def parse(text: str) -> dict | None:
    try:
        o = json.loads(text)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


NEG = re.compile(r"\b(no difference|no change|same as before|identical|unchanged|"
                 r"nothing has changed|remains the same)\b", re.I)


def predicted_change(text: str, obj: dict | None) -> bool | None:
    """True/False if we can tell, None if the output is unusable.

    For the base model (prose, no schema) fall back to phrase matching so the
    comparison is fair rather than automatically scoring it zero.
    """
    if obj is not None and isinstance(obj.get("change_detected"), bool):
        return obj["change_detected"]
    if not text.strip():
        return None
    return not bool(NEG.search(text))


def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP}


def evaluate_model(model, processor, mode, rows, max_new_tokens, label, quiet):
    res = {
        "label": label, "n": 0,
        "changed": {"n": 0, "correct": 0}, "unchanged": {"n": 0, "correct": 0},
        "unusable": 0, "valid_json": 0,
        "overlap_sum": 0.0, "overlap_n": 0,
        "class_tp": 0, "class_fp": 0, "class_fn": 0,
        "samples": [],
    }
    for i, r in enumerate(rows, 1):
        text = generate(model, processor, mode, r, max_new_tokens)
        obj = parse(text)
        truth = bool(r["changeflag"])
        pred = predicted_change(text, obj)
        res["n"] += 1
        if obj is not None:
            res["valid_json"] += 1
        if pred is None:
            res["unusable"] += 1
        bucket = res["changed"] if truth else res["unchanged"]
        bucket["n"] += 1
        if pred is truth:
            bucket["correct"] += 1

        if truth and obj is not None:
            ref = json.loads(r["answer"])
            po, ro = content_words(obj.get("change_summary", "")), content_words(ref["change_summary"])
            if ro:
                res["overlap_sum"] += len(po & ro) / len(ro)
                res["overlap_n"] += 1
            pc = {c for c in (obj.get("changed_classes") or []) if isinstance(c, str)}
            rc = set(ref["changed_classes"])
            res["class_tp"] += len(pc & rc)
            res["class_fp"] += len(pc - rc)
            res["class_fn"] += len(rc - pc)

        if len(res["samples"]) < 6:
            res["samples"].append({
                "file": r["filename"], "truth_changed": truth, "pred_changed": pred,
                "pred": (obj.get("change_summary") if obj else text)[:200],
                "ref": json.loads(r["answer"])["change_summary"][:200],
            })
        if not quiet and i % 25 == 0:
            print(f"    {label}: {i}/{len(rows)} ...", flush=True)
    return res


def report(res: dict) -> None:
    ch, un = res["changed"], res["unchanged"]
    ch_acc = ch["correct"] / max(1, ch["n"])
    un_acc = un["correct"] / max(1, un["n"])
    bal = (ch_acc + un_acc) / 2
    plain = (ch["correct"] + un["correct"]) / max(1, res["n"])

    print(f"\n--- {res['label']} ({res['n']} pairs) ---")
    print(f"  recall on CHANGED   : {ch['correct']:4d}/{ch['n']:<4d} = {ch_acc*100:5.1f}%"
          f"   (miss rate {100-ch_acc*100:5.1f}%  <- the dangerous error)")
    print(f"  recall on UNCHANGED : {un['correct']:4d}/{un['n']:<4d} = {un_acc*100:5.1f}%"
          f"   (false alarm {100-un_acc*100:5.1f}%)")
    print(f"  balanced accuracy   : {bal*100:5.1f}%     plain accuracy: {plain*100:5.1f}%")
    print(f"  valid schema JSON   : {res['valid_json']}/{res['n']}"
          f"   unusable outputs: {res['unusable']}")
    if res["overlap_n"]:
        print(f"  caption content overlap (changed only): "
              f"{res['overlap_sum']/res['overlap_n']*100:5.1f}%")
    tp, fp, fn = res["class_tp"], res["class_fp"], res["class_fn"]
    if tp + fp + fn:
        p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f1 = 2 * p * rc / max(1e-9, p + rc)
        print(f"  changed_classes     : P={p*100:.1f}% R={rc*100:.1f}% F1={f1*100:.1f}%"
              f"  (tp={tp} fp={fp} fn={fn})")

    if bal < 0.55:
        print("  >> NEAR CHANCE. Check whether it collapsed onto one answer.")
    elif ch_acc < 0.5:
        print("  >> Biased toward 'no change': it misses more real changes than it finds.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", default=None)
    ap.add_argument("-n", type=int, default=120, help="val pairs (balanced changed/unchanged)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--no-base", action="store_true", help="skip the base-model comparison")
    ap.add_argument("--save", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter not found: {adapter}")
        return 2
    meta_p = adapter / "train_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    modes = available_modes()
    mode_name = args.mode or meta.get("mode") or next(n for n in MODE_PREFERENCE if n in modes)
    mode = modes[mode_name]

    set_seed()
    val = load_split(CFG.val_jsonl)
    ch = [r for r in val if r["changeflag"] == 1]
    un = [r for r in val if r["changeflag"] == 0]
    half = args.n // 2
    rows = ch[:half] + un[:half]                 # deliberately balanced

    print(RULE)
    print(f"adapter : {adapter}  (step {meta.get('step','?')}, mode {mode_name})")
    print(f"eval set: {len(rows)} val pairs -- {min(half,len(ch))} changed / "
          f"{min(half,len(un))} unchanged")
    print(RULE)

    free()
    processor = build_processor()
    model = build_model(mode, lora=False, grad_ckpt=False, verbose=False)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval(); model.config.use_cache = True

    bad = [n for n, p in model.named_parameters()
           if "lora" in n and not torch.isfinite(p).all()]
    if bad:
        print(f"!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors {bad[:3]}")
        return 1
    print("adapter weight check: all finite\n")

    results = []
    print("evaluating TUNED ...")
    results.append(evaluate_model(model, processor, mode, rows,
                                  args.max_new_tokens, "TUNED", args.quiet))
    if not args.no_base:
        print("evaluating BASE (adapter disabled) ...")
        with model.disable_adapter():
            results.append(evaluate_model(model, processor, mode, rows,
                                          args.max_new_tokens, "BASE", args.quiet))

    print("\n" + RULE)
    print("RESULTS")
    print(RULE)
    for r in results:
        report(r)

    print("\n" + RULE)
    print("SAMPLE PREDICTIONS (tuned)")
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
