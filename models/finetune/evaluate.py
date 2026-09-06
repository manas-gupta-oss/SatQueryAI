r"""Quantitative VRSBench eval: schema conformance, object counting, class F1.

    python -m finetune.evaluate --adapter adapters/qwen25vl-vrsbench-qlora/final -n 60

Why not just eyeball compare.py
-------------------------------
The known weakness of this agent is fine-grained: it read five ships as one
ship, and a bridge as a vehicle. Those errors are invisible in val loss and in
a three-image spot check. This scores them:

  valid JSON       -- does it emit the schema at all
  object count     -- exact-match rate and mean absolute error vs ground truth
                      (the "five ships -> one ship" failure, measured)
  obj_cls micro-F1 -- are the right object categories named
  caption overlap  -- content-word overlap with the reference caption
  qa_pairs count   -- structural completeness

Object count is the headline number here: it is the metric that was clearly
wrong before, so it is the one that can show whether more training helped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

from finetune.common import (
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
STOP = {"the", "a", "an", "is", "are", "of", "in", "on", "and", "to", "at", "with",
        "some", "there", "it", "as", "by", "from", "that", "this", "image", "shows"}


def generate(model, processor, mode, row: dict, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": row["question"]}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    with Image.open(row["image"]) as im:
        img = im.convert("RGB")
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")
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


def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP}


def obj_classes(objs) -> list[str]:
    out = []
    for o in objs or []:
        if isinstance(o, dict):
            c = str(o.get("obj_cls") or "").strip().lower()
            if c:
                out.append(c)
    return out


def evaluate_model(model, processor, mode, rows, max_new_tokens, label, quiet=False):
    res = {"label": label, "n": 0, "valid_json": 0, "corrupted": 0,
           "count_exact": 0, "count_abs_err": 0, "count_n": 0,
           "cls_tp": 0, "cls_fp": 0, "cls_fn": 0,
           "overlap_sum": 0.0, "overlap_n": 0,
           "qa_abs_err": 0, "qa_n": 0, "samples": []}

    for i, r in enumerate(rows, 1):
        text = generate(model, processor, mode, r, max_new_tokens)
        res["n"] += 1
        if text and set(text.strip()) <= {"!", " ", "\n"}:
            res["corrupted"] += 1
        obj = parse(text)
        ref = json.loads(r["answer"])
        if obj is None:
            if len(res["samples"]) < 5:
                res["samples"].append({"image": Path(r["image"]).name, "json": False,
                                       "pred": text[:180], "ref": str(ref.get("caption"))[:180]})
            continue
        res["valid_json"] += 1

        # object count -- the "five ships read as one" failure, measured
        pn, rn = len(obj.get("objects") or []), len(ref.get("objects") or [])
        res["count_n"] += 1
        res["count_abs_err"] += abs(pn - rn)
        if pn == rn:
            res["count_exact"] += 1

        # object classes, multiset-insensitive (set F1)
        pc, rc = set(obj_classes(obj.get("objects"))), set(obj_classes(ref.get("objects")))
        res["cls_tp"] += len(pc & rc); res["cls_fp"] += len(pc - rc); res["cls_fn"] += len(rc - pc)

        po, ro = content_words(obj.get("caption", "")), content_words(ref.get("caption", ""))
        if ro:
            res["overlap_sum"] += len(po & ro) / len(ro)
            res["overlap_n"] += 1

        pq, rq = len(obj.get("qa_pairs") or []), len(ref.get("qa_pairs") or [])
        res["qa_n"] += 1
        res["qa_abs_err"] += abs(pq - rq)

        if len(res["samples"]) < 5:
            res["samples"].append({
                "image": Path(r["image"]).name, "json": True,
                "n_obj_pred": pn, "n_obj_ref": rn,
                "cls_pred": sorted(pc), "cls_ref": sorted(rc),
                "pred": str(obj.get("caption"))[:180], "ref": str(ref.get("caption"))[:180]})

        if not quiet and i % 10 == 0:
            print(f"    {label}: {i}/{len(rows)} ...", flush=True)
    return res


def report(r: dict) -> None:
    n = max(1, r["n"])
    print(f"\n--- {r['label']} ({r['n']} images) ---")
    print(f"  valid schema JSON  : {r['valid_json']}/{r['n']} = {r['valid_json']/n*100:5.1f}%"
          + (f"   CORRUPTED('!'): {r['corrupted']}" if r["corrupted"] else ""))
    if r["count_n"]:
        print(f"  object count exact : {r['count_exact']}/{r['count_n']} = "
              f"{r['count_exact']/r['count_n']*100:5.1f}%"
              f"   MAE {r['count_abs_err']/r['count_n']:.2f}")
    tp, fp, fn = r["cls_tp"], r["cls_fp"], r["cls_fn"]
    if tp + fp + fn:
        p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        print(f"  object class       : P={p*100:5.1f}% R={rc*100:5.1f}% "
              f"F1={2*p*rc/max(1e-9,p+rc)*100:5.1f}%  (tp={tp} fp={fp} fn={fn})")
    if r["overlap_n"]:
        print(f"  caption overlap    : {r['overlap_sum']/r['overlap_n']*100:5.1f}%")
    if r["qa_n"]:
        print(f"  qa_pairs count MAE : {r['qa_abs_err']/r['qa_n']:.2f}")


def load_rows(n: int) -> list[dict]:
    set_seed()
    return load_split(CFG.val_jsonl, compact=True)[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", default=None)
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--base", action="store_true", help="also score the base model")
    ap.add_argument("--save", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter not found: {adapter}"); return 2
    meta_p = adapter / "train_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    modes = available_modes()
    mode_name = args.mode or meta.get("mode") or next(n for n in MODE_PREFERENCE if n in modes)
    mode = modes[mode_name]

    rows = load_rows(args.n)
    print(RULE)
    print(f"adapter : {adapter}  (step {meta.get('step','?')}, mode {mode_name})")
    print(f"eval set: {len(rows)} VRSBench val images")
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
        print(f"!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors {bad[:3]}"); return 1
    print("adapter weight check: all finite\n")

    results = [evaluate_model(model, processor, mode, rows, args.max_new_tokens,
                              f"TUNED step-{meta.get('step','?')}", args.quiet)]
    if args.base:
        print("evaluating BASE ...")
        with model.disable_adapter():
            results.append(evaluate_model(model, processor, mode, rows,
                                          args.max_new_tokens, "BASE", args.quiet))

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
