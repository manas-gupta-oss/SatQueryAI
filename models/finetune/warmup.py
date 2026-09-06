"""Warm-up triage: is the NaN caused by quantization, the dataset, or the model?

Runs three escalating probes and prints a verdict. Nothing here trains a usable
adapter -- it exists so the real run starts from a known-good configuration.

    python -m finetune.warmup                # all three probes
    python -m finetune.warmup --stage data   # just the dataset probe
    python -m finetune.warmup --break-it     # reproduce the original failure

`--break-it` deliberately sets max_len=1024 on the *uncompacted* targets with
the empty-label guard disabled, which is the configuration the original
Unsloth run used. If the truncation hypothesis is right, that NaNs and the
default configuration does not.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import torch

from finetune.common import (
    CFG,
    MODE_PREFERENCE,
    Collator,
    Mode,
    NaNGuard,
    autocast_ctx,
    available_modes,
    build_model,
    build_processor,
    free,
    gpu_report,
    load_split,
    set_seed,
    to_device,
    vram_peak_gb,
)

RULE = "=" * 72


# --------------------------------------------------------------------------
# probe 1 -- forward numerics
# --------------------------------------------------------------------------

def probe_forward(name: str, mode: Mode, rows, collate, topk: int = 6) -> dict:
    """Forward passes only. Hook every submodule; report the first to go
    non-finite and the largest activation magnitudes seen."""
    free()
    model = build_model(mode, lora=False, grad_ckpt=False, verbose=False)
    model.eval()

    records: dict[str, tuple[float, bool]] = {}
    first_bad: list[str] = []

    def hook(mod_name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) and len(out) and torch.is_tensor(out[0]) else out
            if not torch.is_tensor(t) or not t.is_floating_point():
                return
            tf = t.detach().float()
            mx = tf.abs().max().item()
            ok = bool(torch.isfinite(tf).all().item())
            prev = records.get(mod_name, (0.0, True))
            records[mod_name] = (max(prev[0], mx), prev[1] and ok)
            if not ok and not first_bad:
                first_bad.append(mod_name)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]

    losses: list[float] = []
    try:
        with torch.no_grad():
            for r in rows:
                batch, st = collate([r])
                if st["empty"]:
                    continue
                with autocast_ctx(mode):
                    out = model(**to_device(batch))
                losses.append(out.loss.detach().float().item())
    finally:
        for h in handles:
            h.remove()

    peak = vram_peak_gb()
    del model
    free()

    hot = sorted(records.items(), key=lambda kv: -kv[1][0])[:topk]
    return {
        "mode": name,
        "losses": losses,
        "finite": len(losses) > 0 and all(math.isfinite(x) for x in losses),
        "first_bad": first_bad[0] if first_bad else None,
        "n_bad": sum(1 for v in records.values() if not v[1]),
        "hot": hot,
        "peak_gb": peak,
    }


def stage_forward(rows, collate) -> dict:
    print(RULE)
    print("PROBE 1 -- forward numerics (no optimizer)")
    print(RULE)
    results = {}
    for name, mode in available_modes().items():
        print(f"\n[{name}]")
        try:
            res = probe_forward(name, mode, rows, collate)
            results[name] = res
            ls = [round(x, 3) if math.isfinite(x) else x for x in res["losses"]]
            print(f"  losses            : {ls}")
            print(f"  all finite        : {res['finite']}")
            print(f"  non-finite modules: {res['n_bad']}  first: {res['first_bad']}")
            print(f"  peak vram         : {res['peak_gb']:.2f} GB")
            print("  hottest activations (max |x|):")
            for n, (mx, ok) in res["hot"]:
                flag = "NONFINITE" if not ok else "         "
                print(f"    {flag} {mx:12.1f}  {n}")
        except torch.cuda.OutOfMemoryError:
            results[name] = {"mode": name, "finite": False, "error": "OOM"}
            print("  OOM -- does not fit in this GPU (not a numerical verdict)")
            free()
        except Exception as e:                       # noqa: BLE001
            results[name] = {"mode": name, "finite": False, "error": repr(e)[:160]}
            print(f"  FAILED: {repr(e)[:160]}")
            free()
    return results


# --------------------------------------------------------------------------
# probe 2 -- dataset
# --------------------------------------------------------------------------

def stage_data(rows, processor, safe_mode_name: str, safe_mode: Mode,
               n_gpu: int = 60) -> dict:
    print()
    print(RULE)
    print("PROBE 2 -- dataset")
    print(RULE)

    print("\n2a. truncation cliff (tokenizer only)")
    print(f"{'max_len':>8s} {'empty (0 sup tok)':>20s} {'sup p50':>9s} {'sup p10':>9s} {'seq p95':>9s}")
    print("-" * 62)
    cliff = {}
    for L in (768, 1024, 1280, 1536, 2048):
        c = Collator(processor, max_len=L)
        sup, seqs, empty = [], [], 0
        for r in rows:
            _, st = c([r])
            sup.append(st["supervised"][0])
            seqs.append(st["seq_len"])
            empty += st["empty"]
        sup.sort(); seqs.sort()
        cliff[L] = empty
        print(f"{L:>8d} {empty:>12d} ({empty/len(rows)*100:5.1f}%) "
              f"{sup[len(sup)//2]:>9d} {sup[int(len(sup)*.1)]:>9d} {seqs[int(len(seqs)*.95)]:>9d}")
    print("\n  A non-zero 'empty' column means those rows make cross-entropy")
    print("  compute 0/0 = NaN. That is the trap the guard in train.py closes.")

    print(f"\n2b. per-sample forward loss on [{safe_mode_name}] "
          f"(max_len={CFG.max_len}, {n_gpu} samples)")
    collate = Collator(processor, max_len=CFG.max_len)
    free()
    model = build_model(safe_mode, lora=False, grad_ckpt=False, verbose=False)
    model.eval()

    losses, bad, vis = [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows[:n_gpu]):
            batch, st = collate([r])
            if st["empty"]:
                bad.append((i, r["image"], "zero supervised tokens"))
                continue
            with autocast_ctx(safe_mode):
                loss = model(**to_device(batch)).loss.detach().float().item()
            losses.append(loss)
            vis.append(st["n_vision_tok"])
            if not math.isfinite(loss):
                bad.append((i, r["image"], f"loss={loss}"))
    del model
    free()

    ls = sorted(losses)
    print(f"  samples scanned     : {len(losses)}")
    print(f"  loss min/p50/p95/max: {ls[0]:.3f} / {ls[len(ls)//2]:.3f} / "
          f"{ls[int(len(ls)*.95)]:.3f} / {ls[-1]:.3f}")
    print(f"  vision tokens/sample: {min(vis)}-{max(vis)}")
    print(f"  problem samples     : {len(bad)}")
    for x in bad[:8]:
        print("    ", x)
    clean = not bad and all(math.isfinite(x) for x in losses)
    print(f"\n  DATASET_IS_CLEAN = {clean}")
    return {"cliff": cliff, "clean": clean, "losses": losses}


# --------------------------------------------------------------------------
# probe 3 -- micro training
# --------------------------------------------------------------------------

def micro_train(name: str, mode: Mode, rows, collate, steps: int = 30,
                accum: int = 4, lr: float = 1e-4, guard_empty: bool = True) -> dict:
    """Real optimizer steps with a per-step sentinel on loss, grads, weights."""
    free()
    set_seed()
    model = build_model(mode, lora=True, verbose=False)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5))
    scaler = torch.amp.GradScaler("cuda", enabled=mode.needs_scaler)
    guard = NaNGuard()

    log: list[dict] = []
    failure = None
    it = iter(rows)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        acc, used = 0.0, 0

        for _ in range(accum):
            try:
                r = next(it)
            except StopIteration:
                it = iter(rows)
                r = next(it)

            batch, st = collate([r])
            if st["empty"]:
                guard.record_skip("empty", step)
                if guard_empty:
                    continue                          # the fix

            with autocast_ctx(mode):
                loss = model(**to_device(batch)).loss / accum
            lv = loss.detach().float().item()
            if not math.isfinite(lv):
                failure = {
                    "stage": "loss", "step": step, "value": lv,
                    "supervised": st["supervised"], "seq_len": st["seq_len"],
                    "note": ("empty-label batch (0/0)" if st["empty"]
                             else "finite labels but loss non-finite"),
                }
                break
            scaler.scale(loss).backward()
            acc += lv
            used += 1

        if failure:
            break
        if used == 0:
            continue

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, CFG.max_grad_norm)
        if not torch.isfinite(gnorm):
            guard.record_skip("nan_grad", step)
            scaler.step(opt); scaler.update(); sched.step()
            if guard.stalled:
                failure = {"stage": "grad", "step": step,
                           "note": f"{guard.consecutive_skips} consecutive non-finite grads"}
                break
            log.append({"step": step, "loss": acc, "gnorm": float("inf"), "skipped": True})
            continue

        scaler.step(opt); scaler.update(); sched.step()
        guard.record_ok()

        bad = NaNGuard.nonfinite_params(model)
        if bad:
            failure = {"stage": "weights", "step": step, "value": bad[:3],
                       "note": "adapter weights non-finite after optimizer step"}
            break

        log.append({"step": step, "loss": acc, "gnorm": float(gnorm), "skipped": False})

    peak = vram_peak_gb()
    del model, opt, scaler
    free()

    good = [e for e in log if not e["skipped"]]
    return {
        "mode": name, "survived": failure is None, "failure": failure,
        "peak_gb": peak, "steps_done": len(log),
        "skipped": sum(1 for e in log if e["skipped"]),
        "empty_seen": guard.skipped_empty,
        "first_loss": good[0]["loss"] if good else None,
        "last_loss": good[-1]["loss"] if good else None,
    }


def stage_micro(rows, collate, steps: int, guard_empty: bool) -> dict:
    print()
    print(RULE)
    print(f"PROBE 3 -- micro training ({steps} steps, "
          f"max_len={collate.max_len}, guard_empty={guard_empty})")
    print(RULE)
    results = {}
    for name, mode in available_modes().items():
        print(f"\n[{name}]")
        try:
            res = micro_train(name, mode, rows, collate,
                              steps=steps, guard_empty=guard_empty)
        except torch.cuda.OutOfMemoryError:
            res = {"mode": name, "survived": False, "peak_gb": vram_peak_gb(),
                   "failure": {"stage": "oom", "note": "out of memory"}}
            free()
        except Exception as e:                       # noqa: BLE001
            res = {"mode": name, "survived": False, "peak_gb": 0.0,
                   "failure": {"stage": "error", "note": repr(e)[:160]}}
            free()
        results[name] = res
        if res["survived"]:
            print(f"  SURVIVED {res['steps_done']} steps | "
                  f"loss {res['first_loss']:.4f} -> {res['last_loss']:.4f} | "
                  f"{res['empty_seen']} empty-label seen | peak {res['peak_gb']:.2f} GB")
        else:
            print(f"  FAILED: {res['failure']}")
    return results


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

def verdict(fwd: dict, micro: dict) -> str | None:
    print()
    print(RULE)
    print("VERDICT")
    print(RULE)
    print(f"{'mode':20s} {'fwd':>6s} {'train':>7s} {'loss start':>11s} "
          f"{'loss end':>9s} {'peak GB':>8s}  failure")
    print("-" * 92)
    for name in micro:
        f = fwd.get(name, {})
        m = micro[name]
        fok = "ok" if f.get("finite") else "BAD"
        if m["survived"]:
            print(f"{name:20s} {fok:>6s} {'YES':>7s} {m['first_loss']:>11.4f} "
                  f"{m['last_loss']:>9.4f} {m['peak_gb']:>8.2f}  -")
        else:
            fl = m.get("failure") or {}
            print(f"{name:20s} {fok:>6s} {'NO':>7s} {'-':>11s} {'-':>9s} "
                  f"{m.get('peak_gb', 0):>8.2f}  {fl.get('stage')}"
                  f"@{fl.get('step')}: {fl.get('note')}")

    survivors = [n for n, m in micro.items() if m["survived"]]
    best = next((n for n in MODE_PREFERENCE if n in survivors), None)
    print(f"\nsurvivors : {survivors}")
    print(f"recommended: {best}")
    print("\nNote: an 'oom' failure is a memory verdict, not a numerical one.")
    print("Only loss / grad / weights stages indicate a NaN problem.")
    return best


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "forward", "data", "micro"], default="all")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=200,
                    help="rows used by the dataset probe")
    ap.add_argument("--break-it", action="store_true",
                    help="reproduce the original failure: uncompacted targets, "
                         "max_len=1024, empty-label guard OFF")
    args = ap.parse_args()

    gpu = gpu_report()
    print(RULE)
    print(f"GPU        : {gpu['name']}")
    print(f"capability : {gpu['capability']}   bf16 native: {gpu['bf16']}")
    print(f"VRAM       : {gpu['vram_gb']:.2f} GB")
    print(RULE)
    if not gpu["bf16"]:
        print("No native bf16 -- fp16 ViT overflow is a live risk. Probe 1 matters.")
    else:
        print("Native bf16 available -- the fp16 overflow failure mode does not apply.")

    compact = not args.break_it
    max_len = CFG.max_len
    guard_empty = not args.break_it
    if args.break_it:
        print("\n*** --break-it: uncompacted targets, guard OFF. Expect NaN. ***")

    set_seed()
    processor = build_processor()
    collate = Collator(processor, max_len=max_len)

    print(f"\nloading dataset (compact_targets={compact}) ...")
    train = load_split(CFG.train_jsonl, compact=compact)
    print(f"  {len(train)} train rows")
    sample = random.sample(train, min(args.samples, len(train)))

    fwd, micro = {}, {}
    if args.stage in ("all", "forward"):
        fwd = stage_forward(sample[:5], collate)

    safe_name = next((n for n in MODE_PREFERENCE
                      if n in fwd and fwd[n].get("finite")), None)
    modes = available_modes()
    if safe_name is None:
        safe_name = next(iter(modes))
        if args.stage in ("all", "forward"):
            print(f"\nWARNING: no mode passed the forward probe; falling back to {safe_name}")

    if args.stage in ("all", "data"):
        stage_data(sample, processor, safe_name, modes[safe_name])

    if args.stage in ("all", "micro"):
        micro = stage_micro(sample, collate, args.steps, guard_empty)

    if fwd and micro:
        best = verdict(fwd, micro)
        if best is None:
            print("\nNo configuration survived. Do not start training.")
            return 1
        print(f"\nNext: python -m finetune.train --mode {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
