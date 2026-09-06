"""QLoRA fine-tune Qwen2.5-VL-3B on VRSBench inside a 6 GB budget.

    python -m finetune.train                      # defaults: 400 steps, ~40-70 min
    python -m finetune.train --max-steps 150      # quick demo adapter
    python -m finetune.train --mode nf4_bf16      # force a mode
    python -m finetune.train --resume adapters/.../step-200

NaN policy -- three tiers, because the failure modes are genuinely different:

  recoverable, skip the micro-batch
    * zero supervised tokens after truncation  (cross-entropy would do 0/0)
    * non-finite loss on a single sample
    * non-finite gradient norm (GradScaler already skips the step in fp16)

  terminal, stop before the checkpoint is poisoned
    * adapter weights themselves non-finite   -> NonFiniteWeights
    * too many consecutive skips              -> TrainingStalled

The last-known-good adapter is kept in memory and written on a terminal fault,
so a late NaN never costs you the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch

from finetune.common import (
    CFG,
    MODE_PREFERENCE,
    Collator,
    NaNGuard,
    NonFiniteWeights,
    TrainingStalled,
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


def build_optimizer(params, lr: float, weight_decay: float):
    """8-bit AdamW when bitsandbytes provides it -- on a 6 GB card the optimizer
    state is worth reclaiming. Falls back to torch AdamW."""
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(params, lr=lr, betas=(0.9, 0.999),
                                       eps=1e-8, weight_decay=weight_decay)
        return opt, "PagedAdamW8bit"
    except Exception:                                # noqa: BLE001
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=weight_decay)
        return opt, "AdamW(fp32)"


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, collate, rows, mode, limit: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for r in rows[:limit]:
        batch, st = collate([r])
        if st["empty"]:
            continue
        with autocast_ctx(mode):
            loss = model(**to_device(batch)).loss.detach().float().item()
        if math.isfinite(loss):
            total += loss
            count += 1
    model.train()
    return total / max(1, count)


def snapshot_adapter(model) -> dict:
    return {n: p.detach().clone().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, snap: dict) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.data.copy_(snap[n].to(p.device, p.dtype))


def save_adapter(model, processor, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    saved -> {out}")


def train(args) -> int:
    gpu = gpu_report()
    print(RULE)
    print(f"GPU   : {gpu['name']}  ({gpu['capability']}, {gpu['vram_gb']:.2f} GB, "
          f"bf16={gpu['bf16']})")

    modes = available_modes()
    mode_name = args.mode or next(n for n in MODE_PREFERENCE if n in modes)
    if mode_name not in modes:
        print(f"unknown mode {mode_name!r}; available: {list(modes)}")
        return 2
    mode = modes[mode_name]
    print(f"mode  : {mode_name}")
    print(RULE)

    set_seed()
    processor = build_processor()
    collate = Collator(processor, max_len=args.max_len)

    print("loading dataset ...")
    train_rows = load_split(CFG.train_jsonl, compact=True)
    val_rows = load_split(CFG.val_jsonl, compact=True)
    print(f"  train {len(train_rows)} | val {len(val_rows)}")

    print("loading model ...")
    model = build_model(mode, lora=(args.resume is None))
    if args.resume:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"  resumed from {args.resume}")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt, opt_name = build_optimizer(params, args.lr, CFG.weight_decay)
    total_steps = args.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, CFG.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=mode.needs_scaler)
    guard = NaNGuard()

    print(f"  optimizer: {opt_name} | grad_scaler: {mode.needs_scaler}")
    print(f"  {total_steps} steps x {args.grad_accum} accum "
          f"= {total_steps * args.grad_accum} samples")
    print(RULE)

    out_dir = Path(args.out)
    meta = {
        "mode": mode_name, "max_len": args.max_len, "lr": args.lr,
        "grad_accum": args.grad_accum, "max_steps": total_steps,
        "lora_r": CFG.lora_r, "lora_alpha": CFG.lora_alpha,
        "model_id": CFG.model_id, "compact_targets": True,
    }

    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0

    def next_row():
        nonlocal cursor, order
        if cursor >= len(order):
            random.shuffle(order)
            cursor = 0
        r = train_rows[order[cursor]]
        cursor += 1
        return r

    last_good = snapshot_adapter(model)
    last_good_step = 0
    history: list[tuple[int, float, float]] = []
    recent: deque[float] = deque(maxlen=50)
    diverge_warnings = 0
    t0 = time.time()
    step = 0
    fault: Exception | None = None

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train_log.csv"
    csv_fh = csv_path.open("w", encoding="utf-8")
    csv_fh.write("step,loss,grad_norm,lr,elapsed_s\n")
    print(f"  step log: {csv_path}")

    try:
        while step < total_steps:
            opt.zero_grad(set_to_none=True)
            acc, used = 0.0, 0

            for _ in range(args.grad_accum):
                row = next_row()
                try:
                    batch, st = collate([row])
                except Exception as e:               # noqa: BLE001
                    guard.record_skip("empty", step, f"collate failed: {e!r}")
                    continue

                # GUARD 1: truncation left nothing to supervise -> 0/0 = NaN
                if st["empty"]:
                    guard.record_skip("empty", step, Path(row["image"]).name)
                    continue

                with autocast_ctx(mode):
                    loss = model(**to_device(batch)).loss / args.grad_accum

                # GUARD 2: non-finite loss on this sample
                lv = loss.detach().float().item()
                if not math.isfinite(lv):
                    guard.record_skip("nan_loss", step, Path(row["image"]).name)
                    continue

                scaler.scale(loss).backward()
                acc += lv
                used += 1

            if used == 0:
                if guard.stalled:
                    raise TrainingStalled(
                        f"{guard.consecutive_skips} consecutive unusable micro-batches")
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, CFG.max_grad_norm)

            # GUARD 3: non-finite gradients. In fp16 the scaler already skips
            # the step; we only need to notice a run that never recovers.
            if not torch.isfinite(gnorm):
                guard.record_skip("nan_grad", step)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if guard.stalled:
                    raise TrainingStalled(
                        f"{guard.consecutive_skips} consecutive non-finite gradients")
                continue

            scaler.step(opt)
            scaler.update()
            sched.step()
            guard.record_ok()
            step += 1

            # GUARD 4 (terminal): adapter weights themselves corrupted.
            # Cheap enough (29.9M params) to run often; this is the check that
            # stops a corrupted adapter from ever reaching disk.
            if step % args.nan_check_every == 0:
                bad = NaNGuard.nonfinite_params(model)
                if bad:
                    raise NonFiniteWeights(
                        f"non-finite adapter tensors at step {step}: {bad[:3]}")
                last_good = snapshot_adapter(model)
                last_good_step = step

            # GUARD 5 (early warning): divergence precedes NaN. A loss that
            # suddenly departs from its trailing median is the signal you get
            # *before* things become non-finite, so surface it loudly.
            recent.append(acc)
            if len(recent) >= 20:
                med = sorted(recent)[len(recent) // 2]
                if acc > max(3.0, 5.0 * med):
                    diverge_warnings += 1
                    print(f"  !! WARNING step {step}: loss {acc:.3f} vs trailing "
                          f"median {med:.3f} ({acc/max(med,1e-6):.1f}x) -- "
                          f"divergence precursor #{diverge_warnings}")

            history.append((step, acc, float(gnorm)))
            if csv_fh:
                csv_fh.write(f"{step},{acc:.6f},{float(gnorm):.6f},"
                             f"{sched.get_last_lr()[0]:.3e},{time.time()-t0:.1f}\n")
                csv_fh.flush()
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"step {step:4d}/{total_steps}  loss {acc:7.4f}  "
                      f"|g| {float(gnorm):7.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el/step:5.2f}s/step  vram {vram_peak_gb():.2f}GB")
            if args.eval_every and step % args.eval_every == 0:
                vl = evaluate(model, collate, val_rows, mode, CFG.eval_samples)
                print(f"    val loss: {vl:.4f}")
            if args.save_every and step % args.save_every == 0:
                save_adapter(model, processor, out_dir / f"step-{step}", meta | {"step": step})

    except (NonFiniteWeights, TrainingStalled) as e:
        fault = e
        print(f"\n!! TERMINAL: {e}")
        print(f"!! rolling back to the step-{last_good_step} snapshot and saving it")
        restore_adapter(model, last_good)
        step = last_good_step
    except KeyboardInterrupt:
        print("\n-- interrupted, saving current adapter")
    finally:
        if csv_fh:
            csv_fh.close()

    tag = "final" if fault is None else "last-good"
    save_adapter(model, processor, out_dir / tag, meta | {"step": step, "faulted": bool(fault)})

    el = (time.time() - t0) / 60
    print(RULE)
    print(f"{step} steps in {el:.1f} min | {guard.summary()}")
    print(f"divergence warnings: {diverge_warnings} | step log: {csv_path}")
    if history:
        first = sum(h[1] for h in history[:10]) / min(10, len(history))
        last = sum(h[1] for h in history[-10:]) / min(10, len(history))
        print(f"loss: {first:.4f} (first 10) -> {last:.4f} (last 10)")
    if fault is None and val_rows:
        print(f"val loss: {evaluate(model, collate, val_rows, mode, CFG.eval_samples):.4f}")
    print(f"adapter: {out_dir / tag}")
    print(RULE)
    print(f"\nNext: python -m finetune.compare --adapter {out_dir / tag}")
    return 1 if fault else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None, help="nf4_bf16 | nf4_fp32 | nf4_fp16_visfp32 | nf4_fp16")
    ap.add_argument("--max-steps", type=int, default=CFG.max_steps)
    ap.add_argument("--max-len", type=int, default=CFG.max_len)
    ap.add_argument("--grad-accum", type=int, default=CFG.grad_accum)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--out", default=str(CFG.out_dir))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=CFG.log_every)
    ap.add_argument("--eval-every", type=int, default=CFG.eval_every)
    ap.add_argument("--save-every", type=int, default=CFG.save_every)
    ap.add_argument("--nan-check-every", type=int, default=CFG.nan_check_every,
                    help="cadence of the full adapter-weight finiteness audit "
                         "and last-known-good snapshot")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
