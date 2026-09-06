r"""Unsloth variant of the bi-temporal fine-tune.

    # MUST use the isolated unsloth venv -- it needs transformers 5.x, which
    # would break the pinned 4.56.1 stack the other agents were trained on.
    .venv-unsloth\Scripts\python.exe -m bitemporal.train_unsloth --max-steps 1200

Adapters go to adapters/qwen25vl-levircc-unsloth/ so the plain-PEFT run at
adapters/qwen25vl-levircc-qlora/ is never touched.

What differs from bitemporal/train.py
-------------------------------------
Only the model backend. Unsloth's fused Triton kernels replace the plain
transformers forward, and `FastVisionModel.get_peft_model` replaces PEFT's
`get_peft_model`. The dataset, the Collator (imported unchanged from
bitemporal.common), the loss, the optimizer, the LR schedule and all five NaN
guards are identical -- so a difference in the result is attributable to the
backend rather than to the data pipeline.

Measured in the probe: 2.68 GB peak vs 3.38 GB for the plain stack, ~20% less.
That headroom is what makes --micro-batch > 1 possible here; sequences are only
~250 tokens, so batch 1 leaves the GPU badly underutilised.

NaN guards (same five as bitemporal/train.py)
---------------------------------------------
  recoverable: zero-supervised-token batch | non-finite loss | non-finite grad
  terminal   : adapter weights non-finite  | too many consecutive skips
  warning    : loss departing from its 50-step trailing median
"""

from __future__ import annotations

# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from bitemporal.common import (  # noqa: E402
    CFG,
    Collator,
    NaNGuard,
    NonFiniteWeights,
    TrainingStalled,
    free,
    gpu_report,
    load_split,
    set_seed,
    to_device,
    vram_peak_gb,
)

RULE = "=" * 72
UNSLOTH_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
OUT_DEFAULT = CFG.out_dir.parent / "qwen25vl-levircc-unsloth"


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, collate, rows, limit: int) -> float:
    FastVisionModel.for_inference(model)
    total, count = 0.0, 0
    for r in rows[:limit]:
        batch, st = collate([r])
        if st["empty"]:
            continue
        loss = model(**to_device(batch)).loss.detach().float().item()
        if math.isfinite(loss):
            total += loss
            count += 1
    FastVisionModel.for_training(model)
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
    print(f"backend: UNSLOTH ({UNSLOTH_MODEL})")
    print(RULE)

    set_seed()
    print("loading model ...")
    model, processor = FastVisionModel.from_pretrained(
        UNSLOTH_MODEL,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,       # LM only, matching the plain run
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
        lora_dropout=0.0, bias="none", random_state=0,
    )
    FastVisionModel.for_training(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA r={CFG.lora_r} | {n_train/1e6:.1f}M trainable | "
          f"resident {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    collate = Collator(processor, max_len=args.max_len)

    print("loading dataset ...")
    train_rows = load_split(CFG.train_jsonl)
    val_rows = load_split(CFG.val_jsonl)
    n_ch = sum(r.get("changeflag", 0) for r in train_rows)
    print(f"  train {len(train_rows)} ({n_ch} changed / {len(train_rows)-n_ch} unchanged)"
          f" | val {len(val_rows)}")

    probe, pst = collate(train_rows[: args.micro_batch])
    print(f"  probe: seq_len={pst['seq_len']} vision_tok={pst['n_vision_tok']} "
          f"images={pst['n_images']} supervised={pst['supervised']}")
    if pst["n_images"] != 2 * args.micro_batch:
        print(f"  !! expected {2*args.micro_batch} images"); return 2

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, betas=(0.9, 0.999),
                                       eps=1e-8, weight_decay=CFG.weight_decay)
        opt_name = "PagedAdamW8bit"
    except Exception:                                # noqa: BLE001
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=CFG.weight_decay)
        opt_name = "AdamW"
    total_steps = args.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, CFG.warmup_steps, total_steps))
    guard = NaNGuard()

    eff = args.micro_batch * args.grad_accum
    print(f"  optimizer: {opt_name}")
    print(f"  {total_steps} steps x {args.micro_batch} micro x {args.grad_accum} accum "
          f"= effective batch {eff}, {total_steps*eff} samples "
          f"({total_steps*eff/len(train_rows):.2f} epochs)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train_log.csv"
    csv_fh = csv_path.open("w", encoding="utf-8")
    csv_fh.write("step,loss,grad_norm,lr,elapsed_s\n")
    val_csv = out_dir / "val_log.csv"
    val_fh = val_csv.open("w", encoding="utf-8")
    val_fh.write("step,val_loss\n")
    print(f"  step log : {csv_path}")
    print(RULE)

    meta = {
        "backend": "unsloth", "unsloth_model": UNSLOTH_MODEL,
        "max_len": args.max_len, "lr": args.lr,
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "effective_batch": eff, "max_steps": total_steps,
        "lora_r": CFG.lora_r, "lora_alpha": CFG.lora_alpha,
        "task": "bitemporal_change_detection", "dataset": "LEVIR-CC",
        "images_per_sample": 2,
    }

    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0

    def next_rows(k):
        nonlocal cursor
        out = []
        for _ in range(k):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            out.append(train_rows[order[cursor]])
            cursor += 1
        return out

    last_good = snapshot_adapter(model)
    last_good_step = 0
    history: list[tuple[int, float, float]] = []
    recent: deque[float] = deque(maxlen=50)
    val_history: list[tuple[int, float]] = []
    diverge_warnings = 0
    t0 = time.time()
    step = 0
    fault: Exception | None = None

    try:
        while step < total_steps:
            opt.zero_grad(set_to_none=True)
            acc, used = 0.0, 0

            for _ in range(args.grad_accum):
                rows = next_rows(args.micro_batch)
                try:
                    batch, st = collate(rows)
                except Exception as e:               # noqa: BLE001
                    guard.record_skip("empty", step, f"collate failed: {e!r}")
                    continue

                # GUARD 1: nothing to supervise -> cross-entropy would do 0/0
                if st["empty"] == len(rows):
                    guard.record_skip("empty", step, rows[0].get("filename", "?"))
                    continue

                loss = model(**to_device(batch)).loss / args.grad_accum

                # GUARD 2: non-finite loss on this micro-batch
                lv = loss.detach().float().item()
                if not math.isfinite(lv):
                    guard.record_skip("nan_loss", step, rows[0].get("filename", "?"))
                    continue

                loss.backward()
                acc += lv
                used += 1

            if used == 0:
                if guard.stalled:
                    raise TrainingStalled(
                        f"{guard.consecutive_skips} consecutive unusable micro-batches")
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(params, CFG.max_grad_norm)

            # GUARD 3: non-finite gradients
            if not torch.isfinite(gnorm):
                guard.record_skip("nan_grad", step)
                opt.zero_grad(set_to_none=True)
                sched.step()
                if guard.stalled:
                    raise TrainingStalled(
                        f"{guard.consecutive_skips} consecutive non-finite gradients")
                continue

            opt.step()
            sched.step()
            guard.record_ok()
            step += 1

            # GUARD 4 (terminal): adapter weights corrupted
            if step % args.nan_check_every == 0:
                bad = NaNGuard.nonfinite_params(model)
                if bad:
                    raise NonFiniteWeights(
                        f"non-finite adapter tensors at step {step}: {bad[:3]}")
                last_good = snapshot_adapter(model)
                last_good_step = step

            # GUARD 5 (early warning): divergence precedes NaN
            recent.append(acc)
            if len(recent) >= 20:
                med = sorted(recent)[len(recent) // 2]
                if acc > max(3.0, 5.0 * med):
                    diverge_warnings += 1
                    print(f"  !! WARNING step {step}: loss {acc:.3f} vs trailing "
                          f"median {med:.3f} ({acc/max(med,1e-6):.1f}x) -- "
                          f"divergence precursor #{diverge_warnings}")

            history.append((step, acc, float(gnorm)))
            csv_fh.write(f"{step},{acc:.6f},{float(gnorm):.6f},"
                         f"{sched.get_last_lr()[0]:.3e},{time.time()-t0:.1f}\n")
            csv_fh.flush()

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"step {step:4d}/{total_steps}  loss {acc:7.4f}  "
                      f"|g| {float(gnorm):7.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el/step:5.2f}s/step  vram {vram_peak_gb():.2f}GB")
            if args.eval_every and step % args.eval_every == 0:
                vl = evaluate(model, collate, val_rows, CFG.eval_samples)
                val_history.append((step, vl))
                val_fh.write(f"{step},{vl:.6f}\n"); val_fh.flush()
                best = min(val_history, key=lambda x: x[1])
                flag = "  <- best so far" if best[0] == step else f"  (best {best[1]:.4f} @ {best[0]})"
                print(f"    val loss: {vl:.4f}{flag}")
            if args.save_every and step % args.save_every == 0:
                save_adapter(model, processor, out_dir / f"step-{step}", meta | {"step": step})

    except (NonFiniteWeights, TrainingStalled) as e:
        fault = e
        print(f"\n!! TERMINAL: {e}")
        print(f"!! rolling back to the step-{last_good_step} snapshot")
        restore_adapter(model, last_good)
        step = last_good_step
    except KeyboardInterrupt:
        print("\n-- interrupted, saving current adapter")
    finally:
        csv_fh.close()
        val_fh.close()

    tag = "final" if fault is None else "last-good"
    save_adapter(model, processor, out_dir / tag, meta | {"step": step, "faulted": bool(fault)})

    el = (time.time() - t0) / 60
    print(RULE)
    print(f"{step} steps in {el:.1f} min | {guard.summary()}")
    print(f"divergence warnings: {diverge_warnings}")
    if history:
        k = min(10, len(history))
        print(f"loss: {sum(h[1] for h in history[:k])/k:.4f} -> "
              f"{sum(h[1] for h in history[-k:])/k:.4f}")
    if val_history:
        best = min(val_history, key=lambda x: x[1])
        print(f"\nval loss by checkpoint:")
        for s, v in val_history:
            print(f"  step {s:5d}: {v:.4f}{'   <- BEST' if s == best[0] else ''}")
        print(f"\nbest checkpoint: adapters/.../step-{best[0]}  (val {best[1]:.4f})")
        if best[0] != step:
            print(f"NOTE: val bottomed at step {best[0]}, not the final step {step}.")
            print("      Prefer that checkpoint over `final`.")
    print(f"\nadapter: {out_dir / tag}")
    print(RULE)
    return 1 if fault else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="rows per forward pass; Unsloth's memory saving makes >1 viable")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=CFG.max_len)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--nan-check-every", type=int, default=10)
    return train(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
