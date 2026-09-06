# Local QLoRA fine-tune — Qwen2.5-VL-3B on VRSBench (6 GB VRAM)

Plain `.py` modules, no notebook, no TRL, no Unsloth. An explicit training loop
so every NaN is attributable to a specific tensor at a specific step.

```
finetune/
  common.py       config, target compaction, collator, model builder, NaN guards
  warmup.py       three-stage triage: quantization vs dataset vs model
  train.py        the fine-tune itself
  compare.py      base vs tuned, side by side
  fetch_model.py  pre-download the base weights
```

## Run order

```powershell
$env:PYTHONPATH="."
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

.venv\Scripts\python.exe -m finetune.fetch_model          # once, ~7 GB
.venv\Scripts\python.exe -m finetune.warmup --steps 15    # triage
.venv\Scripts\python.exe -m finetune.train --max-steps 400
.venv\Scripts\python.exe -m finetune.compare --adapter adapters\qwen25vl-vrsbench-qlora\final
```

## Why this fits in 6 GB

The binding constraint is not the weights — it is the **logit tensor**
(vocab 151936). Two changes make the difference:

**1. Target compaction** (`compact_target` in `common.py`). The stored answers
use `indent=4` pretty-printing, and `obj_corner` is the same box as `obj_coord`
written as four polygon corners instead of xyxy — 8 floats of pure redundancy.
Dropping it, plus the always-derivable flags, plus compact separators:

| target format | p50 tok | p95 tok | p95 + 324 vision tok |
|---|---|---|---|
| as stored | 689 | 1297 | 1651 |
| **compacted (used here)** | **242** | **362** | **716** |

No information is lost. Measured on the real data: 50% character reduction,
and every kept key (`caption`, `objects`, `qa_pairs`) survives.

**2. `max_len = 1024`.** Because p95 of the full sequence is ~716, essentially
nothing is truncated — measured `empty = 0` across 400 sampled rows at 768,
1024 and 1536. That closes the `0/0 → NaN` trap by construction rather than
relying on the runtime guard.

Resulting budget:

| item | GB |
|---|---|
| NF4 4-bit base weights | ~1.95 |
| LoRA r=16, 7 proj types, 8-bit paged AdamW | ~0.30 |
| logit tensor @ seq 1024 (bf16 + fp32 CE copy + grad) | ~1.16 |
| activations w/ gradient checkpointing | ~0.7 |
| CUDA context + fragmentation | ~0.4 |
| **total** | **~4.5 / 6.0** |

Unquantized fp16/bf16 needs 7.5 GB of weights alone, so NF4 is the only base
format that fits. `available_modes()` reflects that.

## Why bf16 on this card

RTX 4050 is Ada, **sm_89 → native bf16**. The fp16 ViT-overflow failure mode
(activations past 65504 → `inf`) does not apply. `nf4_bf16` is the standard,
well-trodden QLoRA recipe and is the expected winner; `warmup.py` verifies it
rather than assuming it.

## What was wrong before

Both defects sat in `src/`, and both are fixed here.

**Pure-fp16 AdamW.** `src/low_vram.py` loaded weights in fp16 while
`src/train_qwen_vl.py` set `fp16=False, bf16=False` — no autocast, no
`GradScaler`, so AdamW did its update arithmetic in fp16. Its `eps=1e-8` is
below fp16's smallest normal (~6.1e-5), flushes toward zero, and
`m/(sqrt(v)+eps)` divides by ~0 → `inf` → `NaN` adapters. Here, trainable
params are forced to fp32 and a `GradScaler` is used whenever the math is fp16.

**Labels.** `src/train_qwen_vl.py` used `labels = input_ids.clone()`, which
supervises pad tokens, the prompt, *and* `<|image_pad|>` — asking the model to
predict image placeholders. `Collator` masks all three and reports the
supervised-token count.

**Truncation.** `max_seq_length=1024` on the *uncompacted* targets starved 51%
of samples; a completion-only loss on a fully-truncated sample computes `0/0`.
Reproduce it deliberately with `python -m finetune.warmup --break-it`.

**Learning rate.** `3e-6` is a full-fine-tune LR; for LoRA r=16 it is ~30×
too small. Default here is `1e-4` with cosine decay and 20 warmup steps.

## NaN policy

Three tiers, because the failure modes differ:

*Recoverable — skip the micro-batch, keep training:*
- zero supervised tokens after truncation
- non-finite loss on one sample
- non-finite gradient norm (in fp16 the scaler already skips the step)

*Terminal — stop before the checkpoint is poisoned:*
- `NonFiniteWeights` — adapter tensors themselves non-finite
- `TrainingStalled` — 12 consecutive unusable micro-batches

A last-known-good adapter snapshot is kept in memory and refreshed every 25
steps. On a terminal fault it is restored and written to `last-good/`, so a
late NaN never costs the whole run.

`compare.py` refuses to report on an adapter containing non-finite LoRA
tensors — that corruption is exactly what produced the endless `!` output
(argmax over an all-NaN logit vector returns index 0, which is `!` in Qwen's
vocab).

## Reading `compare.py`

Base Qwen2.5-VL-3B answers "Describe the image in detail." with a prose
paragraph — it has no reason to emit the VRSBench schema. A working adapter
answers with parseable JSON. The script loads the 4-bit base **once** and
toggles the adapter with `disable_adapter()`, so both answers come from one
model in one allocation and the adapter is the only variable.

Headline metric is the valid-JSON rate, printed as a summary. Expect format
adherence within ~200–400 steps; factual and localisation accuracy needs far
more and is not what this run delivers.

## Windows notes

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation.
- 6 GB is shared with the desktop compositor — close browsers before a run.
- Uses `sdpa` attention, not flash-attn (painful to build on Windows).
- Pinned to `transformers==4.56.1` / `peft==0.17.1`; transformers v5
  restructured the Qwen2.5-VL internals.
