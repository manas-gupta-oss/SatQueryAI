# satquery — satellite imagery → structured JSON reports

Two fine-tuned specialist agents built on **Qwen2.5-VL-3B-Instruct**, both
trained with QLoRA on a **6 GB RTX 4050 laptop**, both emitting parseable JSON
for the PDF report layer.

| agent | input | answers | adapter |
|---|---|---|---|
| **single-image** | one scene | caption, objects + boxes, QA pairs | `adapters/report-adapter-v2/` |
| **bi-temporal** | two dated images | what changed, where, how much | `adapters/bitemporal-adapter/` |

Both adapters attach to **one** 4-bit base model — the backend holds 3.75B
params once (**2.65 GB VRAM**), not twice.

---

## Quick start (backend / frontend integration)

```powershell
uv venv --python 3.12 .venv-unsloth
uv pip install --python .venv-unsloth\Scripts\python.exe torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-unsloth\Scripts\python.exe -r requirements-unsloth.txt
```

```python
from generate_report_v2 import SatelliteReporter

rep = SatelliteReporter()                        # load ONCE at startup (~40 s)
single = rep.analyze_image("scene.png")
change = rep.analyze_pair("2019.png", "2024.png")   # first arg = EARLIER image
```

> **Construction is expensive, calls are cheap.** In a Flask/FastAPI backend,
> build one `SatelliteReporter` at startup — never per request.

CLI:

```powershell
.venv-unsloth\Scripts\python.exe generate_report_v2.py --image satellite.png --out report.json --pretty
.venv-unsloth\Scripts\python.exe generate_report_v2.py --before a.png --after b.png --out change.json --pretty
```

## Output contract

Both methods return the **same envelope**, so the PDF layer has one shape to
handle. `task` says which agent ran and therefore which `analysis` fields are
populated.

```json
{
  "status": "ok",            // or "error" -- ALWAYS check this; calls never raise
  "error": null,
  "task": "single_image",    // or "bitemporal_change"
  "images": [{"filename","path","width","height","role"}],
  "model": {"base","adapter","backend","trained_steps"},
  "generated_at": "2026-09-05T14:52:03Z",
  "inference_seconds": 4.1,
  "analysis": { ... },
  "summary": { ... }
}
```

`analysis` by task:

- **single_image** — `caption`, `objects[]` (`obj_cls`, `obj_position`, `obj_size`,
  `bbox_normalized`), `qa_pairs[]`
- **bitemporal_change** — `change_detected`, `change_summary`, `changed_classes[]`,
  `change_regions[]` (`class`, `size`, `bbox_normalized`), `change_extent`

**Use `bbox_normalized` for drawing**, not the raw coordinate field. It is
clamped to 0–1; the raw values can fall outside the frame.

`caption` is always a string and the list fields are always lists, even on
failure — no existence checks needed downstream.

## Measured results

**Bi-temporal**, on a balanced 120-pair validation set (60 changed / 60 unchanged,
so chance is exactly 50%):

| | base model | fine-tuned |
|---|---|---|
| balanced accuracy | 50.0% | **94.2%** |
| recall on changed | 100% | 91.7% |
| recall on unchanged | 0% | 96.7% |
| valid schema JSON | 0/120 | **120/120** |
| changed-class F1 | — | 88.6% |
| BLEU-4 | 1.1 | **51.0** |
| METEOR | 17.8 | **69.3** |
| ROUGE-L | 19.8 | **69.3** |
| CIDEr-D | 2.5 | **123.6** |
| BERTScore F1 | 20.4 | **68.1** |

The base model scores exactly chance because it reports change on **every**
pair, including unchanged ones — it never says "no difference".

> Those caption scores are the **all-pairs** average, and half those pairs are
> unchanged, where LEVIR-CC reuses the same few sentences — easy n-grams. On
> **changed pairs only** the fine-tuned model scores BLEU-4 24.6, METEOR 41.5,
> ROUGE-L 41.2, CIDEr-D 41.3, BERTScore 49.8 (base: 1.9 / 18.7 / 22.4 / 2.9 /
> 26.5). Quote that set when comparing against published LEVIR-CC work.

**Single-image**, on 60 validation images:

| | base model | fine-tuned |
|---|---|---|
| valid schema JSON | 0% | **100%** |
| object-class F1 (set) | — | 63.3% |
| object-class F1 (multiset) | — | 57.1% |
| object-count exact match | — | 55% |
| BLEU-4 | 2.0 | **12.1** |
| METEOR | 26.0 | **37.1** |
| ROUGE-L | 17.3 | **33.7** |
| CIDEr-D | 0.6 | **25.1** |
| BERTScore F1 | 15.0 | **44.4** |

Set F1 asks whether the right *kinds* of object were named; multiset F1 also
requires the right *number* of each, so it absorbs the counting weakness below.

Full tables with bootstrap 95% confidence intervals: **[results/METRICS.md](results/METRICS.md)**
(machine-readable in `results/metrics.json`). Every interval sits clear of the
base column, so these gaps are significant rather than sampling noise.

Honest limits: object **counting** is unreliable (five ships may be read as
two), and caption wording differs from reference even when substantively
correct. Claim structured extraction and change detection; do not claim
counting accuracy.

## Orchestrator integration — three rules

**1. Never pass the user's query to the adapters.** Both agents were fine-tuned
on exactly one prompt each (all 5,066 VRSBench rows use `"Describe the image in
detail."`; all 6,815 bi-temporal rows use one change-detection sentence). They
learned *image → schema*, not *question → answer*. Call `analyze_image(path)`
and `analyze_pair(a, b)` with **no `prompt=` argument** so the trained default
is used. Measured effect of overriding it:

| prompt sent | status | objects | boxes |
|---|---|---|---|
| default (trained) | ok | 1 | yes |
| `""` empty | ok | 1 | yes |
| `"How many ships are in this image?"` | **error** | 0 | **none** |
| `"Reply with the single word BANANA."` | **error** | 0 | **none** |

Nothing crashes — the envelope holds and `status` flips to `"error"` — but you
lose the schema and therefore the visual evidence. The user's query belongs in
the orchestrator: use it to route (1 image vs 2), to filter the returned JSON
("how many ships?" is a count over `objects`), and to shape the narrative.

**2. Always check `status`.** It is `"ok"` or `"error"` and calls never raise,
so a missing check fails silently rather than loudly.

**3. Validate before you store.** `validate.py` is stdlib-only, needs no GPU,
and is importable by both the orchestrator and the PDF layer:

```python
from validate import validate

report = reporter.analyze_image("scene.png")
v = validate(report)
if v["severity"] == "reject":
    show(v["user_message"])      # polite and specific, no stack trace
store(report, validation=v)      # store it either way, flagged
```

It checks **self-consistency**, which is what makes it a real hallucination
signal: position contradicting the bounding box, a caption claiming three ships
when two were extracted, `change_detected: true` alongside a "nothing changed"
summary, `changed_classes` disagreeing with the regions, labels outside the
trained vocabulary, duplicate or zero-area boxes.

Measured on the 180 cached predictions:

| | ok | warn | **reject** |
|---|---|---|---|
| single-image, tuned | 53 | 7 | **0** |
| bi-temporal, tuned | 120 | 0 | **0** |
| single-image, base | 0 | 0 | **60** |
| bi-temporal, base | 0 | 0 | **120** |

Zero false rejects on good output, 100% rejection of the base model's
unstructured prose. The 7 warnings are genuine: five captions that contradict
their own object count, and three objects labelled `building`, which is not one
of VRSBench's 26 classes.

> Say **"self-consistency and hallucination detection"**, not "we verify the
> answer is correct". This layer catches the model contradicting itself; nothing
> downstream of the model can confirm a claim against the actual image. The
> first statement is true and defensible, the second is not.

## Paper-style metrics

`metrics.py` computes the caption metrics reviewers expect to see — BLEU-1..4,
METEOR, ROUGE-1/2/L, CIDEr-D, and optionally BERTScore — alongside the task
metrics above, with **bootstrap 95% confidence intervals** so the base-vs-tuned
gaps can be called significant rather than merely asserted.

```powershell
# 1. GPU pass: run both models, cache predictions to results/predictions/
.venv-unsloth\Scripts\python.exe metrics.py generate --task bitemporal --both
.venv-unsloth\Scripts\python.exe metrics.py generate --task single     --both

# 2. CPU pass: score them. Runs in EITHER venv -- no model is loaded.
.venv\Scripts\python.exe metrics.py score --markdown results/METRICS.md --json results/metrics.json
```

Splitting it in two means the expensive GPU pass runs once and can be re-scored
for free when a metric is added. Everything is implemented in-file — no
coco-caption checkout, no Java, no nltk required.

Reading the numbers:

- The **base model is scored on its raw prose**. It emits no JSON, so pulling a
  caption field would zero it out for formatting rather than content. This is
  deliberately generous to the baseline; its formatting failure is reported
  separately as `valid schema JSON`.
- **CIDEr document frequencies come from the evaluation set itself** (as in the
  original paper), so the absolute value depends on set size. Compare columns
  against each other, not against a published number.
- On the **unchanged-pairs** subset CIDEr collapses to ~0 for every model,
  because LEVIR-CC gives those pairs near-identical captions and CIDEr's idf
  term gives no credit for n-grams everyone shares. That is CIDEr working as
  intended; read the change-detection rows there instead.
- **METEOR** runs exact + stem matching. Install `nltk` with the wordnet corpus
  and it defers to the real implementation with the synonym stage; the report
  prints which backend ran. The built-in fallback scores slightly low, never
  high.

BERTScore uses **roberta-large** and is **rescaled with the published baseline**,
which is what makes it comparable to numbers in papers: raw BERTScore cosine
similarities sit near 0.85 even for unrelated sentences, so without rescaling
every model looks equally good. The report names the model and warns in-line if
a baseline was unavailable and rescaling was skipped.

It is opt-in (`--bertscore`) because it downloads roberta-large (~1.4 GB):

```powershell
.venv\Scripts\python.exe -m pip install bert-score
.venv\Scripts\python.exe metrics.py score --bertscore
```

## Repo layout

```
generate_report_v2.py     <- the deliverable: both agents, one model
generate_report.py           v1, single-agent, plain-PEFT stack (fallback)
validate.py               self-consistency / hallucination gate (stdlib only)
metrics.py                BLEU / METEOR / ROUGE / CIDEr / BERTScore + CIs
results/                  cached predictions and the scored tables

finetune/                 single-image agent (VRSBench)
  common.py  train.py  train_unsloth.py  evaluate.py  evaluate_unsloth.py
  export_adapter.py  warmup.py  compare.py  compare3.py  fetch_model.py

bitemporal/               change-detection agent (LEVIR-CC)
  common.py  prepare_data.py  train.py  train_unsloth.py
  evaluate.py  evaluate_unsloth.py

adapters/                 only the bf16 exports are committed (57 MB each)
```

## Two environments, and why

| | `.venv-unsloth` | `.venv` |
|---|---|---|
| purpose | **inference + training** | offline tooling only |
| stack | transformers 5.5, peft 0.20, unsloth | transformers 4.56.1, peft 0.17.1 |
| can load shipped adapters | **yes** | no |

The shipped adapters use peft-0.20 format with **regex `target_modules`**, which
peft 0.17.1 rejects. Anything touching them needs `.venv-unsloth`. The plain
stack survives only for `bitemporal/prepare_data.py` (needs scipy),
`finetune/export_adapter.py`, and the v1 fallback.

## Reproducing

```powershell
# data prep (bi-temporal) -- plain venv, needs scipy
$env:PYTHONPATH="."
.venv\Scripts\python.exe -m bitemporal.prepare_data

# training -- unsloth venv
.venv-unsloth\Scripts\python.exe -m finetune.train_unsloth   --max-steps 600 --micro-batch 2
.venv-unsloth\Scripts\python.exe -m bitemporal.train_unsloth --max-steps 900 --micro-batch 2

# evaluation
.venv-unsloth\Scripts\python.exe -m bitemporal.evaluate_unsloth --adapter adapters/qwen25vl-levircc-unsloth/step-900

# export a checkpoint for shipping (fp32 120 MB -> bf16 57 MB, under GitHub's limit)
.venv\Scripts\python.exe -m finetune.export_adapter --src <checkpoint> --dst <export>
```

Datasets are not committed (~5 GB). VRSBench goes in `data/vrsbench/`;
[LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) images in
`data/images/{train,val,test}/{A,B}/` with `data/LevirCCcaptions.json`, and the
[LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI) change masks in
`data/LEVIR-MCI-dataset/images/{split}/label_rgb/`.

> The LEVIR-MCI readme states road = blue `(0,0,255)`. It is wrong — blue never
> occurs; road is **yellow `(255,255,0)`**. `prepare_data.py` handles this;
> trusting the readme silently discards every road change.

## Training notes

All runs used a five-layer NaN guard (empty-label batch, non-finite loss,
non-finite gradient, adapter-weight audit every 10 steps with rollback, and a
divergence early-warning), plus per-step CSV logging. **Zero NaN events across
every run.**

Earlier NaN failures came from two causes, both fixed: fp16 AdamW (its
`eps=1e-8` underflows below fp16's smallest normal ~6.1e-5, so the update
divides by ~0), and `max_seq_length=1024` truncating 51% of VRSBench targets to
zero supervised tokens, making cross-entropy compute 0/0. The Qwen2.5-VL vision
tower also emits activations near 39,680 against fp16's 65,504 ceiling — bf16
avoids that entirely.
