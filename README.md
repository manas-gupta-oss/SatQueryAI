# SatQueryAI

Ask a question about satellite imagery in plain English. A router picks the
right specialist model, the specialist analyses the pixels, a validation layer
checks the result against itself, and you get back an answer, the imagery with
the detections drawn on it, and a PDF report.

Everything runs locally on one laptop with a 6 GB GPU.

```
        query + image metadata                      pixels
                 │                                     │
            ┌────▼────┐   validated route      ┌───────▼────────┐
  upload ──▶│  BOSS   │───────────────────────▶│  specialist    │
            │ router  │                        │  worker        │
            └────┬────┘                        └───────┬────────┘
                 │  no compatible worker               │  structured JSON
                 ▼                                     ▼
              reject  ─────────────▶ finalize ◀── self-consistency check
                                        │
                                        ▼
                          answer + overlay + PDF report
```

The router never sees pixels. The specialists never see the query. That
separation is the whole design, and the reasons for it are measured, not
assumed — see [models/MODELS.md](models/MODELS.md).

---

## Quick start

```powershell
# backend  (needs .venv-unsloth -- see backend/README.md)
.venv-unsloth\Scripts\python.exe -m uvicorn backend.app:app --port 8000

# frontend
cd frontend; npm install; npm run dev
```

Then open <http://localhost:5173>.

No GPU? It still runs — the specialists degrade to clearly-labelled stubs and
routing, validation and PDF generation all work. See
[backend/README.md](backend/README.md).

---

## The three layers

### `orchestration/` — routing

A LangGraph pipeline with a hard contract. **THE BOSS** (Qwen2.5-3B, 4-bit,
constrained JSON tool-calling) reads the query and the image *metadata* and
names one specialist to run. Its decision then passes a deterministic gate that
re-checks it against each worker's declared input contract, so an incompatible
worker call is structurally impossible — whatever the 3B model believed.

A change query with one image, a SAR scene sent to the optical-only specialist,
a non-co-registered pair: all are refused with a specific, actionable reason
rather than answered wrongly.

The deployed default swaps the 3B router for a deterministic rule router
implementing the same documented rules — instant, zero VRAM, reproducible.
`SATQUERY_ROUTER=llm` switches back.

### `models/` — the specialists

Two Qwen2.5-VL-3B QLoRA fine-tunes, trained on a **6 GB RTX 4050 laptop**, both
emitting parseable JSON. Both adapters attach to **one** 4-bit base, so the
backend holds 3.75B params once — 2.65 GB VRAM, not twice that.

| | base model | fine-tuned |
|---|---|---|
| **Bi-temporal** (LEVIR-CC, 120 pairs, chance = 50%) | | |
| balanced accuracy | 50.0% | **94.2%** |
| valid schema JSON | 0 / 120 | **120 / 120** |
| CIDEr-D | 2.5 | **123.6** |
| **Single-image** (VRSBench, 60 images) | | |
| valid schema JSON | 0% | **100%** |
| object-class F1 (set) | — | **63.3%** |
| BLEU-4 | 2.0 | **12.1** |

Full tables with bootstrap 95% confidence intervals:
[models/results/METRICS.md](models/results/METRICS.md).

**Honest limits:** object counting is unreliable, and caption wording differs
from reference even when substantively correct. This system does structured
extraction and change detection; it is not a source of verified counts. Every
generated PDF says so.

### `backend/` — integration and reporting

FastAPI. Wires the specialists into the graph, applies the user's query to the
model's JSON output (never to the adapter prompt), draws the bounding boxes, and
renders the PDF. See [backend/README.md](backend/README.md).

### `frontend/` — the UI

React 19 + Vite + Tailwind. Upload, query, live routing visualisation, result
panel with the annotated imagery and the report download.

---

## What the validation layer does and does not claim

Every model output is checked for **self-consistency**: a stated position that
disagrees with its own bounding box, a caption claiming three ships when two
were extracted, labels outside the trained vocabulary, degenerate boxes.
Self-contradiction is strong evidence of a hallucination, so this catches a real
class of bad output.

It **cannot** verify a claim against the actual image — nothing downstream of
the model can. Measured on 180 cached predictions: zero false rejects on good
output, 100% rejection of the base model's unstructured prose.

---

## Status

| | |
|---|---|
| worker1 — single-image (VRSBench) | trained, wired in |
| worker3 — bi-temporal change (LEVIR-CC) | trained, wired in |
| worker2 — cross-modal optical+SAR | contract defined, **no model yet** — routes to an honest stub |

---

## Tests

```powershell
$env:PYTHONPATH="."
python -m tests.test_routing            # LLM emissions -> graph edge
python -m tests.test_backend_routing    # real user phrasings -> graph edge
```

No GPU, weights or network required.
