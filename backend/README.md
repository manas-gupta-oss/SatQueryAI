# SatQueryAI backend

The layer that joins the three halves of this repo and produces the deliverable
they were missing:

```
frontend/        React UI                 ──▶ this API
orchestration/   LangGraph + BOSS router  ──▶ build_graph(worker_impls=, boss_impl=)
models/          two QLoRA specialists    ──▶ backend/workers/nodes.py
                                          ──▶ a PDF report per query
```

Nothing in `orchestration/` or `models/` was modified. Both are plugged in
through the extension points they already documented.

---

## Run it

Two processes. Backend first.

```powershell
# from the repo root
.venv-unsloth\Scripts\python.exe -m uvicorn backend.app:app --port 8000 --reload
```

```powershell
cd frontend
npm install          # first time only
npm run dev          # http://localhost:5173
```

Open <http://localhost:5173>, drop in a satellite image, ask a question, and the
result panel gives you the answer, the annotated imagery, and a **Download PDF**
button.

### Environment

The backend imports `models/generate_report_v2.py`, which needs `unsloth` +
`transformers 5.5` — so it belongs in **`.venv-unsloth`**, not the plain `.venv`.
See `models/requirements-unsloth.txt` for that environment, then add the backend
layer on top:

```powershell
uv pip install --python .venv-unsloth\Scripts\python.exe -r backend\requirements.txt
```

The backend also runs with **no ML stack at all** — the specialists degrade to
clearly-labelled stubs and routing, validation and PDF generation still work.
That is the path to use for frontend work on a machine with no GPU.

---

## Configuration

Every setting is an environment variable with a working default (`backend/config.py`).

| variable | default | what it does |
|---|---|---|
| `SATQUERY_ROUTER` | `heuristic` | `heuristic` = the deterministic rule router; `llm` = the Qwen2.5-3B BOSS |
| `SATQUERY_WORKERS` | `auto` | `auto` falls back to stubs if the model will not load; `real` fails loudly instead; `stub` never loads a model |
| `SATQUERY_PRELOAD` | `0` | `1` warms the vision model at startup (~40 s) instead of on the first query |
| `SATQUERY_BENCHMARK_MODE` | `1` | accept PNG/JPEG. Off = GeoTIFF only |
| `SATQUERY_DATA_DIR` | `backend/_data` | uploads, overlays and generated PDFs |
| `SATQUERY_CORS_ORIGINS` | localhost 5173/4173 | comma-separated |
| `SATQUERY_MAX_UPLOAD_BYTES` | `52428800` | 50 MB, matching the frontend's client-side check |

### Why the router defaults to rules, not the 3B model

Both routers emit the same `BossDecision` and both pass through the same
deterministic gate (`tool_schema.validate_decision`), so an incompatible worker
call is structurally impossible either way. The difference is cost and risk:

|  | `heuristic` (default) | `llm` |
|---|---|---|
| startup | instant | ~40 s model load |
| VRAM | none | ~2.2 GB, on top of the vision stack's 2.65 GB |
| reproducible | exactly | greedy decoding, so yes in practice |
| handles novel phrasing | no — keyword families only | yes |

On a 6 GB laptop that is already holding the specialists, the rule router is the
one that will not surprise you in front of an audience. `SATQUERY_ROUTER=llm`
switches to the real BOSS and needs `PyYAML` + `lm-format-enforcer`; its settings
live in `configs/boss_config.yaml`.

### Why `benchmark_mode` is on

`tool_schema.check_compatibility` only accepts PNG/JPEG when the input bundle is
flagged as benchmark data; otherwise GeoTIFF is required. Both specialists were
fine-tuned on PNG benchmark datasets (VRSBench, LEVIR-CC), so PNG *is* the
in-distribution input here. Turning the flag off enforces the stricter
operational contract and will reject ordinary PNG uploads.

---

## API

| endpoint | purpose |
|---|---|
| `GET /api/health` | which router, which workers, whether the model is loaded |
| `GET /api/workers` | the live registry — downgrades to `stub` if a load has failed |
| `POST /api/upload` | one image, `multipart/form-data` |
| `POST /api/query` | run the graph, generate the report |
| `GET /api/report/{id}` | the PDF |
| `GET /api/report/{id}/json` | the structured result behind it |
| `GET /media/...` | stored imagery and annotated overlays |

Interactive docs at <http://localhost:8000/docs>.

The query response is whatever `finalize_node` assembled — it is not rebuilt
here, so the orchestrator keeps sole ownership of the response shape — plus
`report_url`, `overlay_url`, `analysis`, `self_consistency` and `degraded`.

---

## Layout

```
app.py              FastAPI app, CORS, static mounts, lifespan
config.py           every setting, all env-overridable
graph_runtime.py    compiles the graph once with this deployment's router + workers
store.py            in-process upload and report registries
schemas.py          wire format only

routing/heuristic.py    the rule router: R1-R5 and V1-V7 as code
workers/reporter.py     lazy, lock-guarded singleton around SatelliteReporter
workers/answers.py      user query applied to the model's JSON, never to the adapter
workers/nodes.py        worker1 / worker3 LangGraph nodes
report/overlay.py       bounding boxes and before/after strips (PIL)
report/pdf.py           the PDF itself (reportlab)
routes/                 upload, query, report, meta
```

---

## Three constraints this code is built around

All three come from `models/MODELS.md` and are not optional.

**1. The user's query never reaches the adapters.** Both were fine-tuned on
exactly one prompt each and learned *image → schema*, not *question → answer*.
Overriding the prompt measurably destroys the JSON — and with it the bounding
boxes. So `reporter.py` calls `analyze_image(path)` / `analyze_pair(a, b)` with
no `prompt=`, and the query is applied afterwards, in `workers/answers.py`, to
the JSON that came back. "How many ships?" becomes a count over
`analysis.objects`; "highlight the harbour" becomes a filter over the same list.

**2. `status` is always checked.** `analyze_*` never raises, so an unchecked
status fails silently rather than loudly.

**3. Validation is self-consistency, not verification.** `models/validate.py`
runs on every envelope and its verdict is reported in the API and in the PDF. It
catches the model contradicting itself — a position that disagrees with its own
box, a caption whose count disagrees with the objects extracted, labels outside
the trained vocabulary. It cannot check a claim against the actual pixels, and
nothing downstream of the model can. The PDF says so, in those words.

Two smaller ones: boxes are drawn from `bbox_normalized` (clamped) and never the
raw coordinates, which can fall outside the frame; and `analyze_pair` is
order-sensitive, so `image_assignment` decides which upload is `pre` — from role
hints, then timestamps, then upload order, with the assumption recorded in the
audit trail.

---

## worker2 is deliberately a stub

The cross-modal optical+SAR specialist has a registry entry and a full input
contract, but no trained model. It keeps the orchestration-layer stub, so a
query routed there returns an honest "not implemented" instead of a fabricated
answer, and `/api/workers` reports it as `stub`. Dropping a
`WorkerId.WORKER2 → node` entry into `model_worker_impls()` is the only change
needed once a model exists.

---

## Tests

```powershell
$env:PYTHONPATH="."
python -m tests.test_routing            # 18 cases, LLM emissions -> graph edge
python -m tests.test_backend_routing    # 24 cases, real phrasings -> graph edge
```

Neither needs a GPU, weights or a network.
