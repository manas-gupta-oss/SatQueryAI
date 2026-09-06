r"""Satellite imagery -> structured JSON report. Both specialist agents, one model.

Runtime: the Unsloth venv (.venv-unsloth). Both adapters are Unsloth/peft-0.20
format, so one environment serves the whole backend.

CLI
---
    # single image (VRSBench agent)
    python generate_report_v2.py --image satellite.png

    # image pair (bi-temporal change agent) -- first image is the EARLIER one
    python generate_report_v2.py --before 2019.png --after 2024.png

    python generate_report_v2.py --image a.png --out report.json --pretty

Python
------
    from generate_report_v2 import SatelliteReporter

    rep = SatelliteReporter()                  # loads ONCE, ~2.7 GB VRAM
    single = rep.analyze_image("scene.png")
    change = rep.analyze_pair("2019.png", "2024.png")

Both adapters attach to ONE 4-bit base and are switched with `set_adapter()`,
so the backend holds 3.75B params once rather than twice. Verified: switching
back and forth reproduces byte-identical output, so there is no state leakage
between agents.

Keep one SatelliteReporter alive for the process lifetime. Construction loads
the model (~40 s); each analyze call is a few seconds afterwards.

Output contract
---------------
Both methods return the SAME envelope, so the PDF layer has one shape to
handle. `task` says which agent produced it and therefore which `analysis`
fields are populated.

    {
      "status": "ok" | "error",
      "error":  null | "message",
      "task":   "single_image" | "bitemporal_change",
      "images": [{"filename","path","width","height","role"}],
      "model":  {"base","adapter","backend","trained_steps"},
      "generated_at": "2026-09-05T14:52:03Z",
      "inference_seconds": 4.1,
      "analysis": {...},          # task-specific, always present, never null
      "summary":  {...}
    }

single_image  analysis: {caption, objects[], qa_pairs[]}
bitemporal    analysis: {change_detected, change_summary, changed_classes[],
                         change_regions[], change_extent}

Every bounding box carries `bbox_normalized` -- [x0,y0,x1,y1] clamped to 0..1 --
so the renderer never receives an out-of-range coordinate.
"""

from __future__ import annotations

# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
SINGLE_ADAPTER = PROJECT_ROOT / "adapters" / "report-adapter-v2"
PAIR_ADAPTER = PROJECT_ROOT / "adapters" / "bitemporal-adapter"

# These must match the prompts the adapters were trained on.
SINGLE_PROMPT = "Describe the image in detail."
PAIR_PROMPT = ("These are two satellite images of the same location. The first "
               "image is the earlier one and the second is the later one. "
               "Describe what has changed between them.")

SINGLE_NAME = "default"          # name the first adapter loads under
PAIR_NAME = "bitemporal"


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _balanced_object(text: str) -> str | None:
    """First balanced {...} run, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_model_json(text: str) -> tuple[dict | None, str | None]:
    if not text or not text.strip():
        return None, "model returned empty output"
    if set(text.strip()) <= {"!", " ", "\n"}:
        return None, "model output is all '!' -- adapter weights are corrupted"
    for cand in (text.strip(), _FENCE.sub("", text.strip()), _balanced_object(text)):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, "model output is not parseable as a JSON object"


def _clamp_bbox(coord: Any) -> list[float] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in coord[:4])
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [round(min(max(v, 0.0), 1.0), 4) for v in (x0, y0, x1, y1)]


def normalize_single(obj: dict) -> dict:
    cap = obj.get("caption")
    objects = obj.get("objects")
    if isinstance(objects, dict):
        objects = [objects]
    out_objs = []
    for i, o in enumerate(objects or []):
        if not isinstance(o, dict):
            continue
        out_objs.append({
            "obj_id": o.get("obj_id", i),
            "obj_cls": str(o.get("obj_cls") or "").strip(),
            "referring_sentence": str(o.get("referring_sentence") or "").strip(),
            "obj_position": str(o.get("obj_position") or "").strip(),
            "obj_size": str(o.get("obj_size") or "").strip(),
            "obj_coord": o.get("obj_coord"),
            "bbox_normalized": _clamp_bbox(o.get("obj_coord")),
        })
    qa = obj.get("qa_pairs")
    if isinstance(qa, dict):
        qa = [qa]
    out_qa = []
    for i, q in enumerate(qa or []):
        if isinstance(q, dict):
            out_qa.append({
                "ques_id": q.get("ques_id", i + 1),
                "question": str(q.get("question") or "").strip(),
                "answer": str(q.get("answer") or "").strip(),
                "type": str(q.get("type") or "").strip(),
            })
    return {"caption": cap.strip() if isinstance(cap, str) else "",
            "objects": out_objs, "qa_pairs": out_qa}


def normalize_pair(obj: dict) -> dict:
    regions = obj.get("change_regions")
    if isinstance(regions, dict):
        regions = [regions]
    out_regions = []
    for r in regions or []:
        if not isinstance(r, dict):
            continue
        out_regions.append({
            "class": str(r.get("class") or "").strip(),
            "size": str(r.get("size") or "").strip(),
            "bbox": r.get("bbox"),
            "bbox_normalized": _clamp_bbox(r.get("bbox")),
        })
    classes = obj.get("changed_classes")
    if isinstance(classes, str):
        classes = [classes]
    summary = obj.get("change_summary")
    return {
        "change_detected": bool(obj.get("change_detected")),
        "change_summary": summary.strip() if isinstance(summary, str) else "",
        "changed_classes": [str(c).strip() for c in (classes or []) if str(c).strip()],
        "change_regions": out_regions,
        "change_extent": str(obj.get("change_extent") or "").strip(),
    }


EMPTY_SINGLE = {"caption": "", "objects": [], "qa_pairs": []}
EMPTY_PAIR = {"change_detected": False, "change_summary": "", "changed_classes": [],
              "change_regions": [], "change_extent": ""}


# ---------------------------------------------------------------------------

class SatelliteReporter:
    """One 4-bit base, both LoRA adapters, switched per call."""

    def __init__(self, single_adapter: str | Path = SINGLE_ADAPTER,
                 pair_adapter: str | Path = PAIR_ADAPTER, verbose: bool = True):
        self.single_adapter = Path(single_adapter)
        self.pair_adapter = Path(pair_adapter)
        for p in (self.single_adapter, self.pair_adapter):
            if not p.exists():
                raise FileNotFoundError(f"adapter not found: {p}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required (~2.7 GB VRAM in 4-bit).")

        def _meta(p: Path) -> dict:
            f = p / "train_meta.json"
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

        self.meta = {"single_image": _meta(self.single_adapter),
                     "bitemporal_change": _meta(self.pair_adapter)}

        if verbose:
            print(f"loading base + 2 adapters ({self.single_adapter.name}, "
                  f"{self.pair_adapter.name}) ...")

        self.model, self.processor = FastVisionModel.from_pretrained(
            str(self.single_adapter), load_in_4bit=True)
        self.model.load_adapter(str(self.pair_adapter), adapter_name=PAIR_NAME)
        FastVisionModel.for_inference(self.model)
        self.model.config.use_cache = True

        bad = [n for n, p in self.model.named_parameters()
               if "lora" in n and not torch.isfinite(p).all()]
        if bad:
            raise RuntimeError(
                f"adapter corrupted: {len(bad)} non-finite LoRA tensors "
                f"(e.g. {bad[:3]}). Re-export from a clean checkpoint.")

        self._active = SINGLE_NAME
        if verbose:
            print(f"ready. {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
                  f"adapters: {list(self.model.peft_config)}")

    # -- internals ---------------------------------------------------------

    def _use(self, name: str) -> None:
        if self._active != name:
            self.model.set_adapter(name)
            self._active = name

    def _generate(self, images: list[Image.Image], prompt: str,
                  max_new_tokens: int) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _envelope(self, task: str, images: list[dict], status: str, error: str | None,
                  analysis: dict, seconds: float, summary: dict,
                  raw: str | None = None) -> dict:
        m = self.meta.get(task, {})
        env = {
            "status": status,
            "error": error,
            "task": task,
            "images": images,
            "model": {
                "base": m.get("unsloth_model", "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"),
                "adapter": (self.single_adapter if task == "single_image"
                            else self.pair_adapter).name,
                "backend": m.get("backend", "unsloth"),
                "trained_steps": m.get("step"),
            },
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inference_seconds": round(seconds, 2),
            "analysis": analysis,
            "summary": summary,
        }
        if raw is not None:
            env["raw_output"] = raw
        return env

    @staticmethod
    def _img_meta(path: Path, role: str, size: tuple[int, int]) -> dict:
        return {"filename": path.name, "path": str(path),
                "width": size[0], "height": size[1], "role": role}

    def _open(self, path: Path):
        with Image.open(path) as im:
            return im.convert("RGB"), im.size

    # -- public ------------------------------------------------------------

    def analyze_image(self, image_path: str | Path,
                      prompt: str = SINGLE_PROMPT,
                      max_new_tokens: int = 640) -> dict:
        """Single-image scene analysis. Never raises; check result["status"]."""
        p = Path(image_path)
        blank = [self._img_meta(p, "scene", (0, 0))]
        if not p.exists():
            return self._envelope("single_image", blank, "error",
                                  f"image not found: {p}", dict(EMPTY_SINGLE), 0.0,
                                  {"n_objects": 0, "object_classes": [], "n_qa_pairs": 0})
        t0 = time.time()
        try:
            img, size = self._open(p)
            self._use(SINGLE_NAME)
            text = self._generate([img], prompt, max_new_tokens)
        except Exception as e:                       # noqa: BLE001
            return self._envelope("single_image", blank, "error", f"inference failed: {e}",
                                  dict(EMPTY_SINGLE), time.time() - t0,
                                  {"n_objects": 0, "object_classes": [], "n_qa_pairs": 0})
        secs = time.time() - t0
        imgs = [self._img_meta(p, "scene", size)]

        obj, err = parse_model_json(text)
        if obj is None:
            return self._envelope("single_image", imgs, "error", err,
                                  {**EMPTY_SINGLE, "caption": text[:2000]}, secs,
                                  {"n_objects": 0, "object_classes": [], "n_qa_pairs": 0},
                                  raw=text)
        a = normalize_single(obj)
        return self._envelope("single_image", imgs, "ok", None, a, secs, {
            "n_objects": len(a["objects"]),
            "object_classes": sorted({o["obj_cls"] for o in a["objects"] if o["obj_cls"]}),
            "n_qa_pairs": len(a["qa_pairs"]),
        })

    def analyze_pair(self, before: str | Path, after: str | Path,
                     prompt: str = PAIR_PROMPT,
                     max_new_tokens: int = 448) -> dict:
        """Bi-temporal change analysis. `before` MUST be the earlier image."""
        pb, pa = Path(before), Path(after)
        blank = [self._img_meta(pb, "before", (0, 0)), self._img_meta(pa, "after", (0, 0))]
        empty_sum = {"change_detected": False, "n_regions": 0, "changed_classes": []}
        for p in (pb, pa):
            if not p.exists():
                return self._envelope("bitemporal_change", blank, "error",
                                      f"image not found: {p}", dict(EMPTY_PAIR), 0.0,
                                      empty_sum)
        t0 = time.time()
        try:
            ib, sb = self._open(pb)
            ia, sa = self._open(pa)
            self._use(PAIR_NAME)
            text = self._generate([ib, ia], prompt, max_new_tokens)
        except Exception as e:                       # noqa: BLE001
            return self._envelope("bitemporal_change", blank, "error",
                                  f"inference failed: {e}", dict(EMPTY_PAIR),
                                  time.time() - t0, empty_sum)
        secs = time.time() - t0
        imgs = [self._img_meta(pb, "before", sb), self._img_meta(pa, "after", sa)]

        obj, err = parse_model_json(text)
        if obj is None:
            return self._envelope("bitemporal_change", imgs, "error", err,
                                  {**EMPTY_PAIR, "change_summary": text[:2000]},
                                  secs, empty_sum, raw=text)
        a = normalize_pair(obj)
        return self._envelope("bitemporal_change", imgs, "ok", None, a, secs, {
            "change_detected": a["change_detected"],
            "n_regions": len(a["change_regions"]),
            "changed_classes": a["changed_classes"],
        })


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Satellite imagery -> structured JSON report (both agents).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="single image (VRSBench agent)")
    ap.add_argument("--before", help="earlier image (bi-temporal agent)")
    ap.add_argument("--after", help="later image (bi-temporal agent)")
    ap.add_argument("--single-adapter", default=str(SINGLE_ADAPTER))
    ap.add_argument("--pair-adapter", default=str(PAIR_ADAPTER))
    ap.add_argument("--out", help="write JSON here instead of stdout")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.image and not (args.before and args.after):
        default = PROJECT_ROOT / "satellite.png"
        if default.exists():
            args.image = str(default)
        else:
            ap.error("give --image, or both --before and --after")
    if bool(args.before) != bool(args.after):
        ap.error("--before and --after must be given together")

    try:
        rep = SatelliteReporter(args.single_adapter, args.pair_adapter,
                                verbose=not args.quiet)
    except Exception as e:                           # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(e)}, indent=2))
        return 1

    reports = []
    if args.image:
        reports.append(rep.analyze_image(args.image))
    if args.before:
        reports.append(rep.analyze_pair(args.before, args.after))

    if not args.quiet:
        for r in reports:
            s = r["summary"]
            print(f"  [{r['task']}] {r['status']}  {s}  {r['inference_seconds']}s")
            if r["status"] == "error":
                print(f"     error: {r['error']}")

    indent = 2 if args.pretty else None
    payload = reports[0] if len(reports) == 1 else reports
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=indent, ensure_ascii=False),
                       encoding="utf-8")
        if not args.quiet:
            print(f"\nreport -> {out}")
    else:
        print(json.dumps(payload, indent=indent, ensure_ascii=False))

    return 0 if all(r["status"] == "ok" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
