r"""Build the bi-temporal (change-detection) training set from LEVIR-CC.

    python -m bitemporal.prepare_data                    # writes train/val/test jsonl
    python -m bitemporal.prepare_data --all-captions     # 5 rows per pair instead of 1
    python -m bitemporal.prepare_data --no-regions       # skip mask-derived bboxes
    python -m bitemporal.prepare_data --stats-only       # inspect without writing

Inputs
------
    data/LevirCCcaptions.json                       10,077 pairs x 5 captions
    data/images/{split}/{A,B}/<file>.png            pre/post images, 256x256
    data/LEVIR-MCI-dataset/images/{split}/label_rgb/<file>.png   change masks

Output
------
    data/bitemporal/{train,val,test}.jsonl

Each row:
    {"image_a", "image_b", "question", "answer", "changeflag", "filename"}

`answer` is compact JSON in the same spirit as the VRSBench agent, so both
specialists hand the report layer the same kind of payload:

    {"change_detected": true,
     "change_summary": "A lane appears on the left and several houses are built at the bottom.",
     "changed_classes": ["building", "road"],
     "change_regions": [{"class": "building", "bbox": [0.12,0.55,0.34,0.78], "size": "small"}],
     "change_extent": "small"}

Two things this script fixes about the raw data
-----------------------------------------------
1. Every LEVIR-CC caption is lowercase, space-padded and has a detached final
   period (" a lane appears on the left ."). Training on that verbatim makes
   the PDF report read badly, so captions are normalised to sentence case with
   attached punctuation.
2. Captions describe *what* changed but never *where*. The LEVIR-MCI masks are
   used to derive grounded `change_regions`, giving the report drawable boxes
   labelled by class (road / building) instead of prose alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
CAPTIONS = DATA / "LevirCCcaptions.json"
IMAGES = DATA / "images"
MASKS = DATA / "LEVIR-MCI-dataset" / "images"
OUT_DIR = DATA / "bitemporal"

# The order matters and the model cannot infer it: the first image is always
# the earlier one. Stating it explicitly stops the model guessing the direction
# of change ("built" vs "demolished") from context alone.
QUESTION = ("These are two satellite images of the same location. The first "
            "image is the earlier one and the second is the later one. "
            "Describe what has changed between them.")

# label_rgb encoding. NOTE: the dataset readme claims road = (0,0,255) blue,
# but blue never occurs in any mask. Surveying 800 train masks gives only:
#   (0,0,0) background | (255,0,0) building, 413 files | (255,255,0) road, 211 files
# Cross-checked against the grayscale `label` variant, where white=building
# appears in 179 files and gray=road in 84 -- the same ~2:1 ratio. So road is
# YELLOW, not blue; trusting the readme silently discards every road change.
CLASS_COLORS = {"road": (255, 255, 0), "building": (255, 0, 0)}

MIN_REGION_PX = 30          # ignore specks; 30/65536 px is well under 0.05% of frame
MAX_REGIONS = 6             # keep the target short and the report readable


# ---------------------------------------------------------------------------
# caption normalisation
# ---------------------------------------------------------------------------

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_MULTISPACE = re.compile(r"\s+")


def normalize_caption(raw: str) -> str:
    """' a lane appears on the left .' -> 'A lane appears on the left.'"""
    s = _MULTISPACE.sub(" ", raw.strip())
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s


def pick_canonical(sentences: list[dict]) -> str:
    """One description per pair, for output consistency in the report.

    The longest caption is chosen: LEVIR-CC sentences are short (p50 = 8 words)
    and length correlates with how many distinct changes were mentioned. For
    no-change pairs every caption is equivalent, so the choice does not matter.
    """
    best, best_len = "", -1
    for s in sentences:
        text = normalize_caption(s.get("raw", ""))
        n = len(text.split())
        if n > best_len:
            best, best_len = text, n
    return best


# ---------------------------------------------------------------------------
# mask -> grounded regions
# ---------------------------------------------------------------------------

def extract_regions(mask_path: Path) -> tuple[list[dict], float]:
    """Connected components per class -> normalised bboxes + changed area fraction."""
    from scipy import ndimage

    if not mask_path.exists():
        return [], 0.0
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("RGB"))

    h, w = arr.shape[:2]
    total = float(h * w)
    regions: list[dict] = []
    changed_px = 0

    for cls, color in CLASS_COLORS.items():
        binary = np.all(arr == np.array(color, dtype=arr.dtype), axis=-1)
        n_px = int(binary.sum())
        if n_px == 0:
            continue
        changed_px += n_px

        labeled, n = ndimage.label(binary)
        if n == 0:
            continue
        for sl_y, sl_x in ndimage.find_objects(labeled):
            comp = labeled[sl_y, sl_x]
            area = int((comp > 0).sum())
            if area < MIN_REGION_PX:
                continue
            regions.append({
                "class": cls,
                "bbox": [round(sl_x.start / w, 3), round(sl_y.start / h, 3),
                         round(sl_x.stop / w, 3), round(sl_y.stop / h, 3)],
                "_area": area,
            })

    regions.sort(key=lambda r: -r["_area"])
    kept = []
    for r in regions[:MAX_REGIONS]:
        frac = r.pop("_area") / total
        r["size"] = "large" if frac > 0.05 else ("medium" if frac > 0.01 else "small")
        kept.append(r)
    for r in regions[MAX_REGIONS:]:
        r.pop("_area", None)

    return kept, changed_px / total


def extent_label(frac: float) -> str:
    if frac <= 0.0:
        return "none"
    if frac < 0.01:
        return "minimal"
    if frac < 0.05:
        return "small"
    if frac < 0.15:
        return "moderate"
    return "extensive"


# ---------------------------------------------------------------------------
# row building
# ---------------------------------------------------------------------------

def build_answer(caption: str, changed: bool, regions: list[dict],
                 frac: float) -> str:
    classes = sorted({r["class"] for r in regions})
    payload = {
        "change_detected": bool(changed),
        "change_summary": caption,
        "changed_classes": classes,
        "change_regions": regions,
        # A changed pair with no mask coverage is real but unmeasured: LEVIR-MCI
        # only annotates building and road, so captions about removed trees or
        # unsurfaced tracks leave the mask empty. Saying "none" there would
        # contradict change_detected=true, so mark it unquantified instead.
        "change_extent": (extent_label(frac) if not changed
                          else (extent_label(frac) if regions else "unquantified")),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_split(entries: list[dict], split: str, use_regions: bool,
                all_captions: bool) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats: Counter = Counter()

    for e in entries:
        fn = e["filename"]
        img_a = IMAGES / split / "A" / fn
        img_b = IMAGES / split / "B" / fn
        if not img_a.exists() or not img_b.exists():
            stats["missing_image"] += 1
            continue

        changed = bool(e.get("changeflag", 0))
        regions: list[dict] = []
        frac = 0.0
        if use_regions and changed:
            regions, frac = extract_regions(MASKS / split / "label_rgb" / fn)
            if not regions:
                stats["changed_but_no_mask_regions"] += 1

        sentences = e.get("sentences", [])
        texts = ([normalize_caption(s.get("raw", "")) for s in sentences]
                 if all_captions else [pick_canonical(sentences)])

        for text in texts:
            if not text:
                stats["empty_caption"] += 1
                continue
            rows.append({
                "image_a": str(img_a),
                "image_b": str(img_b),
                "question": QUESTION,
                "answer": build_answer(text, changed, regions, frac),
                "changeflag": int(changed),
                "filename": fn,
            })
        stats["pairs"] += 1
        stats["changed" if changed else "unchanged"] += 1
        stats["regions"] += len(regions)

    return rows, stats


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-captions", action="store_true",
                    help="emit all 5 captions as separate rows (5x data, more varied output)")
    ap.add_argument("--no-regions", action="store_true",
                    help="skip mask-derived bounding boxes")
    ap.add_argument("--stats-only", action="store_true", help="do not write files")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    if not CAPTIONS.exists():
        print(f"missing captions: {CAPTIONS}")
        return 2
    if not IMAGES.exists():
        print(f"missing images: {IMAGES}")
        return 2

    use_regions = not args.no_regions
    if use_regions and not MASKS.exists():
        print(f"note: masks not found at {MASKS}; continuing without regions")
        use_regions = False

    data = json.loads(CAPTIONS.read_text(encoding="utf-8"))["images"]
    by_split: dict[str, list[dict]] = {}
    for e in data:
        by_split.setdefault(e["split"], []).append(e)

    out_dir = Path(args.out)
    if not args.stats_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    grand = Counter()
    for split in ("train", "val", "test"):
        entries = by_split.get(split, [])
        if not entries:
            continue
        rows, stats = build_split(entries, split, use_regions, args.all_captions)
        grand.update(stats)

        print(f"[{split}] {len(rows)} rows from {stats['pairs']} pairs "
              f"({stats['changed']} changed / {stats['unchanged']} unchanged)"
              f"  regions={stats['regions']}")
        for k in ("missing_image", "empty_caption", "changed_but_no_mask_regions"):
            if stats[k]:
                print(f"         {k}: {stats[k]}")

        if not args.stats_only:
            path = out_dir / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"         -> {path}")

    # target-length sanity, which is what decides the VRAM budget
    if not args.stats_only:
        tr = out_dir / "train.jsonl"
        if tr.exists():
            lens = []
            with tr.open(encoding="utf-8") as f:
                for line in f:
                    lens.append(len(json.loads(line)["answer"]))
            lens.sort()
            n = len(lens)
            print(f"\ntarget length (chars): p50={lens[n//2]} p95={lens[int(n*.95)]} max={lens[-1]}")
            print(f"est. target tokens   : p50~{lens[n//2]//3.6:.0f} p95~{lens[int(n*.95)]//3.6:.0f}")
            print(f"+ 2x81 vision tokens + ~40 prompt = p95 total ~"
                  f"{lens[int(n*.95)]//3.6 + 162 + 40:.0f} tokens")
    print(f"\ntotal pairs: {grand['pairs']}  regions extracted: {grand['regions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
