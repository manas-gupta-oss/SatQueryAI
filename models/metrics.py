r"""Paper-style caption metrics for both satquery agents.

    # 1. generate predictions (needs .venv-unsloth -- loads the model)
    .venv-unsloth\Scripts\python.exe metrics.py generate --task bitemporal --both
    .venv-unsloth\Scripts\python.exe metrics.py generate --task single     --both

    # 2. score them (pure CPU, runs in EITHER venv -- no model loaded)
    .venv\Scripts\python.exe metrics.py score --task bitemporal --task single \
        --markdown results/METRICS.md

Why two phases
--------------
Generation needs the Unsloth stack; BERTScore wants `bert-score`, which pulls
its own transformers expectations. Splitting them means the expensive GPU pass
runs once, is cached to jsonl, and can be re-scored with new metrics for free.

Metrics implemented here, self-contained (no coco-caption / Java / nltk needed):

    BLEU-1..4    Papineni et al. 2002, corpus-level, closest-length brevity
                 penalty, multi-reference clipping
    ROUGE-1/2    n-gram F1, max over references
    ROUGE-L      LCS-based F with beta=1.2 (the coco-caption convention)
    METEOR       exact + stem matching with the standard alpha=0.9, beta=3,
                 gamma=0.5 fragmentation penalty. NOTE: no WordNet synonym
                 stage unless nltk+wordnet is installed, in which case the real
                 nltk implementation is used and the report says so.
    CIDEr-D      Vedantam et al. 2015, tf-idf over n=1..4 with the length
                 gaussian (sigma=6) and count clipping. Document frequency is
                 computed over the evaluation set itself, as in the original.
    BERTScore    optional; needs `pip install bert-score`. Reported rescaled
                 with the published baseline so values are comparable to papers.

Plus the task metrics that the caption scores cannot express: schema validity,
balanced change-detection accuracy, object counting error, class F1.

Every corpus metric also carries a bootstrap 95% confidence interval, so the
base-vs-tuned gaps can be reported as significant or not rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
RULE = "=" * 78

# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
# The coco-caption suite runs the PTB tokenizer (a Java jar) and then lowercases
# and drops punctuation. This reproduces the *effect* of that pipeline without
# the jar: everything is compared as lowercase alphanumeric words. Applied
# identically to hypotheses and references, so no side is advantaged.

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


def ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------
# Corpus-level, which is what papers report: precision is the ratio of SUMS
# over the corpus, not the mean of per-sentence ratios. Per-sentence stats are
# kept so the bootstrap can re-aggregate them on each resample.

MAX_N = 4


def bleu_stats(hyp: list[str], refs: list[list[str]], max_n: int = MAX_N) -> dict:
    clip = [0] * max_n
    tot = [0] * max_n
    for n in range(1, max_n + 1):
        h = ngrams(hyp, n)
        if not h:
            continue
        # Multi-reference clipping: an n-gram may be counted up to the MAXIMUM
        # number of times it appears in any single reference.
        ceiling: Counter = Counter()
        for r in refs:
            rc = ngrams(r, n)
            for g, c in rc.items():
                if c > ceiling[g]:
                    ceiling[g] = c
        clip[n - 1] = sum(min(c, ceiling[g]) for g, c in h.items())
        tot[n - 1] = sum(h.values())
    # Brevity penalty uses the reference whose length is closest to the
    # hypothesis, ties broken toward the shorter one.
    ref_len = min((len(r) for r in refs),
                  key=lambda L: (abs(L - len(hyp)), L)) if refs else 0
    return {"clip": clip, "tot": tot, "hyp_len": len(hyp), "ref_len": ref_len}


def bleu_from_stats(stats: list[dict], max_n: int = MAX_N) -> dict[str, float]:
    if not stats:
        return {f"BLEU-{n}": 0.0 for n in range(1, max_n + 1)}
    c = sum(s["hyp_len"] for s in stats)
    r = sum(s["ref_len"] for s in stats)
    bp = 1.0 if c > r else (math.exp(1.0 - r / c) if c > 0 else 0.0)

    log_p = []
    out: dict[str, float] = {}
    for n in range(max_n):
        num = sum(s["clip"][n] for s in stats)
        den = sum(s["tot"][n] for s in stats)
        # A zero numerator would send log to -inf and zero out every higher-order
        # BLEU. coco-caption floors it the same way rather than dropping the
        # order entirely.
        p = (num if num > 0 else 1e-9) / (den if den > 0 else 1e-9)
        log_p.append(math.log(p))
        out[f"BLEU-{n + 1}"] = bp * math.exp(sum(log_p) / (n + 1))
    return out


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: list[str], refs: list[list[str]], beta: float = 1.2) -> float:
    """LCS F-measure, beta=1.2 -- recall-weighted, the coco-caption setting."""
    best = 0.0
    for ref in refs:
        if not hyp or not ref:
            continue
        l = _lcs(hyp, ref)
        if l == 0:
            continue
        p, rc = l / len(hyp), l / len(ref)
        f = ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)
        best = max(best, f)
    return best


def rouge_n(hyp: list[str], refs: list[list[str]], n: int) -> float:
    """n-gram F1, max over references."""
    h = ngrams(hyp, n)
    best = 0.0
    for ref in refs:
        r = ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum(min(c, r[g]) for g, c in h.items())
        if overlap == 0:
            continue
        p, rc = overlap / sum(h.values()), overlap / sum(r.values())
        best = max(best, 2 * p * rc / (p + rc))
    return best


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------
# The reference implementation is a Java jar with a WordNet synonym stage and a
# paraphrase table. If nltk + the wordnet corpus are installed we defer to nltk
# (which has the synonym stage); otherwise this runs the exact + stem stages
# only. The fallback scores slightly LOWER than published METEOR because
# synonym matches are missed -- never higher -- so it cannot flatter the model.
# meteor_backend() reports which one ran, and the printed table says so too.

_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ies", "ied", "ed",
             "es", "ly", "s")


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_NLTK_METEOR = None


def meteor_backend() -> str:
    """'nltk-wordnet' if the real implementation is importable, else 'builtin'."""
    global _NLTK_METEOR
    if _NLTK_METEOR is None:
        try:
            from nltk.corpus import wordnet
            from nltk.translate.meteor_score import meteor_score
            wordnet.synsets("test")          # forces the corpus to actually load
            _NLTK_METEOR = meteor_score
        except Exception:                    # noqa: BLE001  missing pkg or corpus
            _NLTK_METEOR = False
    return "nltk-wordnet" if _NLTK_METEOR else "builtin"


def _align(hyp: list[str], ref: list[str]) -> list[tuple[int, int]]:
    """Greedy two-stage alignment: exact matches first, then stem matches."""
    pairs: list[tuple[int, int]] = []
    used_h: set[int] = set()
    used_r: set[int] = set()
    for key in (lambda w: w, _stem):
        rmap: dict[str, list[int]] = defaultdict(list)
        for j, w in enumerate(ref):
            if j not in used_r:
                rmap[key(w)].append(j)
        for i, w in enumerate(hyp):
            if i in used_h:
                continue
            cand = rmap.get(key(w))
            while cand and cand[0] in used_r:
                cand.pop(0)
            if cand:
                j = cand.pop(0)
                used_h.add(i)
                used_r.add(j)
                pairs.append((i, j))
    return sorted(pairs)


def _chunks(pairs: list[tuple[int, int]]) -> int:
    """Contiguous runs -- matches adjacent in BOTH sequences count as one."""
    if not pairs:
        return 0
    n = 1
    for (i0, j0), (i1, j1) in zip(pairs, pairs[1:]):
        if not (i1 == i0 + 1 and j1 == j0 + 1):
            n += 1
    return n


def meteor(hyp: list[str], refs: list[list[str]],
           alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    if meteor_backend() == "nltk-wordnet":
        try:
            return float(_NLTK_METEOR(refs, hyp))
        except Exception:                    # noqa: BLE001  fall through to builtin
            pass
    best = 0.0
    for ref in refs:
        if not hyp or not ref:
            continue
        pairs = _align(hyp, ref)
        m = len(pairs)
        if m == 0:
            continue
        p, rc = m / len(hyp), m / len(ref)
        fmean = p * rc / (alpha * p + (1 - alpha) * rc)
        pen = gamma * (_chunks(pairs) / m) ** beta
        best = max(best, fmean * (1 - pen))
    return best


# ---------------------------------------------------------------------------
# CIDEr-D
# ---------------------------------------------------------------------------
# Vedantam et al. 2015. Follows the coco-caption CIDEr-D implementation:
# tf-idf n-gram vectors for n=1..4, candidate counts clipped by the reference,
# cosine similarity per order, a gaussian length penalty (sigma=6) that punishes
# padding the caption to farm n-gram overlap, averaged over orders and
# references, times 10.
#
# Document frequency comes from the evaluation set's own references, exactly as
# in the original. That makes CIDEr sensitive to the eval set: a CIDEr computed
# over 120 pairs is NOT directly comparable to one published over 1929. Compare
# base vs tuned within this table, not against a paper's absolute number.

CIDER_SIGMA = 6.0


def _cider_df(all_refs: list[list[list[str]]]) -> dict[tuple, int]:
    df: Counter = Counter()
    for refs in all_refs:
        seen: set[tuple] = set()
        for r in refs:
            for n in range(1, MAX_N + 1):
                seen.update(ngrams(r, n).keys())
        for g in seen:
            df[g] += 1
    return df


def _cider_vec(toks: list[str], df: dict, log_n_docs: float):
    vec = [defaultdict(float) for _ in range(MAX_N)]
    norm = [0.0] * MAX_N
    length = 0
    for n in range(1, MAX_N + 1):
        for g, tf in ngrams(toks, n).items():
            idf = log_n_docs - math.log(max(1.0, float(df.get(g, 0))))
            vec[n - 1][g] = tf * idf
            norm[n - 1] += (tf * idf) ** 2
            if n == 1:
                length += tf
    return vec, [math.sqrt(x) for x in norm], length


def _cider_sim(vh, nh, lh, vr, nr, lr) -> float:
    delta = float(lh - lr)
    vals = []
    for n in range(MAX_N):
        v = sum(min(c, vr[n].get(g, 0.0)) * vr[n].get(g, 0.0)
                for g, c in vh[n].items())
        if nh[n] > 0 and nr[n] > 0:
            v /= nh[n] * nr[n]
        v *= math.exp(-(delta ** 2) / (2 * CIDER_SIGMA ** 2))
        vals.append(v)
    return sum(vals) / MAX_N


def cider_d(hyps: list[list[str]], all_refs: list[list[list[str]]]) -> list[float]:
    """Per-item CIDEr-D scores; the corpus score is their mean."""
    if not hyps:
        return []
    df = _cider_df(all_refs)
    log_n = math.log(float(len(all_refs)))
    out = []
    for hyp, refs in zip(hyps, all_refs):
        vh, nh, lh = _cider_vec(hyp, df, log_n)
        s = 0.0
        for ref in refs:
            vr, nr, lr = _cider_vec(ref, df, log_n)
            s += _cider_sim(vh, nh, lh, vr, nr, lr)
        out.append(10.0 * s / max(1, len(refs)))
    return out


# ---------------------------------------------------------------------------
# BERTScore (optional)
# ---------------------------------------------------------------------------

def bert_score_f1(hyps: list[str], refs: list[list[str]],
                  model_type: str | None = None) -> tuple[list[float], str] | None:
    """Per-item rescaled F1, or None if the package is not installed.

    rescale_with_baseline maps the raw cosine similarities (which sit around
    0.85 even for unrelated text) onto a spread-out range. Papers report the
    rescaled number; without it every model looks equally good.
    """
    try:
        from bert_score import score as _score
    except ImportError:
        return None
    note = model_type or "roberta-large (lang=en default)"
    kw = {"lang": "en", "verbose": False, "batch_size": 16}
    if model_type:
        kw["model_type"] = model_type
    try:
        _, _, f1 = _score(hyps, refs, rescale_with_baseline=True, **kw)
    except Exception:                        # noqa: BLE001  no baseline for this model
        _, _, f1 = _score(hyps, refs, rescale_with_baseline=False, **kw)
        note += " [NOT baseline-rescaled -- absolute values inflated]"
    return [float(x) for x in f1], note


# ---------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------
# Resamples the evaluation set with replacement and re-aggregates. For the
# per-item metrics that is a mean over resampled scores; for BLEU, which is a
# ratio of corpus sums, the per-sentence STATS are resampled and recombined --
# averaging per-sentence BLEU would be a different (and wrong) quantity.

def boot_ci(per_item: list[float], n: int, seed: int = 0) -> tuple[float, float]:
    if n <= 0 or len(per_item) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(per_item)
    means = []
    for _ in range(n):
        means.append(sum(per_item[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return (means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))])


def boot_ci_bleu(stats: list[dict], order: int, n: int,
                 seed: int = 0) -> tuple[float, float]:
    if n <= 0 or len(stats) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(stats)
    key = f"BLEU-{order}"
    vals = []
    for _ in range(n):
        sample = [stats[rng.randrange(k)] for _ in range(k)]
        vals.append(bleu_from_stats(sample)[key])
    vals.sort()
    return (vals[int(0.025 * n)], vals[min(n - 1, int(0.975 * n))])


# ---------------------------------------------------------------------------
# pulling text and structure out of a model's raw output
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def loose_json(text: str) -> dict | None:
    """Parse the first balanced {...} in the text. Returns None if there is none."""
    s = _FENCE.sub("", (text or "").strip())
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    if i < 0:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(i, len(s)):
        ch = s[j]
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
                try:
                    o = json.loads(s[i:j + 1])
                    return o if isinstance(o, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


CAPTION_FIELD = {"single": "caption", "bitemporal": "change_summary"}


def hypothesis_text(raw: str, task: str) -> str:
    """The sentence to score with BLEU/ROUGE/METEOR/CIDEr/BERTScore.

    The tuned models emit JSON, so the caption is a field. The base model emits
    prose and no JSON at all -- scoring its raw prose (rather than an empty
    string) is deliberately GENEROUS to the base model: it means the caption
    metrics measure description quality, not JSON formatting. Formatting is
    reported separately as the valid-schema rate.
    """
    obj = loose_json(raw)
    if obj is not None:
        v = obj.get(CAPTION_FIELD[task])
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _FENCE.sub("", (raw or "").strip())


# ---------------------------------------------------------------------------
# task metrics -- what the caption scores cannot see
# ---------------------------------------------------------------------------

NEG = re.compile(r"\b(no difference|no change|same as before|identical|unchanged|"
                 r"nothing has changed|remains the same)\b", re.I)

SINGLE_KEYS = {"caption", "objects", "qa_pairs"}
BITEMPORAL_KEYS = {"change_detected", "change_summary", "changed_classes"}


def pred_changed(raw: str) -> bool:
    """Did the model say something changed? JSON field if present, else phrasing."""
    obj = loose_json(raw)
    if obj is not None and isinstance(obj.get("change_detected"), bool):
        return obj["change_detected"]
    return not bool(NEG.search(raw or ""))


def obj_classes(objs) -> list[str]:
    out = []
    for o in objs or []:
        if isinstance(o, dict):
            c = str(o.get("obj_cls") or "").strip().lower()
            if c:
                out.append(c)
    return out


def _micro_f1(pred_counts: Counter, ref_counts: Counter) -> tuple[int, int, int]:
    tp = sum((pred_counts & ref_counts).values())
    return tp, sum(pred_counts.values()), sum(ref_counts.values())


def task_metrics(records: list[dict], task: str) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    valid = 0
    for r in records:
        o = loose_json(r["pred_raw"])
        need = SINGLE_KEYS if task == "single" else BITEMPORAL_KEYS
        if o is not None and need <= set(o.keys()):
            valid += 1
    out["valid schema JSON (%)"] = 100.0 * valid / n

    if task == "bitemporal":
        ch = [r for r in records if r.get("changeflag") == 1]
        un = [r for r in records if r.get("changeflag") == 0]
        rc = sum(pred_changed(r["pred_raw"]) for r in ch) / len(ch) if ch else float("nan")
        ru = sum(not pred_changed(r["pred_raw"]) for r in un) / len(un) if un else float("nan")
        out["recall, changed (%)"] = 100.0 * rc
        out["recall, unchanged (%)"] = 100.0 * ru
        out["balanced accuracy (%)"] = 100.0 * (rc + ru) / 2
        tp = pp = rp = 0
        for r in ch:
            o = loose_json(r["pred_raw"]) or {}
            p = Counter(str(x).lower() for x in (o.get("changed_classes") or [])
                        if isinstance(x, str))
            g = Counter(str(x).lower() for x in (r.get("ref_classes") or []))
            a, b, c = _micro_f1(p, g)
            tp += a; pp += b; rp += c
        if ch:
            prec = tp / pp if pp else 0.0
            rec = tp / rp if rp else 0.0
            out["changed-class F1 (%)"] = 100.0 * (2 * prec * rec / (prec + rec)
                                                   if prec + rec else 0.0)
        else:
            out["changed-class F1 (%)"] = float("nan")
    else:
        exact = err = cnt = 0
        # Two F1s, because they answer different questions and disagree by ~6
        # points. SET F1 asks "did it name the right kinds of object" and is what
        # finetune/evaluate.py reports. MULTISET F1 additionally requires the
        # right NUMBER of each kind, so it inherits the known counting weakness.
        # Reporting only one invites the question of why the other is missing.
        stp = spp = srp = 0
        mtp = mpp = mrp = 0
        for r in records:
            o = loose_json(r["pred_raw"])
            ref = json.loads(r["ref_json"]) if r.get("ref_json") else {}
            gold = ref.get("objects") or []
            if o is None:
                continue
            got = o.get("objects") or []
            cnt += 1
            exact += int(len(got) == len(gold))
            err += abs(len(got) - len(gold))
            pc, rc = obj_classes(got), obj_classes(gold)
            a, b, c = _micro_f1(Counter(set(pc)), Counter(set(rc)))
            stp += a; spp += b; srp += c
            a, b, c = _micro_f1(Counter(pc), Counter(rc))
            mtp += a; mpp += b; mrp += c
        if cnt:
            out["object count exact (%)"] = 100.0 * exact / cnt
            out["object count MAE"] = err / cnt
        for name, (tp, pp, rp) in (("set", (stp, spp, srp)),
                                   ("multiset", (mtp, mpp, mrp))):
            prec = tp / pp if pp else 0.0
            rec = tp / rp if rp else 0.0
            out[f"object-class F1, {name} (%)"] = 100.0 * (
                2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out


# ---------------------------------------------------------------------------
# scoring a set of cached predictions
# ---------------------------------------------------------------------------

CAPTION_ORDER = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR",
                 "ROUGE-1", "ROUGE-2", "ROUGE-L", "CIDEr-D (x100)"]


def score_records(records: list[dict], task: str, bootstrap: int,
                  bertscore_model: str | None, use_bertscore: bool) -> dict:
    hyp_txt = [hypothesis_text(r["pred_raw"], task) for r in records]
    hyps = [tokenize(t) for t in hyp_txt]
    refs = [[tokenize(x) for x in r["refs"]] for r in records]

    stats = [bleu_stats(h, rs) for h, rs in zip(hyps, refs)]
    scores = {k: v * 100 for k, v in bleu_from_stats(stats).items()}
    ci: dict[str, tuple[float, float]] = {}
    for n in range(1, MAX_N + 1):
        lo, hi = boot_ci_bleu(stats, n, bootstrap)
        ci[f"BLEU-{n}"] = (lo * 100, hi * 100)

    per_item = {
        "METEOR": [meteor(h, rs) for h, rs in zip(hyps, refs)],
        "ROUGE-1": [rouge_n(h, rs, 1) for h, rs in zip(hyps, refs)],
        "ROUGE-2": [rouge_n(h, rs, 2) for h, rs in zip(hyps, refs)],
        "ROUGE-L": [rouge_l(h, rs) for h, rs in zip(hyps, refs)],
        "CIDEr-D (x100)": cider_d(hyps, refs),
    }
    bs_note = None
    if use_bertscore:
        got = bert_score_f1(hyp_txt, [r["refs"] for r in records], bertscore_model)
        if got is None:
            bs_note = "not installed -- pip install bert-score"
        else:
            vals, bs_note = got
            per_item["BERTScore F1"] = vals

    for k, vals in per_item.items():
        scores[k] = 100.0 * sum(vals) / len(vals) if vals else 0.0
        lo, hi = boot_ci(vals, bootstrap)
        ci[k] = (lo * 100, hi * 100)

    scores.update(task_metrics(records, task))
    return {"n": len(records), "scores": scores, "ci": ci,
            "bertscore_note": bs_note, "meteor_backend": meteor_backend()}


def metric_order(scores: dict, use_bertscore: bool) -> list[str]:
    order = list(CAPTION_ORDER)
    if use_bertscore and "BERTScore F1" in scores:
        order.insert(order.index("CIDEr-D (x100)") + 1, "BERTScore F1")
    order += [k for k in scores if k not in order]
    return [k for k in order if k in scores]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.2f}"


def _jsonable(v: float):
    """NaN/inf are not valid strict JSON -- a JS reader would throw on them."""
    return None if v is None or v != v or v in (float("inf"), float("-inf")) else v


def render_table(title: str, results: dict[str, dict], use_bertscore: bool,
                 markdown: bool = False) -> str:
    names = [n for n in ("base", "tuned") if n in results]
    if not names:
        return ""
    ref = results[names[-1]]
    order = metric_order(ref["scores"], use_bertscore)
    head = ["metric"] + [{"base": "base model",
                          "tuned": "fine-tuned"}[n] for n in names]
    if len(names) == 2:
        head.append("delta")
    if "tuned" in results:
        head.append("95% CI (fine-tuned)")

    rows = []
    for m in order:
        row = [m]
        vals = {}
        for n in names:
            v = results[n]["scores"].get(m, float("nan"))
            vals[n] = v
            row.append(fmt(v))
        if len(names) == 2:
            d = vals["tuned"] - vals["base"]
            row.append("n/a" if d != d else f"{d:+.2f}")
        if "tuned" in results:
            lo, hi = results["tuned"]["ci"].get(m, (float("nan"),) * 2)
            row.append("--" if lo != lo else f"[{lo:.2f}, {hi:.2f}]")
        rows.append(row)

    if markdown:
        out = [f"### {title}", "",
               "| " + " | ".join(head) + " |",
               "|" + "|".join(["---"] * len(head)) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out) + "\n"

    w = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    line = "  ".join(h.ljust(w[i]) if i == 0 else h.rjust(w[i])
                     for i, h in enumerate(head))
    out = [title, "-" * len(line), line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                             for i, c in enumerate(r)))
    return "\n".join(out)


PREAMBLE = """Higher is better for every row except `object count MAE`.
BLEU / METEOR / ROUGE / BERTScore are shown as percentages; CIDEr-D is scaled
x100, the convention in the LEVIR-CC change-captioning literature.

The base model is scored on its RAW prose (it emits no JSON), which is generous
to it: the caption metrics judge description quality alone. Its inability to
produce the schema shows up only in `valid schema JSON`.

READ THE SUBSETS, NOT JUST THE AGGREGATE. Half of this evaluation set is
unchanged pairs, and LEVIR-CC labels almost all of those with the same handful
of sentences ("the scene is the same as before"). A model that has learned to
recognise no-change scores near-perfect n-gram overlap on that half, which pulls
the `all rows` averages up. The `changed pairs only` block is the honest
difficulty: it is real description against five varied human captions, and it is
the number to quote when comparing against published LEVIR-CC results.

CIDEr document frequencies are computed over this evaluation set, so the
absolute value depends on set size -- compare the columns against each other,
not against a published number. On the `unchanged pairs only` subset CIDEr
collapses toward zero for EVERY model, base and tuned alike: LEVIR-CC gives
near-identical no-change captions to all of those pairs, so their n-grams carry
no inverse document frequency and earn no credit. That is CIDEr working as
designed (it rewards distinctive agreement), not a scoring failure -- read the
change-detection rows there instead."""


# ---------------------------------------------------------------------------
# prediction cache
# ---------------------------------------------------------------------------

def pred_path(task: str, which: str) -> Path:
    return PRED_DIR / f"{task}_{which}.jsonl"


def load_preds(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_preds(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# generation (GPU phase -- needs .venv-unsloth)
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER = {
    "single": PROJECT_ROOT / "adapters" / "report-adapter-v2",
    "bitemporal": PROJECT_ROOT / "adapters" / "bitemporal-adapter",
}


def levircc_refs() -> dict[str, list[str]]:
    """filename -> its 5 human captions.

    LEVIR-CC ships five captions per pair. Scoring against all five is the
    standard protocol and is what published BLEU/CIDEr numbers assume; scoring
    against one would understate every model.
    """
    p = PROJECT_ROOT / "data" / "LevirCCcaptions.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {im["filename"]: [s["raw"].strip() for s in im["sentences"]]
            for im in d.get("images", [])}


def build_eval_rows(task: str, n: int) -> list[dict]:
    """Eval rows carrying both the prompt inputs and the references to cache."""
    if task == "bitemporal":
        from bitemporal.common import CFG, load_split, set_seed

        set_seed()
        val = load_split(CFG.val_jsonl)
        ch = [r for r in val if r["changeflag"] == 1]
        un = [r for r in val if r["changeflag"] == 0]
        half = n // 2
        rows = ch[:half] + un[:half]         # balanced -> chance is exactly 50%
        refmap = levircc_refs()
        for r in rows:
            ans = json.loads(r["answer"])
            r["_refs"] = refmap.get(r["filename"]) or [ans.get("change_summary", "")]
            r["_ref_classes"] = ans.get("changed_classes") or []
            r["_id"] = r["filename"]
        return rows

    from finetune.evaluate import load_rows

    rows = load_rows(n)
    for r in rows:
        ans = json.loads(r["answer"])
        r["_refs"] = [ans.get("caption", "")]
        r["_ref_classes"] = []
        r["_id"] = Path(r["image"]).name
    return rows


def do_generate(args) -> int:
    from unsloth import FastVisionModel                      # noqa: PLC0415
    import torch                                             # noqa: PLC0415

    task = args.task
    adapter = Path(args.adapter or DEFAULT_ADAPTER[task])
    if not adapter.exists():
        print(f"adapter not found: {adapter}")
        return 2

    # Import the SAME generate() the published eval used, so the prompt, image
    # handling and decoding settings are identical and the numbers line up.
    scfg = None
    if task == "bitemporal":
        from bitemporal.common import Mode, free
        from bitemporal.evaluate import generate as gen
    else:
        from finetune.common import CFG as scfg
        from finetune.common import Mode, free
        from finetune.evaluate import generate as gen

    rows = build_eval_rows(task, args.n)
    which = ["tuned", "base"] if args.both else ["tuned"]
    if args.base_only:
        which = ["base"]

    print(RULE)
    print(f"task     : {task}")
    print(f"adapter  : {adapter}")
    print(f"eval set : {len(rows)} rows   references/row: "
          f"{len(rows[0]['_refs']) if rows else 0}")
    print(f"models   : {', '.join(which)}")
    print(RULE)

    free()
    model, processor = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    if task == "single" and scfg is not None:
        # Match the training pixel budget so vision-token counts line up.
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            try:
                ip.size = {"shortest_edge": scfg.min_pixels,
                           "longest_edge": scfg.max_pixels}
            except Exception:                                # noqa: BLE001
                pass

    bad = [nm for nm, p in model.named_parameters()
           if "lora" in nm and not torch.isfinite(p).all()]
    if bad:
        print(f"!! ADAPTER CORRUPTED: {len(bad)} non-finite LoRA tensors {bad[:3]}")
        return 1
    print(f"adapter weight check: all finite | "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    mode = Mode("nf4", torch.bfloat16, torch.bfloat16)

    for label in which:
        out = []
        print(f"\ngenerating [{label}] ...")
        for i, row in enumerate(rows, 1):
            if label == "base":
                with model.disable_adapter():
                    raw = gen(model, processor, mode, row, args.max_new_tokens)
            else:
                raw = gen(model, processor, mode, row, args.max_new_tokens)
            out.append({
                "id": row["_id"], "task": task, "model": label,
                "pred_raw": raw, "refs": row["_refs"],
                "ref_json": row["answer"], "ref_classes": row["_ref_classes"],
                "changeflag": row.get("changeflag"),
            })
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")
        p = pred_path(task, label)
        save_preds(p, out)
        print(f"saved -> {p}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TASK_TITLE = {
    "bitemporal": "BI-TEMPORAL CHANGE CAPTIONING (LEVIR-CC)",
    "single": "SINGLE-IMAGE DESCRIPTION (VRSBench)",
}


def do_score(args) -> int:
    blocks: list[str] = []
    payload: dict = {"meteor_backend": meteor_backend(),
                     "bootstrap": args.bootstrap, "tasks": {}}
    md_tables: list[str] = []
    any_task = False

    for task in args.task:
        cached: dict[str, list[dict]] = {}
        for which in ("base", "tuned"):
            p = pred_path(task, which)
            if p.exists():
                cached[which] = load_preds(p)
        if "tuned" not in cached:
            print(f"[skip] {task}: no predictions at {pred_path(task, 'tuned')}\n"
                  f"       run:  metrics.py generate --task {task} --both")
            continue
        any_task = True

        subsets = [("all rows", lambda r: True)]
        if task == "bitemporal":
            subsets += [("changed pairs only", lambda r: r.get("changeflag") == 1),
                        ("unchanged pairs only", lambda r: r.get("changeflag") == 0)]

        payload["tasks"][task] = {}
        for sub_name, keep in subsets:
            scored: dict[str, dict] = {}
            for which, recs in cached.items():
                rows = [r for r in recs if keep(r)]
                if rows:
                    scored[which] = score_records(rows, task, args.bootstrap,
                                                  args.bertscore_model,
                                                  args.bertscore)
            if not scored:
                continue
            n = scored["tuned"]["n"] if "tuned" in scored else scored["base"]["n"]
            title = f"{TASK_TITLE[task]}  --  {sub_name}, n={n}"
            blocks.append(render_table(title, scored, args.bertscore))
            md_tables.append(render_table(title, scored, args.bertscore,
                                          markdown=True))
            payload["tasks"][task][sub_name] = {
                k: {"n": v["n"],
                    "scores": {m: _jsonable(x) for m, x in v["scores"].items()},
                    "ci95": {m: [_jsonable(lo), _jsonable(hi)]
                             for m, (lo, hi) in v["ci"].items()}}
                for k, v in scored.items()}
            note = scored.get("tuned", {}).get("bertscore_note")
            if args.bertscore and note:
                blocks.append(f"BERTScore model: {note}")

    if not any_task:
        return 2

    mb = meteor_backend()
    header = (f"{RULE}\nSATQUERY -- QUANTITATIVE EVALUATION\n{RULE}\n\n{PREAMBLE}\n\n"
              f"METEOR backend    : {mb}"
              + ("  (exact+stem only, no WordNet synonym stage -- conservative)"
                 if mb == "builtin" else "")
              + f"\nBootstrap resamples: {args.bootstrap}\n")
    print(header)
    for b in blocks:
        print(b + "\n")

    if args.markdown:
        md = ["# satquery -- quantitative evaluation", "", PREAMBLE, "",
              f"METEOR backend: `{mb}` | bootstrap resamples: {args.bootstrap}", ""]
        md += md_tables
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md), encoding="utf-8")
        print(f"markdown -> {out}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json     -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the models, cache predictions (needs GPU)")
    g.add_argument("--task", choices=["single", "bitemporal"], required=True)
    g.add_argument("--adapter", default=None)
    g.add_argument("-n", type=int, default=None,
                   help="eval rows (default: 120 bi-temporal / 60 single)")
    g.add_argument("--max-new-tokens", type=int, default=None)
    g.add_argument("--both", action="store_true",
                   help="also generate base-model output, for the comparison column")
    g.add_argument("--base-only", action="store_true")

    s = sub.add_parser("score", help="score cached predictions (CPU only)")
    s.add_argument("--task", action="append", choices=["single", "bitemporal"],
                   default=None, help="repeatable; default is both")
    s.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples for the 95%% CI; 0 disables")
    s.add_argument("--bertscore", action="store_true",
                   help="also compute BERTScore (needs: pip install bert-score)")
    s.add_argument("--bertscore-model", default=None,
                   help="e.g. microsoft/deberta-xlarge-mnli; default is roberta-large")
    s.add_argument("--markdown", default=None, help="write a markdown table here")
    s.add_argument("--json", default=None, help="write the raw numbers here")

    args = ap.parse_args()
    if args.cmd == "generate":
        if args.n is None:
            args.n = 120 if args.task == "bitemporal" else 60
        if args.max_new_tokens is None:
            args.max_new_tokens = 320 if args.task == "bitemporal" else 512
        return do_generate(args)
    if not args.task:
        args.task = ["bitemporal", "single"]
    return do_score(args)


if __name__ == "__main__":
    sys.exit(main())
