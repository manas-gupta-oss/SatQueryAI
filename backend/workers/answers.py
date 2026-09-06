"""
Turn a specialist's structured JSON into the answer the user actually asked for.

This module is the concrete form of rule 1 in models/MODELS.md: *never pass the
user's query to the adapters*. Both were fine-tuned on exactly one prompt each
and learned image -> schema, not question -> answer; overriding the prompt
measurably destroys the schema and with it the visual evidence.

So the query is applied HERE instead, to the JSON the adapter produced:

    "how many ships are there?"   -> count over analysis.objects
    "highlight the harbour"       -> filter analysis.objects, return their boxes
    "what changed?"               -> analysis.change_summary + extent + regions

Everything below is string and list manipulation over an already-generated
envelope. No model runs in this file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from orchestration.state import TaskParams, TaskType

Boxes = List[List[float]]
Labels = List[str]
Derived = Tuple[str, Boxes, Labels]

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of", "to",
    "this", "that", "these", "those", "there", "here", "it", "its", "and", "or",
    "do", "does", "did", "can", "you", "me", "my", "i", "please", "show", "tell",
    "image", "images", "scene", "picture", "photo", "satellite", "any", "some",
    "what", "which", "how", "many", "much", "where", "who", "when", "why",
}

_NUMBER_WORDS = {
    0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOPWORDS]


def _singular(word: str) -> str:
    """Crude depluraliser - enough to match 'ships' against obj_cls 'ship'."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxzh":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stems(text: str) -> set:
    return {_singular(w) for w in _content_words(text)}


def _label_of(obj: Dict[str, Any]) -> str:
    cls = (obj.get("obj_cls") or "").strip()
    return cls or "object"


def _boxes_from(items: List[Dict[str, Any]], label_key: str) -> Derived:
    boxes: Boxes = []
    labels: Labels = []
    for item in items:
        box = item.get("bbox_normalized")
        if isinstance(box, list) and len(box) == 4:
            boxes.append([float(v) for v in box])
            labels.append(str(item.get(label_key) or "region").strip() or "region")
    return "", boxes, labels


def _count_phrase(n: int, noun: str) -> str:
    word = _NUMBER_WORDS.get(n, str(n))
    plural = noun if n == 1 else (noun if noun.endswith("s") else f"{noun}s")
    return f"{word} {plural}"


def _object_inventory(objects: List[Dict[str, Any]]) -> str:
    """'three ships, one harbour' - the extracted objects, grouped by class."""
    counts: Dict[str, int] = {}
    for obj in objects:
        cls = _label_of(obj)
        counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return ""
    parts = [_count_phrase(n, cls) for cls, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return ", ".join(parts)


def _match_objects(phrase: str, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Objects whose class or referring sentence overlaps the phrase's stems."""
    wanted = _stems(phrase)
    if not wanted:
        return []
    matched = []
    for obj in objects:
        haystack = _stems(f"{_label_of(obj)} {obj.get('referring_sentence', '')}")
        if wanted & haystack:
            matched.append(obj)
    return matched


def _best_qa_pair(question: str, qa_pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The adapter emits its own QA pairs. If one of them is close enough to what
    the user asked, it is a real answer rather than a paraphrase of the caption.
    """
    wanted = _stems(question)
    if not wanted:
        return None
    best, best_score = None, 0.0
    for pair in qa_pairs:
        candidate = _stems(pair.get("question", ""))
        if not candidate:
            continue
        overlap = len(wanted & candidate) / len(wanted | candidate)
        if overlap > best_score:
            best, best_score = pair, overlap
    # Jaccard 0.34 is roughly "a third of the content words agree". Below that
    # the pair is about something else and quoting it would be misleading.
    return best if best_score >= 0.34 else None


# --------------------------------------------------------------------------- #
# Single-image tasks
# --------------------------------------------------------------------------- #


def _answer_counting(question: str, objects: List[Dict[str, Any]]) -> Optional[str]:
    """Handle 'how many X' directly against the extracted object list."""
    match = re.search(r"how many\s+([a-z][a-z\s-]{0,40})", question.lower())
    if not match:
        return None
    noun = match.group(1).strip().strip("?.!,")
    matched = _match_objects(noun, objects)
    if not matched:
        return (
            f"No {noun} were extracted from this scene. The specialist identified "
            f"{_object_inventory(objects) or 'no objects'}."
        )
    label = _label_of(matched[0])
    return (
        f"{_count_phrase(len(matched), label).capitalize()} were extracted from this scene."
        if len(matched) != 1
        else f"One {label} was extracted from this scene."
    )


def derive_single(task: TaskType, params: TaskParams, analysis: Dict[str, Any]) -> Derived:
    caption = (analysis.get("caption") or "").strip()
    objects = analysis.get("objects") or []
    qa_pairs = analysis.get("qa_pairs") or []

    if task is TaskType.GROUNDING:
        phrase = (params.target_phrase or "").strip()
        matched = _match_objects(phrase, objects)
        if matched:
            _, boxes, labels = _boxes_from(matched, "obj_cls")
            positions = ", ".join(
                f"{_label_of(o)} ({o.get('obj_position') or 'position not stated'})"
                for o in matched
            )
            text = (
                f"Located {_count_phrase(len(matched), _label_of(matched[0]))} matching "
                f"'{phrase}': {positions}. Bounding boxes are drawn on the annotated image."
            )
            return text, boxes, labels
        _, boxes, labels = _boxes_from(objects, "obj_cls")
        inventory = _object_inventory(objects)
        text = (
            f"Nothing matching '{phrase}' was extracted from this scene. "
            + (
                f"The specialist did localise {inventory}; those boxes are shown instead."
                if inventory
                else "No localisable objects were extracted from this scene."
            )
        )
        return text, boxes, labels

    if task is TaskType.SINGLE_VQA:
        question = (params.question or "").strip()
        _, boxes, labels = _boxes_from(objects, "obj_cls")

        counted = _answer_counting(question, objects)
        if counted:
            # Counting is the model's documented weak point - say so rather than
            # presenting a count as reliable. See "Honest limits" in MODELS.md.
            return (
                counted + " Object counting is the least reliable part of this "
                "model's output; treat the count as indicative.",
                boxes,
                labels,
            )

        pair = _best_qa_pair(question, qa_pairs)
        if pair:
            return (
                f"{pair['answer']}\n\n(Answered from the specialist's own extracted "
                f"question-answer pair: \"{pair['question']}\")",
                boxes,
                labels,
            )

        matched = _match_objects(question, objects)
        if matched:
            positions = "; ".join(
                f"{_label_of(o)} - {o.get('obj_position') or 'position not stated'}, "
                f"{o.get('obj_size') or 'size not stated'}"
                for o in matched
            )
            return (
                f"{caption}\n\nRelevant objects extracted for your question: {positions}.",
                boxes,
                labels,
            )

        suffix = (
            "\n\nThe scene description above is the specialist's full output; it did not "
            "extract anything specific to this question."
        )
        return (caption + suffix if caption else "The specialist produced no usable description."), boxes, labels

    # CAPTIONING
    _, boxes, labels = _boxes_from(objects, "obj_cls")
    inventory = _object_inventory(objects)
    text = caption or "The specialist produced no usable description."
    if inventory:
        text += f"\n\nObjects extracted and localised: {inventory}."
    if qa_pairs:
        text += (
            f"\n\n{_count_phrase(len(qa_pairs), 'question-answer pair').capitalize()} "
            f"were also extracted from the scene."
        )
    return text, boxes, labels


# --------------------------------------------------------------------------- #
# Bi-temporal tasks
# --------------------------------------------------------------------------- #


def derive_change(task: TaskType, params: TaskParams, analysis: Dict[str, Any]) -> Derived:
    detected = bool(analysis.get("change_detected"))
    summary = (analysis.get("change_summary") or "").strip()
    classes = analysis.get("changed_classes") or []
    regions = analysis.get("change_regions") or []
    extent = (analysis.get("change_extent") or "").strip()

    _, boxes, labels = _boxes_from(regions, "class")

    headline = summary or (
        "Change was detected between the two acquisitions."
        if detected
        else "No change was detected between the two acquisitions."
    )

    detail: List[str] = []
    if classes:
        detail.append(f"Changed feature classes: {', '.join(classes)}.")
    if extent:
        detail.append(f"Change extent: {extent}.")
    if regions:
        detail.append(
            f"{_count_phrase(len(regions), 'change region').capitalize()} localised and "
            f"drawn on the annotated 'after' image."
        )

    if task is TaskType.CHANGE_VQA:
        question = (params.question or "").strip()
        focus = (params.change_focus or "").strip()
        if focus:
            wanted = _stems(focus)
            hit = [c for c in classes if wanted & _stems(c)]
            if hit:
                lead = f"Yes - {', '.join(hit)} changed between the two dates."
            elif detected:
                lead = (
                    f"No change was reported for '{focus}'. The change that was detected "
                    f"involves {', '.join(classes) if classes else 'unspecified features'}."
                )
            else:
                lead = f"No change was detected for '{focus}', or anywhere else in the scene."
            return "\n\n".join([lead, headline] + ([" ".join(detail)] if detail else [])), boxes, labels
        if question:
            # A yes/no question with no extractable subject still deserves a direct
            # answer, given against what the specialist actually reported.
            lead = (
                "Yes - change was detected between the two acquisitions."
                if detected
                else "No - no change was detected between the two acquisitions."
            )
            detail.append(
                "This answer is derived from the change-detection output above; the "
                "specialist analyses the image pair, not the question text."
            )
            return "\n\n".join([lead, headline] + [" ".join(detail)]), boxes, labels

    body = "\n\n".join([headline] + ([" ".join(detail)] if detail else []))
    return body, boxes, labels
