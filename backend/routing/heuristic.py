"""
Deterministic rule router - a drop-in BOSS that needs no model.

This is a faithful implementation of the routing rules R1-R5 and validation
rules V1-V7 written down in orchestration/nodes/boss_node.py._ROUTING_RULES,
expressed as code instead of as a prompt. It exists for two reasons:

  * a live demo cannot afford a 40 s model load and a VRAM gamble on a 6 GB
    laptop that is already holding the vision specialists;
  * it makes the routing layer testable and reproducible without weights.

It is wired in exactly where the real BOSS goes, via
`boss_node.make_boss_node(...)`, and its output passes through the *same*
deterministic gate (`tool_schema.validate_decision`) that constrains the 3B
router. Set SATQUERY_ROUTER=llm to swap the Qwen2.5-3B BOSS back in; nothing
else in the stack changes.

What it does NOT do: understand paraphrase, resolve genuinely novel phrasings,
or reason about anything outside these keyword families. That is the honest
trade for determinism, and it is why the LLM path is kept alive rather than
deleted.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from orchestration.state import (
    BossDecision,
    ImageMeta,
    InputBundle,
    Modality,
    TaskParams,
    TaskType,
    Validation,
    ValidationCode,
    ValidationStatus,
    WorkerId,
)
from orchestration.tool_schema import OPTICAL_LIKE, rejection_decision, validate_decision

# --------------------------------------------------------------------------- #
# Keyword families
# --------------------------------------------------------------------------- #

CHANGE_WORDS = {
    "change", "changed", "changes", "changing", "difference", "differences",
    "differ", "compare", "comparison", "compared", "before", "after",
    "since", "grew", "grow", "growth", "expansion", "expanded",
    "added", "removed", "demolished", "constructed", "construction",
    "built", "deforestation", "urbanisation", "urbanization", "temporal",
    "earlier", "later", "over time", "bi-temporal", "bitemporal",
}

#: The subset that is unambiguously bi-temporal. A single image plus one of
#: these is a hard image-count mismatch, not a captioning request.
STRICT_CHANGE_WORDS = {
    "change", "changed", "changes", "difference", "differences", "compare",
    "comparison", "compared", "before", "after", "bi-temporal", "bitemporal",
    "over time",
}

GROUNDING_WORDS = {
    "locate", "localise", "localize", "highlight", "where", "find", "mark",
    "point out", "show me", "bounding box", "bounding boxes", "pinpoint",
}

#: Phrases that ask for a description of the whole scene. The multi-word entries
#: are deliberately anchored to "this/the image": bare "what is in" would also
#: swallow "what is in the top right of the image?", which is a specific VQA
#: question about a region, not a request to caption the scene.
CAPTION_WORDS = {
    "describe", "description", "caption", "summarise", "summarize", "summary",
    "overview", "tell me about", "report on", "analyse", "analyze",
    "what is in this image", "what is in the image", "what's in this image",
    "what's in the image", "what does this show", "what do you see",
}

QUESTION_LEADS = (
    "what", "how", "which", "who", "where", "when", "why", "is", "are", "does",
    "do", "did", "can", "could", "has", "have", "was", "were", "any",
)

MAP_WORDS = {"where", "map", "location", "locate", "region", "regions", "area", "areas"}

#: Yes/no auxiliaries. R4 separates a *specific* change question ("has X
#: increased?", "how many Y were added?") from an *open* one ("what changed?",
#: "describe the differences"). A yes/no lead or a quantity lead is what makes a
#: change query specific; "what changed?" leads with an interrogative pronoun and
#: names nothing, so it is the open case and belongs to change_description.
YESNO_LEADS = {"is", "are", "was", "were", "has", "have", "had", "did", "do",
               "does", "can", "could", "any"}
_QUANTITY_LEAD = re.compile(r"^\s*how\s+(?:many|much)\b", re.IGNORECASE)

#: Verbs stripped off the front of a grounding query to recover the target phrase.
_GROUNDING_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:highlight|locate|localise|localize|find|mark|show me|point out|identify|"
    r"draw a box around|draw boxes around|where\s+(?:is|are|can i find))\s+",
    re.IGNORECASE,
)
_TRAILING_NOISE = re.compile(
    r"\b(?:in|on|within|inside)\s+(?:this|the)\s+(?:image|scene|picture)\b.*$",
    re.IGNORECASE,
)

_FILLER_PHRASES = {"it", "this", "that", "them", "the image", "the scene", "the picture"}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _mentions(text: str, vocabulary) -> bool:
    lowered = text.lower()
    words = set(_tokens(lowered))
    for entry in vocabulary:
        if " " in entry or "-" in entry:
            if entry in lowered:
                return True
        elif entry in words:
            return True
    return False


def _looks_like_a_question(query: str) -> bool:
    stripped = query.strip().lower()
    if "?" in stripped:
        return True
    first = _tokens(stripped)[:1]
    return bool(first) and first[0] in QUESTION_LEADS


def extract_target_phrase(query: str) -> Optional[str]:
    """Recover the object phrase a grounding request wants localised."""
    phrase = _GROUNDING_PREFIX.sub("", query.strip(), count=1)
    phrase = _TRAILING_NOISE.sub("", phrase)
    phrase = phrase.strip().strip("?.!,;:").strip()
    if not phrase or phrase.lower() in _FILLER_PHRASES:
        return None
    return phrase


def extract_change_focus(query: str) -> Optional[str]:
    """
    The subject of a change question, best-effort: 'has the built-up area grown?'
    -> 'built-up area'. Returns None rather than guessing badly.
    """
    match = re.search(
        r"\b(?:how much|how many|has|have|did|is|are|was|were)\s+"
        r"(?:the\s+|any\s+|new\s+)?([a-z][a-z\s-]{2,40}?)\s*"
        r"(?:been\s+)?(?:changed|change|grown|grow|increased|decreased|added|"
        r"removed|built|appear|appeared|expanded|expand|shrunk|shrank|shrunken|"
        r"reduced|declined|disappeared|vanished|cleared|widened|developed|"
        r"demolished|constructed)\b",
        query.lower(),
    )
    if match:
        focus = match.group(1).strip()
        if focus and focus not in {"there", "it", "this", "that"}:
            return focus
    return None


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #


def _is_specific_change_question(query: str, focus: Optional[str]) -> bool:
    """
    Does this pair query name something in particular to answer about (change_vqa),
    or is it an open "what is different?" (change_description)?

    Being wrong in the open direction is cheap - a change description answers
    "what changed?" completely. Being wrong in the specific direction produces a
    yes/no answer to a question the user did not ask, so the bar for 'specific'
    is deliberately the higher one.
    """
    if _mentions(query, CAPTION_WORDS):
        return False
    if focus:
        return True
    if _QUANTITY_LEAD.match(query):
        return True
    lead = _tokens(query)[:1]
    return bool(lead) and lead[0] in YESNO_LEADS


def _split_modalities(images: List[ImageMeta]) -> Tuple[List[ImageMeta], List[ImageMeta]]:
    optical = [i for i in images if i.modality in OPTICAL_LIKE or i.modality is Modality.UNKNOWN]
    sar = [i for i in images if i.modality is Modality.SAR]
    return optical, sar


def _decide_single(query: str, bundle: InputBundle) -> BossDecision:
    image = bundle.images[0]

    # R2 - a lone SAR scene belongs to worker2, never to the VRSBench specialist.
    if image.modality is Modality.SAR:
        restated = _mentions(query, CAPTION_WORDS) and not _looks_like_a_question(query)
        return BossDecision(
            task_type=TaskType.SINGLE_VQA,
            target_worker=WorkerId.WORKER2,
            params=TaskParams(question=query.strip() or "Describe this SAR scene."),
            validation=Validation(status=ValidationStatus.PASS),
            audit_summary=(
                "Single SAR image with no optical counterpart. worker1 is tuned on optical "
                "imagery only, so the query is answered as single-image VQA on worker2, the "
                "SAR-capable specialist."
            ),
            confidence=0.85,
            assumptions=(
                ["worker2 does not caption; the request was restated as a question"]
                if restated
                else []
            ),
        )

    # R1 - grounding, but only when there is actually a phrase to localise (V6).
    if _mentions(query, GROUNDING_WORDS):
        phrase = extract_target_phrase(query)
        if phrase:
            return BossDecision(
                task_type=TaskType.GROUNDING,
                target_worker=WorkerId.WORKER1,
                params=TaskParams(target_phrase=phrase, return_visual_evidence=True),
                validation=Validation(status=ValidationStatus.PASS),
                audit_summary=(
                    "The query asks for a location, so it was routed to worker1 as text-guided "
                    "grounding with target_phrase set to the feature named in the query."
                ),
                confidence=0.82,
            )
        return rejection_decision(
            ValidationCode.MISSING_REQUIRED_PARAM,
            "the query asks to locate something but names no object to localise; "
            "name the feature, for example 'highlight the ships in the harbour'",
            task=TaskType.GROUNDING,
        )

    # R1 - a specific question about content.
    if _looks_like_a_question(query) and not _mentions(query, CAPTION_WORDS):
        return BossDecision(
            task_type=TaskType.SINGLE_VQA,
            target_worker=WorkerId.WORKER1,
            params=TaskParams(question=query.strip()),
            validation=Validation(status=ValidationStatus.PASS),
            audit_summary=(
                "One optical image and a specific question about its content, so the query was "
                "routed to worker1 as single-image visual question answering."
            ),
            confidence=0.86,
        )

    # R1 - open description. Also the fallback for anything unclassified, which
    # is the safe default: captioning needs no parameters and cannot misread the
    # request, only under-serve it.
    explicit = _mentions(query, CAPTION_WORDS)
    return BossDecision(
        task_type=TaskType.CAPTIONING,
        target_worker=WorkerId.WORKER1,
        params=TaskParams(detail_level="detailed", return_visual_evidence=True),
        validation=Validation(status=ValidationStatus.PASS),
        audit_summary=(
            "One optical image and an open-ended request, so the query was routed to worker1 for "
            "detailed scene captioning with object grounding."
            if explicit
            else "One optical image and a query that names no specific task, so it defaults to "
            "worker1 scene captioning, the most informative answer that cannot misread the "
            "request."
        ),
        confidence=0.8 if explicit else 0.5,
        assumptions=(
            []
            if explicit
            else ["the request was open-ended; a full scene description was produced"]
        ),
    )


def _decide_pair(query: str, bundle: InputBundle) -> BossDecision:
    optical, sar = _split_modalities(bundle.images)
    wants_change = _mentions(query, CHANGE_WORDS)

    # R3 - one optical + one SAR and no temporal wording is a fusion request.
    if sar and optical and not wants_change:
        return BossDecision(
            task_type=TaskType.CROSS_MODAL_FUSION,
            target_worker=WorkerId.WORKER2,
            params=TaskParams(question=query.strip() or None, return_visual_evidence=True),
            validation=Validation(status=ValidationStatus.PASS),
            audit_summary=(
                "The pair is one optical and one SAR image of the same area with no temporal "
                "wording, so it was routed to worker2 for cross-modal fusion."
            ),
            confidence=0.84,
            image_assignment={"optical": optical[0].image_id, "sar": sar[0].image_id},
        )

    # R4 - everything else about a pair is change analysis.
    focus = extract_change_focus(query)
    wants_map = _mentions(query, MAP_WORDS)
    specific = _is_specific_change_question(query, focus)

    if specific:
        return BossDecision(
            task_type=TaskType.CHANGE_VQA,
            target_worker=WorkerId.WORKER3,
            params=TaskParams(
                question=query.strip(),
                change_focus=focus,
                return_change_map=wants_map,
                return_visual_evidence=True,
            ),
            validation=Validation(status=ValidationStatus.PASS),
            audit_summary=(
                "Two images of the same area and a specific question about what differs between "
                "them, so the query was routed to worker3 as change-based VQA"
                + (" focused on the feature named in the query." if focus else ".")
            ),
            confidence=0.85,
        )

    return BossDecision(
        task_type=TaskType.CHANGE_DESCRIPTION,
        target_worker=WorkerId.WORKER3,
        params=TaskParams(
            change_focus=focus,
            detail_level="detailed",
            return_change_map=wants_map,
            return_visual_evidence=True,
        ),
        validation=Validation(status=ValidationStatus.PASS),
        audit_summary=(
            "Two images of the same area and an open request about what differs, so the query "
            "was routed to worker3 for change description."
        ),
        confidence=0.83 if wants_change else 0.6,
        assumptions=(
            []
            if wants_change
            else ["two images were supplied with no explicit task; bi-temporal change was assumed"]
        ),
    )


def heuristic_decide(query: str, bundle: InputBundle) -> BossDecision:
    """
    (query, bundle) -> BossDecision, the same signature the LLM router satisfies.

    The returned decision always goes through validate_decision(), so a routing
    bug here cannot dispatch an incompatible worker - it can only produce a
    rejection carrying a specific reason.
    """
    query = (query or "").strip()
    count = bundle.image_count

    # V7
    if count == 0:
        return rejection_decision(
            ValidationCode.NO_IMAGES,
            "no image was supplied; every task requires at least one image",
        )

    # V1 - the registry caps every worker at two images, so three has no route.
    if count > 2:
        return rejection_decision(
            ValidationCode.IMAGE_COUNT_MISMATCH,
            f"{count} images were supplied; every specialist accepts at most two",
        )

    # V1 - a change request against a single image is the classic mismatch.
    if count == 1 and _mentions(query, STRICT_CHANGE_WORDS):
        return rejection_decision(
            ValidationCode.IMAGE_COUNT_MISMATCH,
            "change analysis compares two acquisitions of the same area, but only one image "
            "was supplied; upload the earlier and the later image",
            task=TaskType.CHANGE_DESCRIPTION,
        )

    decision = _decide_single(query, bundle) if count == 1 else _decide_pair(query, bundle)

    # The same hard gate the 3B router's output is put through. V2-V5 are
    # enforced there rather than duplicated here, so the rules keep one home.
    return validate_decision(decision, bundle)
