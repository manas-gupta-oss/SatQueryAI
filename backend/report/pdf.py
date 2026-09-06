"""
Render one query's result as a PDF report.

The report is assembled from the same envelope the specialists return, so what
the PDF states and what the API returned cannot drift apart. Three editorial
rules, all inherited from models/MODELS.md:

  * boxes are drawn from `bbox_normalized`, never the raw coordinates;
  * the self-consistency verdict is reported as self-consistency, never as
    "verified correct" - nothing downstream of the model can check a claim
    against the actual pixels;
  * the measured limits (object counting is unreliable; wording differs from
    reference even when substantively right) are printed in the report itself
    rather than left for the reader to discover.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Palette and styles
# --------------------------------------------------------------------------- #

INK = colors.HexColor("#131820")
INK_SOFT = colors.HexColor("#454f5e")
INK_FAINT = colors.HexColor("#78828f")
ACCENT = colors.HexColor("#0f6fbe")
ACCENT_SOFT = colors.HexColor("#eaf2fb")
LINE = colors.HexColor("#d7dde5")
OK = colors.HexColor("#1a7f52")
WARN = colors.HexColor("#a86500")
BAD = colors.HexColor("#b3261e")

PAGE_MARGIN = 16 * mm
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN

SEVERITY_COLOURS = {"ok": OK, "warn": WARN, "reject": BAD}


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "SQBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
        leading=14, textColor=INK, alignment=TA_LEFT, spaceAfter=0,
    )
    return {
        "title": ParagraphStyle(
            "SQTitle", parent=body, fontName="Helvetica-Bold", fontSize=19,
            leading=23, textColor=INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "SQSubtitle", parent=body, fontSize=9, leading=12, textColor=INK_FAINT,
        ),
        "h2": ParagraphStyle(
            "SQH2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=15,
            textColor=INK, spaceBefore=2, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "SQH3", parent=body, fontName="Helvetica-Bold", fontSize=9.5, leading=13,
            textColor=INK_SOFT, spaceBefore=2, spaceAfter=2,
        ),
        "body": body,
        "small": ParagraphStyle(
            "SQSmall", parent=body, fontSize=8.2, leading=11.5, textColor=INK_SOFT,
        ),
        "faint": ParagraphStyle(
            "SQFaint", parent=body, fontSize=7.8, leading=11, textColor=INK_FAINT,
        ),
        "mono": ParagraphStyle(
            "SQMono", parent=body, fontName="Courier", fontSize=7.6, leading=10.6,
            textColor=INK_SOFT,
        ),
        "cell": ParagraphStyle(
            "SQCell", parent=body, fontSize=8.4, leading=11.5,
        ),
        "cellhead": ParagraphStyle(
            "SQCellHead", parent=body, fontName="Helvetica-Bold", fontSize=8.2,
            leading=11, textColor=colors.white,
        ),
        "lead": ParagraphStyle(
            "SQLead", parent=body, fontSize=10.5, leading=15.5, textColor=INK,
        ),
    }


def _esc(value: Any) -> str:
    return escape("" if value is None else str(value))


def _plain(value: Any) -> str:
    """
    Enum members reach the PDF as Python objects, because finalize_node calls
    model_dump() without a JSON round-trip. str(ValidationStatus.PASS) renders as
    'ValidationStatus.PASS', so unwrap to the value the user should see.
    """
    if value is None:
        return ""
    return str(getattr(value, "value", value))


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def _section(title: str, st: Dict[str, ParagraphStyle]) -> List[Any]:
    return [
        Spacer(1, 9),
        Paragraph(_esc(title).upper(), st["h2"]),
        HRFlowable(width="100%", thickness=0.8, color=LINE, spaceBefore=1, spaceAfter=6),
    ]


def _kv_table(rows: Sequence[Sequence[str]], st: Dict[str, ParagraphStyle],
              key_width: float = 38 * mm) -> Table:
    data = [
        [Paragraph(f"<b>{_esc(k)}</b>", st["small"]), Paragraph(_esc(v), st["cell"])]
        for k, v in rows
    ]
    table = Table(data, colWidths=[key_width, CONTENT_WIDTH - key_width])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]))
    return table


def _grid_table(header: Sequence[str], rows: Sequence[Sequence[str]],
                widths: Sequence[float], st: Dict[str, ParagraphStyle]) -> Table:
    data = [[Paragraph(_esc(h), st["cellhead"]) for h in header]]
    data += [[Paragraph(_esc(c), st["cell"]) for c in row] for row in rows]
    table = Table(data, colWidths=list(widths), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
    ]))
    return table


def _callout(text: str, st: Dict[str, ParagraphStyle], accent=ACCENT,
             background=ACCENT_SOFT, style: str = "body") -> Table:
    table = Table([[Paragraph(text, st[style])]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, accent),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def _figure(path: Path, caption: str, st: Dict[str, ParagraphStyle],
            max_width: float = CONTENT_WIDTH, max_height: float = 105 * mm) -> List[Any]:
    """An image scaled to fit the frame, with its caption kept on the same page."""
    try:
        with PILImage.open(path) as probe:
            width, height = probe.size
    except Exception as exc:
        logger.warning("could not place %s in the report: %s", path, exc)
        return []
    if not width or not height:
        return []
    scale = min(max_width / width, max_height / height)
    flowable = RLImage(str(path), width=width * scale, height=height * scale)
    flowable.hAlign = "LEFT"
    return [KeepTogether([flowable, Spacer(1, 3), Paragraph(_esc(caption), st["faint"])])]


def _bbox_text(box: Optional[Sequence[float]]) -> str:
    if not box or len(box) != 4:
        return "not localised"
    return "[" + ", ".join(f"{float(v):.3f}" for v in box) + "]"


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #


def _make_page_decorator(request_id: str):
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(INK_FAINT)
        canvas.drawString(PAGE_MARGIN, 10 * mm, f"SatQueryAI  ·  {request_id}")
        canvas.drawRightString(A4[0] - PAGE_MARGIN, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(PAGE_MARGIN, 13 * mm, A4[0] - PAGE_MARGIN, 13 * mm)
        canvas.restoreState()

    return decorate


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #


def _header(response: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    banner = Table(
        [[Paragraph("<b>SatQueryAI</b>", ParagraphStyle(
            "SQBrand", parent=st["body"], fontSize=13, leading=16, textColor=colors.white))]],
        colWidths=[CONTENT_WIDTH],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [
        banner,
        Spacer(1, 10),
        Paragraph("Satellite Imagery Analysis Report", st["title"]),
        Paragraph(
            f"Generated {_esc(generated)} &nbsp;·&nbsp; Request {_esc(response.get('request_id'))}",
            st["subtitle"],
        ),
        Spacer(1, 8),
    ]


def _routing_section(response: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    validation = response.get("validation") or {}
    router_confidence = response.get("router_confidence")
    rows = [
        ("Query", response.get("query") or "-"),
        ("Task selected", (response.get("task_type") or "none").replace("_", " ")),
        ("Specialist dispatched", response.get("worker") or "none - request rejected"),
        ("Routing validation",
         f"{_plain(validation.get('status')) or 'n/a'}"
         + (f" ({_plain(validation.get('code'))})" if validation.get("code") else "")),
        ("Router confidence",
         f"{router_confidence:.0%}" if isinstance(router_confidence, (int, float)) else "not reported"),
    ]
    flow = _section("Query and routing decision", st)
    flow.append(_kv_table(rows, st))
    if response.get("audit_summary"):
        flow += [Spacer(1, 7), _callout(
            f"<b>Why this route:</b> {_esc(response['audit_summary'])}", st, style="small")]
    assumptions = response.get("assumptions") or []
    if assumptions:
        flow += [Spacer(1, 7), Paragraph("Assumptions made", st["h3"])]
        for item in assumptions:
            flow.append(Paragraph(f"•&nbsp; {_esc(item)}", st["small"]))
    return flow


def _imagery_section(
    image_records: Sequence[Any], overlay_path: Optional[Path], strip_path: Optional[Path],
    st: Dict[str, ParagraphStyle],
) -> List[Any]:
    flow = _section("Imagery analysed", st)

    rows = [
        [
            record.filename,
            record.modality.value,
            record.fmt.value,
            f"{record.width} × {record.height}" if record.width else "unknown",
            record.timestamp or "not supplied",
        ]
        for record in image_records
    ]
    flow.append(_grid_table(
        ["File", "Modality", "Format", "Pixels", "Acquired"],
        rows,
        [CONTENT_WIDTH * 0.32, CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.13,
         CONTENT_WIDTH * 0.18, CONTENT_WIDTH * 0.23],
        st,
    ))
    flow.append(Spacer(1, 10))

    if strip_path and strip_path.exists():
        flow += _figure(strip_path, "Figure 1 — The image pair, earlier acquisition on the left.", st)
        flow.append(Spacer(1, 8))
    if overlay_path and overlay_path.exists():
        flow += _figure(
            overlay_path,
            "Figure — Model output overlaid on the imagery. Boxes are drawn from the "
            "clamped normalised coordinates the specialist emitted.",
            st,
        )
    elif not strip_path:
        for record in image_records:
            flow += _figure(Path(record.path), f"Figure — {record.filename}", st)
            flow.append(Spacer(1, 6))
    return flow


def _findings_section(response: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow = _section("Findings", st)
    answer = (response.get("answer") or "").strip()
    if not answer:
        answer = response.get("error") or "No answer was produced."
    for block in [b for b in answer.split("\n\n") if b.strip()]:
        flow.append(Paragraph(_esc(block.strip()).replace("\n", "<br/>"), st["lead"]))
        flow.append(Spacer(1, 6))
    return flow


def _single_detail_section(analysis: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    flow: List[Any] = []
    caption = (analysis.get("caption") or "").strip()
    objects = analysis.get("objects") or []
    qa_pairs = analysis.get("qa_pairs") or []

    flow += _section("Scene description", st)
    flow.append(Paragraph(
        _esc(caption) if caption else "The specialist produced no caption.", st["body"]))

    flow += _section(f"Objects extracted ({len(objects)})", st)
    if objects:
        rows = [
            [
                str(index + 1),
                obj.get("obj_cls") or "unlabelled",
                obj.get("obj_position") or "-",
                obj.get("obj_size") or "-",
                _bbox_text(obj.get("bbox_normalized")),
            ]
            for index, obj in enumerate(objects)
        ]
        flow.append(_grid_table(
            ["#", "Class", "Position", "Size", "Bounding box (x0, y0, x1, y1)"],
            rows,
            [CONTENT_WIDTH * 0.06, CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.19,
             CONTENT_WIDTH * 0.13, CONTENT_WIDTH * 0.40],
            st,
        ))
        flow.append(Spacer(1, 5))
        flow.append(Paragraph(
            "Coordinates are normalised to the image frame (0 = left/top, 1 = right/bottom) "
            "and clamped, so they can be scaled to any display size.", st["faint"]))
        referring = [o for o in objects if (o.get("referring_sentence") or "").strip()]
        if referring:
            flow.append(Spacer(1, 9))
            flow.append(Paragraph("Referring expressions", st["h3"]))
            for index, obj in enumerate(objects):
                sentence = (obj.get("referring_sentence") or "").strip()
                if sentence:
                    flow.append(Paragraph(
                        f"<b>{index + 1}. {_esc(obj.get('obj_cls') or 'object')}</b> — "
                        f"{_esc(sentence)}", st["small"]))
    else:
        flow.append(Paragraph("No objects were extracted from this scene.", st["body"]))

    if qa_pairs:
        flow += _section(f"Question-answer pairs generated ({len(qa_pairs)})", st)
        flow.append(_grid_table(
            ["Question", "Answer", "Type"],
            [[q.get("question") or "-", q.get("answer") or "-", q.get("type") or "-"]
             for q in qa_pairs],
            [CONTENT_WIDTH * 0.47, CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.20],
            st,
        ))
        flow.append(Spacer(1, 5))
        flow.append(Paragraph(
            "These pairs are produced by the specialist from the image alone; they are not "
            "answers to the user's query.", st["faint"]))
    return flow


def _change_detail_section(analysis: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    detected = bool(analysis.get("change_detected"))
    regions = analysis.get("change_regions") or []
    classes = analysis.get("changed_classes") or []

    flow = _section("Change analysis", st)
    verdict = "CHANGE DETECTED" if detected else "NO CHANGE DETECTED"
    flow.append(_callout(
        f"<b>{verdict}</b>",
        st,
        accent=WARN if detected else OK,
        background=colors.HexColor("#fdf4e3") if detected else colors.HexColor("#eaf6f0"),
    ))
    flow.append(Spacer(1, 8))
    flow.append(_kv_table([
        ("Change summary", analysis.get("change_summary") or "-"),
        ("Changed classes", ", ".join(classes) if classes else "none reported"),
        ("Change extent", analysis.get("change_extent") or "not reported"),
        ("Regions localised", str(len(regions))),
    ], st))

    if regions:
        flow += _section(f"Change regions ({len(regions)})", st)
        flow.append(_grid_table(
            ["#", "Class", "Size", "Bounding box (x0, y0, x1, y1)"],
            [[str(i + 1), r.get("class") or "unlabelled", r.get("size") or "-",
              _bbox_text(r.get("bbox_normalized"))] for i, r in enumerate(regions)],
            [CONTENT_WIDTH * 0.07, CONTENT_WIDTH * 0.28, CONTENT_WIDTH * 0.17,
             CONTENT_WIDTH * 0.48],
            st,
        ))
        flow.append(Spacer(1, 5))
        flow.append(Paragraph(
            "Regions are located in the later acquisition and are drawn on it above.",
            st["faint"]))
    return flow


def _validation_section(validation: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    severity = str(validation.get("severity") or "ok").lower()
    issues = validation.get("issues") or []
    colour = SEVERITY_COLOURS.get(severity, INK_SOFT)
    headline = {
        "ok": "PASSED — no internal contradictions found",
        "warn": "PASSED WITH WARNINGS — review the items below",
        "reject": "REJECTED — this output contradicts itself and was not treated as a finished result",
    }.get(severity, severity.upper())

    flow = _section("Self-consistency check", st)
    flow.append(_callout(
        f"<b>{_esc(headline)}</b>",
        st,
        accent=colour,
        background={"ok": colors.HexColor("#eaf6f0"), "warn": colors.HexColor("#fdf4e3")}.get(
            severity, colors.HexColor("#fdeceb")),
    ))
    flow.append(Spacer(1, 8))

    if issues:
        flow.append(_grid_table(
            ["Severity", "Code", "Finding"],
            [[str(i.get("severity") or "-"), str(i.get("code") or "-"),
              " ".join(filter(None, [str(i.get("message") or ""),
                                     f"({i.get('where')})" if i.get("where") else ""]))]
             for i in issues],
            [CONTENT_WIDTH * 0.13, CONTENT_WIDTH * 0.24, CONTENT_WIDTH * 0.63],
            st,
        ))
        flow.append(Spacer(1, 7))

    flow.append(Paragraph(
        "<b>What this check does and does not claim.</b> It tests the specialist's output for "
        "internal contradiction and implausibility — a stated position that disagrees with the "
        "bounding box, a caption whose count disagrees with the objects extracted, labels outside "
        "the trained vocabulary, degenerate or duplicate boxes. Self-contradiction is strong "
        "evidence of a hallucination, so this catches a real class of bad output. It cannot "
        "verify a claim against the actual pixels; nothing downstream of the model can. Read this "
        "as self-consistency and hallucination detection, not as confirmation that the answer is "
        "correct.", st["small"]))
    return flow


def _provenance_section(
    envelope: Dict[str, Any], response: Dict[str, Any], router: str,
    st: Dict[str, ParagraphStyle],
) -> List[Any]:
    model = envelope.get("model") or {}
    task = envelope.get("task")
    flow = _section("Model provenance and measured performance", st)
    flow.append(_kv_table([
        ("Specialist", f"{response.get('worker') or 'n/a'} — {str(task or 'n/a').replace('_', ' ')}"),
        ("Base model", model.get("base") or "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"),
        ("LoRA adapter", model.get("adapter") or "n/a"),
        ("Training steps", str(model.get("trained_steps") or "n/a")),
        ("Inference backend", model.get("backend") or "n/a"),
        ("Inference time", f"{envelope.get('inference_seconds')} s"
         if envelope.get("inference_seconds") is not None else "n/a"),
        ("Router", router),
    ], st))
    flow.append(Spacer(1, 9))

    if task == "bitemporal_change":
        flow.append(Paragraph("Measured on a balanced 120-pair LEVIR-CC validation set", st["h3"]))
        rows = [
            ["Balanced accuracy", "50.0%", "94.2%"],
            ["Recall, changed pairs", "100%", "91.7%"],
            ["Recall, unchanged pairs", "0%", "96.7%"],
            ["Valid schema JSON", "0 / 120", "120 / 120"],
            ["Changed-class F1", "—", "88.6%"],
            ["BLEU-4 (changed pairs only)", "1.9", "24.6"],
            ["CIDEr-D (all pairs, ×100)", "2.5", "123.6"],
        ]
        note = (
            "The base model scores exactly chance because it reports change on every pair, "
            "including unchanged ones — it never says \"no difference\"."
        )
    else:
        flow.append(Paragraph("Measured on a 60-image VRSBench validation set", st["h3"]))
        rows = [
            ["Valid schema JSON", "0%", "100%"],
            ["Object-class F1 (set)", "—", "63.3%"],
            ["Object-class F1 (multiset)", "—", "57.1%"],
            ["Object-count exact match", "—", "55%"],
            ["BLEU-4", "2.0", "12.1"],
            ["METEOR", "26.0", "37.1"],
            ["CIDEr-D (×100)", "0.6", "25.1"],
        ]
        note = (
            "Set F1 asks whether the right kinds of object were named; multiset F1 also requires "
            "the right number of each."
        )

    flow.append(Spacer(1, 4))
    flow.append(_grid_table(
        ["Metric", "Base model", "Fine-tuned"],
        rows,
        [CONTENT_WIDTH * 0.50, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.25],
        st,
    ))
    flow.append(Spacer(1, 5))
    flow.append(Paragraph(_esc(note), st["faint"]))
    flow.append(Spacer(1, 9))
    flow.append(_callout(
        "<b>Known limits of this system.</b> Object counting is unreliable — five ships may be "
        "read as two — so treat any count as indicative rather than measured. Caption wording "
        "often differs from the reference even when the substance is right. The specialists were "
        "fine-tuned on benchmark imagery (VRSBench, LEVIR-CC) and have not been validated on "
        "operational sensor products. What this system supports is structured extraction and "
        "change detection; it is not a source of verified counts.",
        st, accent=WARN, background=colors.HexColor("#fdf4e3"), style="small"))
    return flow


def _trace_section(response: Dict[str, Any], st: Dict[str, ParagraphStyle]) -> List[Any]:
    lines = response.get("execution_trace") or []
    if not lines:
        return []
    heading = _section("Execution trace", st)
    intro = Paragraph(
        "Every step the pipeline took, in order, as recorded by the orchestrator.", st["faint"])
    body = "<br/>".join(_esc(_plain(line)) for line in lines)
    table = Table([[Paragraph(body, st["mono"])]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8fa")),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    # Heading, intro and box travel together; an orphaned heading at a page
    # foot is the most common way a generated report looks unfinished.
    return heading[:1] + [KeepTogether(heading[1:] + [intro, Spacer(1, 5), table])]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_report(
    out_path: Path,
    response: Dict[str, Any],
    image_records: Sequence[Any],
    envelope: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    overlay_path: Optional[Path] = None,
    strip_path: Optional[Path] = None,
    router: str = "heuristic",
) -> Path:
    """
    Write the PDF and return its path.

    `envelope` is the raw generate_report_v2 result; when it is absent (a
    rejected query, or a stubbed worker) the report still renders, covering the
    routing decision and why no analysis was produced. A report is always
    produced - a rejection with a clear reason is a legitimate result.
    """
    st = _styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title=f"SatQueryAI report {response.get('request_id', '')}",
        author="SatQueryAI",
        subject=response.get("query", ""),
    )

    flow: List[Any] = []
    flow += _header(response, st)
    flow += _routing_section(response, st)

    status = response.get("status")
    if status != "ok":
        flow += _section("Outcome", st)
        reason = (
            response.get("error")
            or (response.get("validation") or {}).get("reason")
            or "The request was not executed."
        )
        flow.append(_callout(
            f"<b>No analysis was produced.</b><br/>{_esc(reason)}",
            st, accent=BAD, background=colors.HexColor("#fdeceb")))
        flow.append(Spacer(1, 8))
        flow.append(Paragraph(
            "This is a deliberate refusal, not a failure: the router validated the query "
            "against each specialist's input contract and found no specialist that could "
            "answer it correctly. Acting anyway would have produced a confident wrong answer.",
            st["small"]))

    if image_records:
        flow += _imagery_section(image_records, overlay_path, strip_path, st)

    if status == "ok":
        flow += _findings_section(response, st)

        analysis = (envelope or {}).get("analysis") or {}
        task = (envelope or {}).get("task")
        if task == "single_image":
            flow += _single_detail_section(analysis, st)
        elif task == "bitemporal_change":
            flow += _change_detail_section(analysis, st)

        if validation:
            flow += _validation_section(validation, st)
        if envelope:
            flow += _provenance_section(envelope, response, router, st)

    flow += _trace_section(response, st)
    flow.append(Spacer(1, 12))
    flow.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=5))
    flow.append(Paragraph(
        "Generated by SatQueryAI — an agentic remote-sensing vision-language system. The router "
        "selects a specialist; the specialist analyses the imagery; the self-consistency layer "
        "checks the result before it is reported.", st["faint"]))

    decorate = _make_page_decorator(str(response.get("request_id") or ""))
    doc.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    return out_path
