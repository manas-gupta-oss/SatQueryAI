"""
Draw the specialists' visual evidence onto the uploaded imagery.

Every box drawn here comes from `bbox_normalized`, never the raw `obj_coord` /
`bbox` field: models/MODELS.md is explicit that the raw coordinates can fall
outside the frame while the normalised ones are clamped to 0-1. Drawing the raw
values is how boxes end up hanging off the edge of the picture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

#: Distinct, colour-blind-safe outline colours, cycled per label.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 158, 218),    # blue
    (255, 143, 0),    # amber
    (0, 176, 118),    # green
    (222, 60, 90),    # red
    (150, 108, 232),  # violet
    (0, 190, 190),    # teal
)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2 has no size argument
        return ImageFont.load_default()


def _colour_for(label: str, seen: List[str]) -> Tuple[int, int, int]:
    if label not in seen:
        seen.append(label)
    return PALETTE[seen.index(label) % len(PALETTE)]


def draw_boxes(
    image_path: Path,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    out_path: Path,
    max_width: int = 1400,
) -> Optional[Path]:
    """
    Write an annotated copy of `image_path` to `out_path`.

    Returns the path written, or None if there was nothing to draw or the source
    could not be opened - an annotation failure must never fail a query.
    """
    if not boxes:
        return None
    try:
        with Image.open(image_path) as src:
            img = src.convert("RGB")
    except Exception as exc:
        logger.warning("could not open %s for annotation: %s", image_path, exc)
        return None

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * ratio))), Image.LANCZOS)

    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size
    line_width = max(2, round(min(width, height) / 320))
    font = _font(max(13, round(min(width, height) / 42)))
    seen: List[str] = []

    for index, box in enumerate(boxes):
        if len(box) != 4:
            continue
        label = str(labels[index]) if index < len(labels) else "region"
        colour = _colour_for(label, seen)
        x0, y0, x1, y1 = (
            box[0] * width, box[1] * height, box[2] * width, box[3] * height,
        )
        if x1 - x0 < 1 or y1 - y0 < 1:
            # A degenerate box is flagged by validate.py; draw a marker so the
            # reviewer can see where the model put it rather than silently
            # dropping it.
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(4, line_width * 3)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=line_width)
            continue
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=line_width)

        text = f"{index + 1}. {label}"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
        pad = max(3, line_width)
        # Keep the caption inside the frame when the box hugs the top edge.
        ly = y0 - th - 2 * pad
        if ly < 0:
            ly = min(y0 + pad, height - th - 2 * pad)
        lx = min(max(0, x0), max(0, width - tw - 2 * pad))
        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=colour + (235,))
        draw.text((lx + pad - tx0, ly + pad - ty0), text, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def side_by_side(
    before_path: Path,
    after_path: Path,
    out_path: Path,
    gap: int = 16,
    panel_width: int = 720,
) -> Optional[Path]:
    """A labelled before/after strip, used as the change report's key figure."""
    try:
        with Image.open(before_path) as b, Image.open(after_path) as a:
            before, after = b.convert("RGB"), a.convert("RGB")
    except Exception as exc:
        logger.warning("could not build a before/after strip: %s", exc)
        return None

    def _fit(img: Image.Image) -> Image.Image:
        ratio = panel_width / img.width
        return img.resize((panel_width, max(1, int(img.height * ratio))), Image.LANCZOS)

    before, after = _fit(before), _fit(after)
    band = max(28, panel_width // 22)
    height = max(before.height, after.height) + band
    canvas = Image.new("RGB", (panel_width * 2 + gap, height), (17, 20, 26))
    canvas.paste(before, (0, band))
    canvas.paste(after, (panel_width + gap, band))

    draw = ImageDraw.Draw(canvas)
    font = _font(max(13, band // 2))
    draw.text((6, band // 4), "BEFORE (earlier acquisition)", fill=(215, 220, 230), font=font)
    draw.text((panel_width + gap + 6, band // 4), "AFTER (later acquisition)",
              fill=(215, 220, 230), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path
