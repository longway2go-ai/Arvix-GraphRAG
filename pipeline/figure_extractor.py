"""
pipeline/08_figure_extractor.py
─────────────────────────────────
Stage 8: Extract figures and their captions from research paper PDFs.

FREE tools used (zero API cost):
  - PyMuPDF (fitz)  → extracts embedded images from PDF pages + renders pages
    pip install PyMuPDF
  - Pillow           → image saving, minimal filtering
    pip install Pillow

Two-pass strategy:
  Pass 1 → Extract embedded images directly from the PDF binary
            (fastest, highest quality — works for vector + raster figures)
  Pass 2 → Associate each image with its nearest caption text
            (captions identified by "Figure N" / "Fig. N" patterns)

Output stored in:
  data/extracted/<arxiv_id>/figures/fig_<n>_p<page>.png   (image)
  data/extracted/<arxiv_id>/figures/fig_<n>_p<page>.json  (metadata + caption)
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models import ExtractedElement, ElementType
from pipeline.layout_detector import RawElement


# ── Caption detection ──────────────────────────────────────────────────────────

CAPTION_PATTERN = re.compile(
    r"^(figure|fig\.?|fig\s+|table\s+|tab\.?\s*)\s*(\d+[a-z]?)[:\.\s]",
    re.IGNORECASE,
)

def _is_caption(text: str) -> bool:
    return bool(CAPTION_PATTERN.match(text.strip()))

def _parse_caption_number(text: str) -> Optional[str]:
    """Extract 'Figure 3' or 'Fig. 5a' from caption text."""
    m = re.match(
        r"^(figure|fig\.?|fig\s+|table|tab\.?)\s*(\d+[a-z]?)",
        text.strip(),
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}"
    return None


# ── PyMuPDF image extractor ───────────────────────────────────────────────────

MIN_IMAGE_SIZE = 100 * 100    # pixels — ignore tiny icons/logos

def _extract_images_from_pdf(
    pdf_path: str,
    figures_dir: Path,
    min_size: int = MIN_IMAGE_SIZE,
) -> list[dict]:
    """
    Extract all embedded images from a PDF using PyMuPDF.
    Returns a list of dicts: {page_number, image_path, width, height, xref}.

    PyMuPDF iterates the PDF's cross-reference table to find image objects
    directly — this is much more reliable than rendering each page and doing
    image segmentation.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    images_found = []

    doc = fitz.open(pdf_path)
    fig_counter = 0

    for page_idx, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]   # PDF cross-reference number for the image

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w * h < min_size:
                continue   # skip tiny decorative images

            ext       = base_image.get("ext", "png")
            img_bytes = base_image.get("image", b"")
            if not img_bytes:
                continue

            # Save image
            fig_counter += 1
            img_filename = f"fig_{fig_counter:03d}_p{page_idx}.{ext}"
            img_path = figures_dir / img_filename
            img_path.write_bytes(img_bytes)

            images_found.append({
                "page_number": page_idx,
                "image_path":  str(img_path),
                "width":       w,
                "height":      h,
                "xref":        xref,
                "fig_counter": fig_counter,
            })

    doc.close()
    logger.debug(f"Extracted {len(images_found)} embedded images from {Path(pdf_path).name}")
    return images_found


# ── Caption ↔ figure matching ─────────────────────────────────────────────────

def _match_captions_to_figures(
    images: list[dict],
    caption_elements: list[RawElement],
) -> dict[int, str]:
    """
    Match each figure (by page) to its nearest caption text.

    Strategy: for each figure, look for a caption element on the same page
    or the adjacent page. Prefer captions with explicit "Figure N" labels.

    Returns: dict mapping fig_counter → caption_text
    """
    matches: dict[int, str] = {}

    for img in images:
        page = img["page_number"]
        fig_no = img["fig_counter"]
        best = None

        # Look for a caption on the same page or ±1 page
        for cap in caption_elements:
            if abs(cap.page_number - page) <= 1 and _is_caption(cap.text):
                # Prefer explicit match, otherwise take the closest one
                if best is None:
                    best = cap.text
                else:
                    # If the caption has "Figure N" and matches our counter, prefer it
                    cap_num = _parse_caption_number(cap.text)
                    best_num = _parse_caption_number(best) if best else None
                    if cap_num and str(fig_no) in cap_num:
                        best = cap.text

        if best:
            matches[fig_no] = best

    return matches


# ── Main extractor function ────────────────────────────────────────────────────

def extract_figure_elements(
    raw_elements: list[RawElement],
    paper_id: str,
    pdf_path: str,
    save_dir: Path | None = None,
    **kwargs,
) -> list[ExtractedElement]:
    """
    Extract figures (images) and captions from a PDF.

    This extractor receives both Image and FigureCaption RawElements.
    It:
      1. Extracts all embedded images via PyMuPDF
      2. Associates each image with its nearest caption text
      3. Returns one ExtractedElement per figure (content = caption text)

    Args:
        raw_elements: Image and Caption RawElements from the router
        paper_id:     arXiv ID
        pdf_path:     Path to the PDF (for PyMuPDF image extraction)
        save_dir:     Directory to save figure images + JSON

    Returns:
        List of ExtractedElement with element_type="figure"
        Each has metadata["caption"], metadata["image_path"], metadata["fig_number"]
    """
    extracted: list[ExtractedElement] = []

    # Set up output directories
    figures_dir = Path(save_dir) / "figures" if save_dir else Path(f"/tmp/{paper_id}_figs")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Separate caption elements from image elements
    caption_elements = [e for e in raw_elements
                        if e.raw_type in ("FigureCaption", "Caption") or _is_caption(e.text)]
    image_elements   = [e for e in raw_elements if e.raw_type == "Image"]

    # ── Pass 1: extract embedded images from PDF ──────────────────────────────
    images = _extract_images_from_pdf(pdf_path, figures_dir)

    if not images:
        # If PyMuPDF didn't find embedded images, fall back to unstructured Image elements
        logger.debug(f"[{paper_id}] No embedded images found via PyMuPDF — using unstructured Image elements")
        images = [
            {
                "page_number": el.page_number,
                "image_path":  None,
                "width":       0,
                "height":      0,
                "xref":        None,
                "fig_counter": i + 1,
            }
            for i, el in enumerate(image_elements)
        ]

    # ── Pass 2: match captions to figures ─────────────────────────────────────
    # Also collect captions from raw text that follows Figure N pattern
    all_captions = caption_elements[:]
    for raw in raw_elements:
        if _is_caption(raw.text) and raw not in caption_elements:
            all_captions.append(raw)

    caption_map = _match_captions_to_figures(images, all_captions)

    # ── Build ExtractedElement per figure ─────────────────────────────────────
    for img in images:
        fig_no   = img["fig_counter"]
        caption  = caption_map.get(fig_no, "")
        fig_label = _parse_caption_number(caption) if caption else f"Figure {fig_no}"

        # The "content" of a figure node is its caption (used for RAG retrieval)
        content = caption if caption else f"[Figure {fig_no} on page {img['page_number']}]"

        el = ExtractedElement(
            element_type = ElementType.FIGURE,
            content      = content,
            paper_id     = paper_id,
            page_number  = img["page_number"],
            metadata     = {
                "caption":    caption,
                "fig_number": fig_label,
                "image_path": img.get("image_path"),
                "width":      img.get("width", 0),
                "height":     img.get("height", 0),
                "fig_index":  fig_no,
            },
        )
        extracted.append(el)

        # Save JSON sidecar
        if save_dir:
            json_path = figures_dir / f"fig_{fig_no:03d}_p{img['page_number']}.json"
            json_path.write_text(
                json.dumps(el.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Also emit standalone caption elements ─────────────────────────────────
    # Captions that weren't matched to an image (e.g. figure on next page)
    matched_captions = set(caption_map.values())
    for cap in all_captions:
        if cap.text in matched_captions:
            continue   # already embedded in a figure element
        extracted.append(ExtractedElement(
            element_type = ElementType.CAPTION,
            content      = cap.text,
            paper_id     = paper_id,
            page_number  = cap.page_number,
            metadata     = {
                "fig_number": _parse_caption_number(cap.text),
                "unmatched":  True,
            },
        ))

    logger.debug(
        f"[{paper_id}] Figure extractor: {len(images)} images, "
        f"{len(all_captions)} captions, {len(extracted)} elements total"
    )
    return extracted