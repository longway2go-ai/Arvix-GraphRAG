"""
pipeline/07_equation_extractor.py
───────────────────────────────────
Stage 7: Extract mathematical equations and convert them to LaTeX strings.

FREE tools used (zero API cost):
  - pix2tex (LaTeX-OCR)   → converts equation images to LaTeX
    pip install pix2tex[api]
    Model weights: ~80 MB, downloaded once from HuggingFace on first run.
    Runs entirely on CPU (or GPU if available) — no API call.

  - PyMuPDF (fitz)        → crops the equation bounding box as a PNG image
    pip install PyMuPDF

Pipeline per equation:
  1. Receive a Formula RawElement with bbox from the layout detector
  2. Use PyMuPDF to render that page at high DPI and crop the bbox region
  3. Pass the cropped image to pix2tex → get a LaTeX string
  4. Fall back to raw text if pix2tex fails (still useful for the graph)

Output stored in:
  data/extracted/<arxiv_id>/equations/eq_<n>.png  (cropped image)
  data/extracted/<arxiv_id>/equations/eq_<n>.json (metadata + LaTeX)
"""

from __future__ import annotations

import re
import json
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models import ExtractedElement, ElementType
from pipeline.layout_detector import RawElement


# ── pix2tex loader (lazy singleton) ──────────────────────────────────────────

_latex_ocr = None

def _get_latex_ocr():
    """
    Load pix2tex LatexOCR model once and cache it.
    First run downloads ~80 MB of model weights from HuggingFace.
    """
    global _latex_ocr
    if _latex_ocr is not None:
        return _latex_ocr

    try:
        from pix2tex.cli import LatexOCR
        logger.info("Loading pix2tex LatexOCR model (first run downloads ~80 MB)…")
        _latex_ocr = LatexOCR()
        logger.success("pix2tex model loaded")
    except ImportError:
        logger.warning(
            "pix2tex not installed. Run: pip install pix2tex[api]\n"
            "Equations will be stored as plain text only."
        )
        _latex_ocr = None
    except Exception as e:
        logger.warning(f"pix2tex failed to load: {e}. Using plain text fallback.")
        _latex_ocr = None

    return _latex_ocr


# ── Image cropper ─────────────────────────────────────────────────────────────

def _crop_equation_image(
    pdf_path: str,
    page_number: int,
    bbox: list[float],
    dpi: int = 200,
) -> Optional["PIL.Image.Image"]:
    """
    Render a PDF page at `dpi` and crop the equation bounding box.
    Returns a PIL Image, or None if cropping fails.

    Args:
        pdf_path:    Path to the PDF
        page_number: 1-indexed page number
        bbox:        [x0, y0, x1, y1] in PDF coordinate space (points)
        dpi:         Render resolution. 200 DPI is enough for pix2tex.
    """
    try:
        import fitz      # PyMuPDF
        from PIL import Image
        import io

        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]   # fitz is 0-indexed

        # Scale factor: PDF points → pixels at target DPI
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        # Full-page render as PNG
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Translate bbox from PDF points to pixel coordinates
        w_pts, h_pts = page.rect.width, page.rect.height
        w_px, h_px   = img.width, img.height

        x0 = int(bbox[0] / w_pts * w_px)
        y0 = int(bbox[1] / h_pts * h_px)
        x1 = int(bbox[2] / w_pts * w_px)
        y1 = int(bbox[3] / h_pts * h_px)

        # Add 10px padding to avoid clipping equation edges
        PAD = 10
        x0 = max(0, x0 - PAD)
        y0 = max(0, y0 - PAD)
        x1 = min(w_px, x1 + PAD)
        y1 = min(h_px, y1 + PAD)

        if x1 <= x0 or y1 <= y0:
            return None

        doc.close()
        return img.crop((x0, y0, x1, y1))

    except Exception as e:
        logger.debug(f"Image crop failed (page={page_number}, bbox={bbox}): {e}")
        return None


# ── LaTeX from text fallback ──────────────────────────────────────────────────

# Simple heuristic: if the raw text already looks like LaTeX, keep it.
LATEX_HINT = re.compile(r"\\[a-zA-Z]+\{|_\{|\^\{|\frac|\\sum|\\int|\\partial")

def _text_looks_like_latex(text: str) -> bool:
    return bool(LATEX_HINT.search(text))


def _clean_raw_equation_text(text: str) -> str:
    """
    Minimal clean-up for raw text that is not passed through pix2tex.
    Removes stray whitespace, normalises newlines.
    """
    return re.sub(r"\s+", " ", text).strip()


# ── Main extractor function ────────────────────────────────────────────────────

def extract_equation_elements(
    raw_elements: list[RawElement],
    paper_id: str,
    pdf_path: str,
    save_dir: Path | None = None,
    **kwargs,
) -> list[ExtractedElement]:
    """
    Convert Formula RawElements into ExtractedElement objects with LaTeX content.

    For each equation:
      - If bbox is available: crop the image → run pix2tex → get LaTeX
      - If no bbox or pix2tex fails: store the raw text as-is

    Args:
        raw_elements: Formula RawElements from the router
        paper_id:     arXiv ID
        pdf_path:     Path to the PDF (for image cropping)
        save_dir:     Directory to save equation images + JSON

    Returns:
        List of ExtractedElement with element_type="equation"
        Each has metadata["latex"] (the LaTeX string) and metadata["source"]
        ("pix2tex" | "raw_text" | "latex_hint")
    """
    extracted: list[ExtractedElement] = []
    ocr = _get_latex_ocr()

    eq_dir = None
    if save_dir:
        eq_dir = Path(save_dir) / "equations"
        eq_dir.mkdir(parents=True, exist_ok=True)

    for i, raw in enumerate(raw_elements, start=1):
        raw_text = raw.text.strip()
        latex    = None
        source   = "raw_text"

        # ── Try pix2tex if we have a bbox ──────────────────────────────────────
        if ocr and raw.bbox and raw.page_number:
            img = _crop_equation_image(pdf_path, raw.page_number, raw.bbox)
            if img:
                # Save cropped image for debugging / graph storage
                if eq_dir:
                    img_path = eq_dir / f"eq_{i:03d}_p{raw.page_number}.png"
                    img.save(str(img_path))

                try:
                    latex  = ocr(img)
                    source = "pix2tex"
                    logger.debug(f"  eq {i}: pix2tex → {latex[:60]}")
                except Exception as e:
                    logger.debug(f"  eq {i}: pix2tex failed ({e}), using raw text")

        # ── Fallbacks ──────────────────────────────────────────────────────────
        if latex is None:
            if _text_looks_like_latex(raw_text):
                latex  = raw_text
                source = "latex_hint"
            else:
                latex  = _clean_raw_equation_text(raw_text)
                source = "raw_text"

        if not latex:
            continue

        meta = {
            "latex":       latex,
            "source":      source,
            "raw_text":    raw_text,
        }

        # Save JSON sidecar
        el = ExtractedElement(
            element_type = ElementType.EQUATION,
            content      = latex,          # LaTeX is the primary content
            paper_id     = paper_id,
            page_number  = raw.page_number,
            bbox         = raw.bbox,
            metadata     = meta,
        )

        if eq_dir:
            json_path = eq_dir / f"eq_{i:03d}_p{raw.page_number}.json"
            json_path.write_text(
                json.dumps(el.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        extracted.append(el)

    pix2tex_count = sum(1 for e in extracted if e.metadata.get("source") == "pix2tex")
    logger.debug(
        f"[{paper_id}] Equation extractor: {len(extracted)} equations "
        f"({pix2tex_count} via pix2tex, {len(extracted)-pix2tex_count} plain text)"
    )
    return extracted