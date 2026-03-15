"""
pipeline/03_layout_detector.py
────────────────────────────────
Stage 3: Run layout detection on a downloaded PDF and return a list of
raw typed regions. This stage does NOT extract content — it only labels
what kind of thing each region is.

FREE tools used (zero API cost):
  Primary  → unstructured[local-inference]  (detectron2 + DocLayNet)
  Fallback → pdfplumber + regex heuristics  (if unstructured/detectron2 unavailable)

Usage (standalone):
    python pipeline/03_layout_detector.py --pdf data/pdfs/1706.03762.pdf

Usage (as module):
    from pipeline.layout_detector import detect_layout
    raw_elements = detect_layout("data/pdfs/1706.03762.pdf")
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

console = Console()

# ── Raw layout element (output of this stage, input to the router) ─────────────

@dataclass
class RawElement:
    """
    A single labelled region from the layout detector.
    This is NOT yet extracted content — it's a bounding box + type label.
    The router (stage 4) will decide which extractor handles it.
    """
    raw_type:    str              # unstructured type: "Title", "NarrativeText", etc.
    text:        str              # raw text content (may be partial for images/equations)
    page_number: int
    bbox:        Optional[list[float]] = None   # [x0, y0, x1, y1] in PDF points
    metadata:    dict = field(default_factory=dict)

    # Normalised type assigned after routing
    element_type: Optional[str] = None  # "text" | "table" | "equation" | "figure"


# ── Unstructured element-type → our category mapping ─────────────────────────

# unstructured returns these type strings from partition_pdf()
UNSTRUCTURED_TYPE_MAP = {
    # → text
    "Title":          "title",
    "NarrativeText":  "text",
    "ListItem":       "text",
    "Header":         "title",
    "Footer":         None,      # skip footers
    "PageBreak":      None,
    "PageNumber":     None,
    "UncategorizedText": "text",

    # → table
    "Table":          "table",

    # → equation  (unstructured calls these "Formula")
    "Formula":        "equation",

    # → figure
    "Image":          "figure",
    "FigureCaption":  "caption",
    "Caption":        "caption",
}


# ── Primary: unstructured partition_pdf ───────────────────────────────────────

def _detect_with_unstructured(pdf_path: str) -> list[RawElement]:
    """
    Use unstructured's partition_pdf() for high-quality layout detection.
    Requires: pip install unstructured[local-inference]

    unstructured will:
     1. Convert each PDF page to an image
     2. Run a detectron2 model trained on DocLayNet to classify regions
     3. Return typed Element objects with bounding boxes

    On first run it downloads model weights (~300 MB) from HuggingFace.
    Subsequent runs use the cached model.
    """
    from unstructured.partition.pdf import partition_pdf

    logger.info(f"Running unstructured layout detection on: {Path(pdf_path).name}")

    # strategy="hi_res" uses the ML model; strategy="fast" uses pdfminer (no ML)
    # We try hi_res first, fall back to fast if detectron2 isn't available
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",                    # uses detectron2 DocLayNet
            infer_table_structure=True,           # extract table HTML too
            include_page_breaks=False,
            extract_images_in_pdf=False,          # we handle images in stage 08
        )
        logger.debug("Using strategy=hi_res (detectron2)")
    except Exception as e:
        logger.warning(f"hi_res failed ({e}), falling back to strategy=fast")
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",                      # pdfminer only, no ML needed
            infer_table_structure=True,
        )

    raw = []
    for el in elements:
        # Get the unstructured type name
        type_name = type(el).__name__  # e.g. "NarrativeText", "Table", "Title"

        # Pull coordinates if available
        bbox = None
        if hasattr(el, "metadata") and hasattr(el.metadata, "coordinates"):
            coords = el.metadata.coordinates
            if coords and coords.points:
                pts = coords.points
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

        page_num = 0
        if hasattr(el, "metadata") and hasattr(el.metadata, "page_number"):
            page_num = el.metadata.page_number or 0

        # For tables, unstructured can give us the HTML representation
        meta = {}
        if type_name == "Table" and hasattr(el.metadata, "text_as_html"):
            meta["html"] = el.metadata.text_as_html

        raw.append(RawElement(
            raw_type    = type_name,
            text        = str(el).strip(),
            page_number = page_num,
            bbox        = bbox,
            metadata    = meta,
        ))

    logger.success(f"Detected {len(raw)} elements via unstructured")
    return raw


# ── Fallback: pdfplumber + heuristics ────────────────────────────────────────

def _detect_with_pdfplumber(pdf_path: str) -> list[RawElement]:
    """
    Pure-Python fallback when unstructured / detectron2 is unavailable.
    Uses pdfplumber for text/table extraction plus simple heuristics
    to classify each region.

    Less accurate than the ML approach but requires no model downloads.
    """
    import pdfplumber

    logger.warning("Using pdfplumber fallback (unstructured not available)")

    # Heuristic patterns for section headers in research papers
    SECTION_PATTERN = re.compile(
        r"^\s*(\d+\.?\s+)?(abstract|introduction|related\s+work|methodology|"
        r"method|approach|experiments?|results?|evaluation|discussion|"
        r"conclusion|references?|appendix)\b",
        re.IGNORECASE,
    )
    CAPTION_PATTERN = re.compile(
        r"^\s*(figure|fig\.?|table|tab\.?)\s*\d+",
        re.IGNORECASE,
    )
    EQUATION_HINT = re.compile(
        r"[=∑∫∂∇⊕⊗∈∉∀∃←→⟨⟩αβγδεζηθλμνξπρσφψω]|"
        r"\\[a-zA-Z]+\{|_\{|\^\{",
    )

    raw = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):

            # ── Tables ────────────────────────────────────────────────────────
            for tbl in page.extract_tables() or []:
                if tbl:
                    flat = " | ".join(
                        str(cell) for row in tbl for cell in row if cell
                    )
                    raw.append(RawElement(
                        raw_type    = "Table",
                        text        = flat,
                        page_number = page_idx,
                        element_type = "table",
                        metadata     = {"rows_raw": tbl},
                    ))

            # ── Text blocks ───────────────────────────────────────────────────
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line or len(line) < 3:
                    continue

                # Classify by heuristics
                if CAPTION_PATTERN.match(line):
                    raw_type = "FigureCaption"
                elif SECTION_PATTERN.match(line) and len(line) < 80:
                    raw_type = "Title"
                elif EQUATION_HINT.search(line) and len(line) < 200:
                    raw_type = "Formula"
                else:
                    raw_type = "NarrativeText"

                raw.append(RawElement(
                    raw_type    = raw_type,
                    text        = line,
                    page_number = page_idx,
                ))

    logger.info(f"pdfplumber detected {len(raw)} elements")
    return raw


# ── Unified entry point ───────────────────────────────────────────────────────

def detect_layout(
    pdf_path: str | Path,
    prefer_ml: bool = True,
) -> list[RawElement]:
    """
    Detect document layout for a single PDF.
    Tries unstructured (ML-based) first; falls back to pdfplumber if unavailable.

    Args:
        pdf_path:  Path to the downloaded PDF file
        prefer_ml: Use unstructured/detectron2 when available (recommended)

    Returns:
        List of RawElement objects, one per detected region.
        Each has a raw_type ("NarrativeText", "Table", "Formula", "Image", etc.)
        that stage 4 (element_router) will map to our categories.

    Example:
        elements = detect_layout("data/pdfs/1706.03762.pdf")
        print(f"Found {len(elements)} regions")
        for e in elements[:5]:
            print(f"  page {e.page_number}: {e.raw_type} — {e.text[:50]}")
    """
    pdf_path = str(pdf_path)

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if prefer_ml:
        try:
            return _detect_with_unstructured(pdf_path)
        except ImportError:
            logger.warning("unstructured not installed — using pdfplumber fallback")
        except Exception as e:
            logger.warning(f"unstructured failed: {e} — using pdfplumber fallback")

    return _detect_with_pdfplumber(pdf_path)


def detect_layout_batch(
    pdf_paths: list[str | Path],
    prefer_ml: bool = True,
) -> dict[str, list[RawElement]]:
    """
    Detect layout for multiple PDFs.

    Returns:
        Dict mapping pdf_path → list of RawElement
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting layouts…", total=len(pdf_paths))
        for path in pdf_paths:
            path = str(path)
            try:
                results[path] = detect_layout(path, prefer_ml=prefer_ml)
            except Exception as e:
                logger.error(f"Layout detection failed for {path}: {e}")
                results[path] = []
            progress.advance(task)

    return results


# ── Display helper ─────────────────────────────────────────────────────────────

def display_elements(elements: list[RawElement], max_rows: int = 20) -> None:
    """Print a Rich table summarising detected elements."""
    table = Table(title=f"Detected Elements ({len(elements)} total)", show_lines=True)
    table.add_column("Page", style="dim", width=5, justify="right")
    table.add_column("Raw type", style="cyan", width=18)
    table.add_column("Content preview", style="white", max_width=55)

    for el in elements[:max_rows]:
        table.add_row(
            str(el.page_number),
            el.raw_type,
            el.text[:80].replace("\n", " "),
        )
    if len(elements) > max_rows:
        table.add_row("…", "…", f"({len(elements) - max_rows} more)")

    console.print(table)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run layout detection on a PDF and show detected elements."
    )
    parser.add_argument("--pdf", "-p", required=True, help="Path to the PDF file")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML model, use pdfplumber")
    parser.add_argument("--max-rows", type=int, default=30)
    return parser.parse_args()


def main():
    args = _parse_args()
    elements = detect_layout(args.pdf, prefer_ml=not args.no_ml)
    display_elements(elements, max_rows=args.max_rows)

    # Summary by type
    from collections import Counter
    counts = Counter(e.raw_type for e in elements)
    console.print("\n[bold]Type breakdown:[/bold]")
    for t, n in counts.most_common():
        console.print(f"  {t:<20} {n}")


if __name__ == "__main__":
    main()