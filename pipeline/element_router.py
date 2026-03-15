"""
pipeline/04_element_router.py
───────────────────────────────
Stage 4: Classify each RawElement from the layout detector and dispatch
it to the correct specialized extractor (stages 05-08).

This is a pure dispatch layer — it contains zero extraction logic.
Think of it as a traffic controller: it reads the element type and calls
the right extractor function, then collects all results into one
PaperExtractionResult object.

FREE tools: no external libraries needed — pure Python dispatch logic.

Usage (as module):
    from pipeline.element_router import route_elements
    result = route_elements(paper_id, pdf_path, raw_elements)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models import ExtractedElement, ElementType, PaperExtractionResult
from pipeline.layout_detector import RawElement, UNSTRUCTURED_TYPE_MAP

console = Console()


# ── Type normalisation ─────────────────────────────────────────────────────────

def normalise_type(raw_type: str) -> str | None:
    """
    Map an unstructured raw_type string to our 5-category system.
    Returns None for element types we intentionally skip (footers, page numbers).
    """
    return UNSTRUCTURED_TYPE_MAP.get(raw_type, "text")   # default: treat as text


# ── Router ────────────────────────────────────────────────────────────────────

def route_elements(
    paper_id: str,
    pdf_path:  str | Path,
    raw_elements: list[RawElement],
    save_dir: Path | None = None,
) -> PaperExtractionResult:
    """
    Dispatch every RawElement to its specialized extractor and collect
    results into a PaperExtractionResult.

    Args:
        paper_id:     arXiv ID (e.g. "1706.03762")
        pdf_path:     Path to the PDF (needed by table/equation/figure extractors)
        raw_elements: Output of detect_layout()
        save_dir:     Where to save extraction.json (default: settings.extracted_dir/<paper_id>)

    Returns:
        PaperExtractionResult with all ExtractedElement objects

    Processing order:
        1. Group raw elements by normalised type
        2. Call each extractor with its group
        3. Merge all results
        4. Save to disk
    """
    from config.settings import settings

    pdf_path = Path(pdf_path)
    save_dir = save_dir or (settings.extracted_dir / paper_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = PaperExtractionResult(paper_id=paper_id, pdf_path=str(pdf_path))

    # ── Lazy imports of extractors ────────────────────────────────────────────
    # Each is imported here so stages can be developed independently.
    # If one extractor fails to import, only that type is skipped.
    extractor_registry: dict[str, Callable] = {}

    try:
        from pipeline.text_extractor import extract_text_elements
        extractor_registry[ElementType.TEXT]  = extract_text_elements
        extractor_registry[ElementType.TITLE] = extract_text_elements
    except ImportError as e:
        logger.warning(f"Text extractor unavailable: {e}")

    try:
        from pipeline.table_extractor import extract_table_elements
        extractor_registry[ElementType.TABLE] = extract_table_elements
    except ImportError as e:
        logger.warning(f"Table extractor unavailable: {e}")

    try:
        from pipeline.equation_extractor import extract_equation_elements
        extractor_registry[ElementType.EQUATION] = extract_equation_elements
    except ImportError as e:
        logger.warning(f"Equation extractor unavailable: {e}")

    try:
        from pipeline.figure_extractor import extract_figure_elements
        extractor_registry[ElementType.FIGURE]  = extract_figure_elements
        extractor_registry[ElementType.CAPTION] = extract_figure_elements
    except ImportError as e:
        logger.warning(f"Figure extractor unavailable: {e}")

    # ── Group raw elements by normalised type ─────────────────────────────────
    groups: dict[str, list[RawElement]] = {}
    skipped = 0

    for raw in raw_elements:
        norm = normalise_type(raw.raw_type)
        if norm is None:
            skipped += 1
            continue
        groups.setdefault(norm, []).append(raw)

    logger.info(
        f"[{paper_id}] Routing {len(raw_elements)} elements: "
        + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
        + (f", {skipped} skipped" if skipped else "")
    )

    # ── Dispatch each group to its extractor ──────────────────────────────────
    for element_type, group in groups.items():

        extractor = extractor_registry.get(element_type)
        if extractor is None:
            logger.debug(f"No extractor registered for type '{element_type}' — skipping")
            continue

        logger.debug(f"  → {element_type}: {len(group)} elements")

        try:
            extracted: list[ExtractedElement] = extractor(
                raw_elements = group,
                paper_id     = paper_id,
                pdf_path     = str(pdf_path),
                save_dir     = save_dir,
            )
            for el in extracted:
                result.add(el)

        except Exception as e:
            msg = f"Extractor failed for type '{element_type}': {e}"
            logger.error(msg)
            result.errors.append(msg)

    # ── Save result to disk ───────────────────────────────────────────────────
    out_path = save_dir / "extraction.json"
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success(f"[{paper_id}] {result.summary()} → {out_path}")

    return result


def route_paper(
    paper_id: str,
    pdf_path: str | Path,
    prefer_ml: bool = True,
    save_dir: Path | None = None,
) -> PaperExtractionResult:
    """
    Convenience wrapper: detect layout then route in one call.

    Args:
        paper_id:  arXiv ID
        pdf_path:  Path to PDF
        prefer_ml: Use ML layout detection (recommended)

    Returns:
        Complete PaperExtractionResult
    """
    from pipeline.layout_detector import detect_layout

    raw_elements = detect_layout(pdf_path, prefer_ml=prefer_ml)
    return route_elements(paper_id, pdf_path, raw_elements, save_dir=save_dir)


def route_paper_batch(
    papers: list[dict],      # list of {"paper_id": ..., "pdf_path": ...}
    prefer_ml: bool = True,
) -> list[PaperExtractionResult]:
    """
    Route multiple papers with a progress bar.

    Args:
        papers: List of dicts with keys: paper_id, pdf_path (local_pdf_path also accepted)
        prefer_ml: Use ML layout detection

    Returns:
        List of PaperExtractionResult, one per paper
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting papers…", total=len(papers))

        for p in papers:
            paper_id = p.get("paper_id") or p.get("arxiv_id", "unknown")
            pdf_path = p.get("pdf_path") or p.get("local_pdf_path", "")

            progress.update(task, description=f"Extracting [{paper_id}]…")

            if not pdf_path or not Path(pdf_path).exists():
                logger.warning(f"PDF not found for {paper_id}: {pdf_path}")
                progress.advance(task)
                continue

            try:
                result = route_paper(paper_id, pdf_path, prefer_ml=prefer_ml)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {paper_id}: {e}")

            progress.advance(task)

    # Print summary
    total_elements = sum(len(r.elements) for r in results)
    console.print(
        f"\n[bold green]Extraction complete:[/bold green] "
        f"{len(results)} papers, {total_elements} total elements"
    )
    return results