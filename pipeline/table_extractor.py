"""
pipeline/06_table_extractor.py
───────────────────────────────
Stage 6: Extract structured table data from research paper PDFs.

FREE tools used (zero API cost):
  Primary  → pdfplumber   (pure Python, great for arXiv papers)
             pip install pdfplumber
  Fallback → camelot-py   (better for lattice/bordered tables)
             pip install camelot-py[cv]

Strategy:
  1. First try to use the table HTML already extracted by unstructured
     (fastest — no extra parsing needed)
  2. If unstructured didn't get it, fall back to pdfplumber page-level extraction
  3. Convert raw rows/cols to a clean JSON structure for the graph
  4. Auto-detect if a table is a benchmark results table
     (contains model names + metric columns)

Output stored in:
  data/extracted/<arxiv_id>/tables/table_<n>.json
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


# ── HTML table parser (from unstructured output) ──────────────────────────────

def _parse_html_table(html: str) -> tuple[list[str], list[list[str]]]:
    """
    Parse an HTML table string from unstructured into headers + rows.
    Uses Python's built-in html.parser — no extra dependencies.
    Returns: (headers, rows) where each row is a list of cell strings.
    """
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._row: list[str] = []
            self._cell: str = ""
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag in ("tr",):
                self._row = []
            elif tag in ("td", "th"):
                self._cell = ""
                self._in_cell = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._row.append(self._cell.strip())
                self._in_cell = False
            elif tag == "tr":
                if self._row:
                    self.rows.append(self._row)

        def handle_data(self, data):
            if self._in_cell:
                self._cell += data

    parser = _TableParser()
    parser.feed(html)

    if not parser.rows:
        return [], []

    # Heuristic: first row is headers if all cells are short and non-numeric
    first_row = parser.rows[0]
    is_header = all(len(c) < 40 and not re.match(r"^\d+\.?\d*$", c) for c in first_row if c)
    if is_header and len(parser.rows) > 1:
        return first_row, parser.rows[1:]
    else:
        return [], parser.rows


# ── pdfplumber table extractor ────────────────────────────────────────────────

def _extract_with_pdfplumber(
    pdf_path: str,
    page_numbers: Optional[list[int]] = None,
) -> list[tuple[int, list[str], list[list[str]]]]:
    """
    Use pdfplumber to extract tables from specific pages (or all pages).
    Returns list of (page_number, headers, rows).
    """
    import pdfplumber

    results = []

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_check = (
            [pdf.pages[i - 1] for i in page_numbers if 0 < i <= len(pdf.pages)]
            if page_numbers else pdf.pages
        )

        for page in pages_to_check:
            try:
                tables = page.extract_tables()
                if not tables:
                    continue

                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue

                    # Clean cells: replace None with ""
                    clean = [
                        [str(cell).strip() if cell else "" for cell in row]
                        for row in tbl
                    ]

                    # Heuristic: first row = header if it looks like labels
                    first = clean[0]
                    is_header = all(
                        len(c) < 50 and not re.match(r"^\d+\.?\d*$", c)
                        for c in first if c
                    )
                    headers = first if is_header else []
                    rows    = clean[1:] if is_header else clean

                    results.append((page.page_number, headers, rows))

            except Exception as e:
                logger.debug(f"pdfplumber failed on page {page.page_number}: {e}")

    return results


# ── camelot fallback ───────────────────────────────────────────────────────────

def _extract_with_camelot(
    pdf_path: str,
    pages: str = "all",
) -> list[tuple[int, list[str], list[list[str]]]]:
    """
    Use camelot-py for lattice-bordered tables (works well for formatted papers).
    """
    try:
        import camelot
    except ImportError:
        logger.debug("camelot not installed, skipping")
        return []

    results = []
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        for tbl in tables:
            df = tbl.df
            if df.empty or len(df) < 2:
                continue
            headers = df.iloc[0].tolist()
            rows    = df.iloc[1:].values.tolist()
            results.append((tbl.page, headers, rows))
    except Exception as e:
        logger.debug(f"camelot extraction failed: {e}")

    return results


# ── Benchmark table detector ───────────────────────────────────────────────────

METRIC_WORDS = {
    "accuracy", "f1", "bleu", "rouge", "precision", "recall",
    "perplexity", "map", "auc", "score", "wer", "cer", "fid",
}

def _is_benchmark_table(headers: list[str], rows: list[list[str]]) -> bool:
    """
    Heuristic: a table is a benchmark results table if:
    - It has ≥2 columns
    - At least one header contains a metric word
    - At least one column contains numeric values
    """
    if len(headers) < 2:
        return False

    header_lower = " ".join(headers).lower()
    has_metric = any(m in header_lower for m in METRIC_WORDS)

    has_numbers = any(
        re.search(r"\d+\.?\d*", cell)
        for row in rows
        for cell in row
    )
    return has_metric and has_numbers


# ── Main extractor function ────────────────────────────────────────────────────

def extract_table_elements(
    raw_elements: list[RawElement],
    paper_id: str,
    pdf_path: str,
    save_dir: Path | None = None,
    **kwargs,
) -> list[ExtractedElement]:
    """
    Extract structured table content from Table RawElements.

    Tries three sources in order:
      1. HTML from unstructured (fastest, already parsed)
      2. pdfplumber page re-extraction (most reliable for arXiv)
      3. camelot (best for bordered tables)

    Args:
        raw_elements: Table RawElements from the router
        paper_id:     arXiv ID
        pdf_path:     Path to the PDF (for pdfplumber/camelot)
        save_dir:     Directory to save table JSON files

    Returns:
        List of ExtractedElement with element_type="table"
    """
    extracted: list[ExtractedElement] = []
    table_index = 0

    # Collect page numbers that have tables (for targeted pdfplumber calls)
    table_pages = list({el.page_number for el in raw_elements if el.page_number})

    # ── Source 1: unstructured HTML ────────────────────────────────────────────
    html_handled_pages: set[int] = set()

    for raw in raw_elements:
        html = raw.metadata.get("html", "")
        if not html:
            continue

        headers, rows = _parse_html_table(html)
        if not rows:
            continue

        table_index += 1
        is_benchmark = _is_benchmark_table(headers, rows)

        el = ExtractedElement(
            element_type = ElementType.TABLE,
            content      = _table_to_text(headers, rows),
            paper_id     = paper_id,
            page_number  = raw.page_number,
            bbox         = raw.bbox,
            metadata     = {
                "headers":      headers,
                "rows":         rows,
                "col_count":    len(headers) or (len(rows[0]) if rows else 0),
                "row_count":    len(rows),
                "is_benchmark": is_benchmark,
                "source":       "unstructured_html",
                "table_index":  table_index,
            },
        )
        extracted.append(el)
        html_handled_pages.add(raw.page_number)

        _save_table(el, save_dir, table_index)

    # ── Source 2: pdfplumber (for pages unstructured didn't handle) ────────────
    remaining_pages = [p for p in table_pages if p not in html_handled_pages]
    if remaining_pages:
        try:
            for page_num, headers, rows in _extract_with_pdfplumber(pdf_path, remaining_pages):
                if not rows:
                    continue

                table_index += 1
                is_benchmark = _is_benchmark_table(headers, rows)

                el = ExtractedElement(
                    element_type = ElementType.TABLE,
                    content      = _table_to_text(headers, rows),
                    paper_id     = paper_id,
                    page_number  = page_num,
                    metadata     = {
                        "headers":      headers,
                        "rows":         rows,
                        "col_count":    len(headers) or (len(rows[0]) if rows else 0),
                        "row_count":    len(rows),
                        "is_benchmark": is_benchmark,
                        "source":       "pdfplumber",
                        "table_index":  table_index,
                    },
                )
                extracted.append(el)
                _save_table(el, save_dir, table_index)

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

    logger.debug(
        f"[{paper_id}] Table extractor: {len(extracted)} tables "
        f"({sum(1 for e in extracted if e.metadata.get('is_benchmark')) } benchmark)"
    )
    return extracted


# ── Helpers ────────────────────────────────────────────────────────────────────

def _table_to_text(headers: list[str], rows: list[list[str]]) -> str:
    """Convert a table to a readable text representation for the graph content field."""
    lines = []
    if headers:
        lines.append(" | ".join(str(h) for h in headers))
        lines.append("-" * (sum(len(str(h)) + 3 for h in headers)))
    for row in rows:
        lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def _save_table(el: ExtractedElement, save_dir: Optional[Path], index: int) -> None:
    """Save a table to disk as JSON (for inspection and debugging)."""
    if not save_dir:
        return
    tables_dir = save_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / f"table_{index:03d}_p{el.page_number}.json"
    out.write_text(
        json.dumps(el.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )