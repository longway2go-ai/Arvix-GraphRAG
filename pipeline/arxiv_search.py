"""
pipeline/01_arxiv_search.py
────────────────────────────
Stage 1 of the pipeline: query the arXiv API and return structured paper
metadata. Results are cached locally as JSON so repeat searches are instant.

Usage (standalone):
    python pipeline/01_arxiv_search.py --query "attention mechanism transformer" --max 5

Usage (as module):
    from pipeline.arxiv_search import search_arxiv, fetch_paper_by_id
    papers = search_arxiv("graph neural networks", max_results=10)
"""

from __future__ import annotations

import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import arxiv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings

console = Console()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PaperMetadata:
    """
    Structured representation of a single arXiv paper.
    This is the unit of data that flows through the rest of the pipeline.
    """
    arxiv_id: str                        # e.g.  "2301.07041"
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]               # e.g.  ["cs.LG", "cs.AI"]
    published: str                       # ISO date string
    updated: str
    pdf_url: str                         # direct link to the PDF
    entry_url: str                       # abstract page URL
    primary_category: str
    comment: Optional[str] = None        # author comment (e.g. "ICLR 2024")
    doi: Optional[str] = None
    journal_ref: Optional[str] = None

    # Set by the downloader (stage 2)
    local_pdf_path: Optional[str] = None
    download_status: str = "pending"     # pending | downloaded | failed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_arxiv_result(cls, result: arxiv.Result) -> "PaperMetadata":
        """Convert an arxiv.Result object into our structured PaperMetadata."""
        return cls(
            arxiv_id=result.get_short_id(),        # strips version suffix
            title=result.title.replace("\n", " ").strip(),
            authors=[str(a) for a in result.authors],
            abstract=result.summary.replace("\n", " ").strip(),
            categories=result.categories,
            published=result.published.isoformat(),
            updated=result.updated.isoformat(),
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
            primary_category=result.primary_category,
            comment=result.comment,
            doi=result.doi,
            journal_ref=result.journal_ref,
        )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(query: str, max_results: int) -> str:
    """Deterministic cache filename for a (query, max_results) pair."""
    h = hashlib.md5(f"{query}|{max_results}".encode()).hexdigest()[:10]
    slug = query[:40].replace(" ", "_").lower()
    return f"{slug}_{h}.json"


def _load_from_cache(cache_path: Path) -> Optional[list[PaperMetadata]]:
    if not cache_path.exists():
        return None
    logger.debug(f"Cache hit: {cache_path.name}")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return [PaperMetadata(**p) for p in raw]


def _save_to_cache(papers: list[PaperMetadata], cache_path: Path) -> None:
    cache_path.write_text(
        json.dumps([p.to_dict() for p in papers], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug(f"Cached {len(papers)} papers → {cache_path.name}")


# ── Core search functions ──────────────────────────────────────────────────────

def search_arxiv(
    query: str,
    max_results: int = None,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    use_cache: bool = True,
) -> list[PaperMetadata]:
    """
    Search arXiv by keyword query and return a list of PaperMetadata objects.

    Args:
        query:        Natural language or fielded query (e.g. "ti:transformer AND cat:cs.LG")
        max_results:  How many papers to retrieve (default: from settings)
        sort_by:      arxiv.SortCriterion.Relevance | SubmittedDate | LastUpdatedDate
        use_cache:    If True, returns cached results when available

    Returns:
        List of PaperMetadata, sorted by relevance (or your chosen criterion)

    Example:
        papers = search_arxiv("attention is all you need", max_results=5)
    """
    max_results = max_results or settings.arxiv_max_results

    # ── Check cache ──────────────────────────────────────────────────────────
    cache_file = settings.cache_dir / _cache_key(query, max_results)
    if use_cache:
        cached = _load_from_cache(cache_file)
        if cached is not None:
            logger.info(f"Loaded {len(cached)} papers from cache for: '{query}'")
            return cached

    # ── Hit the API ───────────────────────────────────────────────────────────
    logger.info(f"Searching arXiv: '{query}' (max={max_results})")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    papers: list[PaperMetadata] = []
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=1.0,      # arXiv rate-limit: 1 req/sec
        num_retries=5,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Fetching from arXiv…", total=max_results)
        for result in client.results(search):
            papers.append(PaperMetadata.from_arxiv_result(result))
            progress.advance(task)

    logger.success(f"Found {len(papers)} papers for: '{query}'")

    # ── Save to cache ─────────────────────────────────────────────────────────
    if papers:
        _save_to_cache(papers, cache_file)

    return papers


def fetch_paper_by_id(arxiv_id: str, use_cache: bool = True) -> Optional[PaperMetadata]:
    """
    Fetch a single paper by its arXiv ID (e.g. "1706.03762").

    Args:
        arxiv_id:   arXiv paper ID with or without version (e.g. "1706.03762v5")
        use_cache:  Return cached result if available

    Returns:
        PaperMetadata if found, None otherwise

    Example:
        paper = fetch_paper_by_id("1706.03762")  # "Attention Is All You Need"
    """
    cache_file = settings.cache_dir / f"paper_{arxiv_id.replace('/', '_')}.json"

    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        logger.debug(f"Cache hit for paper: {arxiv_id}")
        return PaperMetadata(**raw)

    logger.info(f"Fetching paper by ID: {arxiv_id}")

    client = arxiv.Client(num_retries=5)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        logger.warning(f"Paper not found: {arxiv_id}")
        return None

    paper = PaperMetadata.from_arxiv_result(results[0])

    cache_file.write_text(
        json.dumps(paper.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.success(f"Fetched: [{paper.arxiv_id}] {paper.title[:80]}")
    return paper


def save_metadata_batch(papers: list[PaperMetadata], output_path: Path) -> Path:
    """
    Persist a list of PaperMetadata to a single JSON file.
    Useful for passing the batch to the next pipeline stage.

    Returns the path written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "query_timestamp": datetime.utcnow().isoformat(),
        "count": len(papers),
        "papers": [p.to_dict() for p in papers],
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved {len(papers)} paper records → {output_path}")
    return output_path


# ── Pretty display ─────────────────────────────────────────────────────────────

def display_results(papers: list[PaperMetadata]) -> None:
    """Print a Rich table summarizing the search results."""
    table = Table(
        title=f"arXiv Search Results ({len(papers)} papers)",
        show_lines=True,
        highlight=True,
    )
    table.add_column("#",        style="dim", width=4, justify="right")
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("Title",    style="bold white", max_width=50)
    table.add_column("Authors",  style="green", max_width=25)
    table.add_column("Category", style="yellow", width=10)
    table.add_column("Date",     style="dim", width=12)

    for i, p in enumerate(papers, 1):
        first_author = p.authors[0] if p.authors else "Unknown"
        author_str = first_author + (f" +{len(p.authors)-1}" if len(p.authors) > 1 else "")
        table.add_row(
            str(i),
            p.arxiv_id,
            p.title[:80] + ("…" if len(p.title) > 80 else ""),
            author_str,
            p.primary_category,
            p.published[:10],
        )

    console.print(table)


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search arXiv and cache paper metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/01_arxiv_search.py --query "attention transformer"
  python pipeline/01_arxiv_search.py --query "graph neural networks" --max 20 --sort date
  python pipeline/01_arxiv_search.py --id 1706.03762
  python pipeline/01_arxiv_search.py --query "BERT language model" --no-cache
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", type=str, help="Search query string")
    group.add_argument("--id",    "-i", type=str, help="Fetch a single paper by arXiv ID")

    parser.add_argument("--max",      "-m", type=int, default=None,      help="Max results (default from .env)")
    parser.add_argument("--sort",     "-s", type=str, default="relevance",
                        choices=["relevance", "date", "updated"],         help="Sort criterion")
    parser.add_argument("--no-cache", action="store_true",                help="Bypass cache and re-fetch")
    parser.add_argument("--output",   "-o", type=str, default=None,      help="Save results to this JSON file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "date":      arxiv.SortCriterion.SubmittedDate,
        "updated":   arxiv.SortCriterion.LastUpdatedDate,
    }

    if args.id:
        paper = fetch_paper_by_id(args.id, use_cache=not args.no_cache)
        if paper:
            display_results([paper])
            if args.output:
                save_metadata_batch([paper], Path(args.output))
    else:
        papers = search_arxiv(
            query=args.query,
            max_results=args.max,
            sort_by=sort_map[args.sort],
            use_cache=not args.no_cache,
        )
        display_results(papers)

        if args.output:
            save_metadata_batch(papers, Path(args.output))
        else:
            # Default: save to data/cache/ with an auto-generated name
            default_out = settings.extracted_dir / f"search_{int(time.time())}.json"
            save_metadata_batch(papers, default_out)
            console.print(f"\n[dim]Results saved → {default_out}[/dim]")


if __name__ == "__main__":
    main()