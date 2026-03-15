"""
pipeline/09_run_pipeline.py
────────────────────────────
Top-level orchestrator. Run the full (or partial) pipeline from a single
command. Each stage is imported as a module and called in sequence.

Usage:
    # Full pipeline (search → download → extract → graph ingest)
    python pipeline/09_run_pipeline.py --query "attention transformer" --stages all

    # Milestone 1 only (search + download)
    python pipeline/09_run_pipeline.py --query "graph neural networks" --stages ingest

    # Start from a specific arXiv ID
    python pipeline/09_run_pipeline.py --id 1706.03762 --stages ingest
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings

console = Console()

STAGE_GROUPS = {
    "ingest":  ["search", "download"],
    "extract": ["search", "download", "layout", "route"],
    "graph":   ["search", "download", "layout", "route", "graph_ingest"],
    "rag":     ["search", "download", "layout", "route", "graph_ingest", "rag_query"],
    "all":     ["search", "download", "layout", "route",
                "graph_ingest", "rag_query"],
}


def run_ingest_stages(query: str = None, arxiv_id: str = None,
                      max_results: int = None, force_download: bool = False) -> list:
    """Run stages 1 and 2: arXiv search + PDF download."""

    console.print(Rule("[bold cyan]Stage 1 · arXiv Search[/bold cyan]"))

    from pipeline.arxiv_search import search_arxiv, fetch_paper_by_id, save_metadata_batch

    if arxiv_id:
        paper = fetch_paper_by_id(arxiv_id)
        papers = [paper] if paper else []
    else:
        papers = search_arxiv(query, max_results=max_results)

    if not papers:
        logger.error("No papers found. Aborting pipeline.")
        return []

    # Persist stage-1 output
    meta_path = settings.extracted_dir / f"stage1_{int(time.time())}.json"
    save_metadata_batch(papers, meta_path)
    console.print(f"[dim]Stage 1 output → {meta_path}[/dim]\n")

    console.print(Rule("[bold cyan]Stage 2 · PDF Download[/bold cyan]"))

    from pipeline.pdf_downloader import download_papers, save_metadata_batch as save_dl

    papers = download_papers(papers, force=force_download)

    # Persist stage-2 output (with local_pdf_path filled in)
    dl_path = settings.extracted_dir / f"stage2_{int(time.time())}.json"
    save_dl(papers, dl_path)
    console.print(f"[dim]Stage 2 output → {dl_path}[/dim]\n")

    return papers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArXiv GraphRAG — full pipeline orchestrator",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", type=str,  help="arXiv search query")
    group.add_argument("--id",          type=str,  help="Single paper arXiv ID")

    parser.add_argument("--stages", "-s", type=str, default="ingest",
                        choices=list(STAGE_GROUPS.keys()),
                        help="Which stage group to run (default: ingest)")
    parser.add_argument("--max",    "-m", type=int, default=None, help="Max results")
    parser.add_argument("--force",  "-f", action="store_true",    help="Force re-download")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold white]ArXiv GraphRAG Pipeline[/bold white]\n"
        f"[dim]Stages: {args.stages}  |  "
        f"{'Query: ' + args.query if args.query else 'ID: ' + args.id}[/dim]",
        border_style="cyan",
    ))

    stages = STAGE_GROUPS[args.stages]

    if "search" in stages or "download" in stages:
        papers = run_ingest_stages(
            query=args.query,
            arxiv_id=args.id,
            max_results=args.max,
            force_download=args.force,
        )
        if not papers:
            return

    # ── Milestone 2: layout detection + extraction ────────────────────────────
    if "layout" in stages or "route" in stages:
        console.print(Rule("[bold cyan]Stages 3–8 · Layout Detection + Extraction[/bold cyan]"))

        from pipeline.element_router import route_paper_batch

        # Convert PaperMetadata list to the dict format expected by route_paper_batch
        paper_dicts = [
            {
                "paper_id": p.arxiv_id,
                "pdf_path": p.local_pdf_path,
            }
            for p in papers
            if p.download_status == "downloaded" and p.local_pdf_path
        ]

        if not paper_dicts:
            console.print("[red]No downloaded PDFs found. Run --stages ingest first.[/red]")
            return

        extraction_results = route_paper_batch(paper_dicts, prefer_ml=True)
        console.print(f"[dim]Extracted {sum(len(r.elements) for r in extraction_results)} elements "
                      f"from {len(extraction_results)} papers[/dim]\n")

    if "graph_ingest" in stages:
        console.print(Rule("[bold cyan]Stage 7 · Graph Ingest → Neo4j[/bold cyan]"))

        from graph.graph_builder import GraphBuilder, load_extraction_results
        from config.settings import settings

        extraction_results = load_extraction_results(settings.extracted_dir)
        paper_dicts = [p.to_dict() for p in papers]

        with GraphBuilder() as builder:
            builder.ingest_topic(
                topic   = args.query or args.id,
                papers  = paper_dicts,
                results = extraction_results,
            )
        console.print("[dim]Graph ingest complete[/dim]\n")

    console.print(Panel.fit("[bold green]✓ Pipeline complete[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()