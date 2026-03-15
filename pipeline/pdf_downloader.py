"""
pipeline/02_pdf_downloader.py
──────────────────────────────
Stage 2 of the pipeline: download arXiv PDFs from the metadata produced
in stage 1. Features:
  - Skip already-downloaded papers (hash-based dedup)
  - Exponential back-off retries via tenacity
  - Rate-limiting (respects arXiv's fair-use policy)
  - Progress bar with live ETA
  - Updates PaperMetadata.local_pdf_path + download_status in-place

Usage (standalone):
    # from a saved search JSON:
    python pipeline/02_pdf_downloader.py --input data/extracted/search_xyz.json

    # for a single arXiv ID:
    python pipeline/02_pdf_downloader.py --id 1706.03762

Usage (as module):
    from pipeline.pdf_downloader import download_papers, download_single
    updated_papers = download_papers(papers)
"""

from __future__ import annotations

import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    FileSizeColumn,
    TransferSpeedColumn,
    SpinnerColumn,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings
from pipeline.arxiv_search import PaperMetadata  # re-use our data model

console = Console()

# arXiv asks bots to identify themselves in the User-Agent header
REQUEST_HEADERS = {
    "User-Agent": "ArXivGraphRAG/1.0 (research project; contact: your@email.com)",
}


# ── PDF naming & dedup ────────────────────────────────────────────────────────

def _pdf_filename(paper: PaperMetadata) -> str:
    """
    Deterministic filename for a paper PDF.
    Format: <arxiv_id>.pdf  (slashes replaced so it's filesystem-safe)
    Example: 2301.07041.pdf
    """
    safe_id = paper.arxiv_id.replace("/", "_")
    return f"{safe_id}.pdf"


def _is_valid_pdf(path: Path, min_bytes: int = 10_000) -> bool:
    """
    Quick sanity check: does the file exist, is it large enough,
    and does it start with the PDF magic bytes?
    """
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with open(path, "rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


# ── Retry-wrapped download ────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    before_sleep=before_sleep_log(logger, "WARNING"),   # type: ignore[arg-type]
    reraise=True,
)
def _fetch_pdf_bytes(url: str, timeout: int = 60) -> bytes:
    """
    Download PDF bytes from a URL with automatic retry on network errors.
    The @retry decorator handles exponential back-off automatically.
    """
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, stream=True)
    response.raise_for_status()  # raises HTTPError for 4xx / 5xx

    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


# ── Single paper download ─────────────────────────────────────────────────────

def download_single(
    paper: PaperMetadata,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> PaperMetadata:
    """
    Download the PDF for a single paper. Mutates and returns the paper
    with updated local_pdf_path and download_status fields.

    Args:
        paper:      PaperMetadata with a valid pdf_url
        output_dir: Where to save (default: settings.pdf_dir)
        force:      Re-download even if the file already exists

    Returns:
        The same PaperMetadata object with updated fields.
    """
    output_dir = output_dir or settings.pdf_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / _pdf_filename(paper)

    # ── Skip if already downloaded ────────────────────────────────────────────
    if not force and _is_valid_pdf(dest):
        logger.debug(f"Skip (exists): {dest.name}")
        paper.local_pdf_path = str(dest)
        paper.download_status = "downloaded"
        return paper

    # ── Download ──────────────────────────────────────────────────────────────
    logger.info(f"Downloading: [{paper.arxiv_id}] {paper.title[:60]}")

    try:
        pdf_bytes = _fetch_pdf_bytes(paper.pdf_url)
    except Exception as exc:
        logger.error(f"Failed to download {paper.arxiv_id}: {exc}")
        paper.download_status = "failed"
        return paper

    # ── Validate & save ───────────────────────────────────────────────────────
    if not pdf_bytes[:5] == b"%PDF-":
        logger.warning(f"Unexpected content (not a PDF): {paper.arxiv_id}")
        paper.download_status = "failed"
        return paper

    dest.write_bytes(pdf_bytes)
    paper.local_pdf_path = str(dest)
    paper.download_status = "downloaded"
    logger.success(f"Saved ({len(pdf_bytes)/1024:.0f} KB): {dest.name}")
    return paper


# ── Batch download ────────────────────────────────────────────────────────────

def download_papers(
    papers: list[PaperMetadata],
    output_dir: Optional[Path] = None,
    delay: Optional[float] = None,
    force: bool = False,
) -> list[PaperMetadata]:
    """
    Download PDFs for a list of papers with progress tracking and rate limiting.

    Args:
        papers:     List of PaperMetadata objects from stage 1
        output_dir: Download destination (default: settings.pdf_dir)
        delay:      Seconds to wait between downloads (default: from settings)
        force:      Re-download even if files already exist

    Returns:
        Same list with updated download_status and local_pdf_path fields.

    Example:
        papers = search_arxiv("transformers", max_results=10)
        papers = download_papers(papers)
        downloaded = [p for p in papers if p.download_status == "downloaded"]
    """
    delay = delay if delay is not None else settings.arxiv_download_delay

    downloaded = sum(
        1 for p in papers
        if _is_valid_pdf((output_dir or settings.pdf_dir) / _pdf_filename(p))
    )
    to_fetch = len(papers) - downloaded if not force else len(papers)

    console.print(
        f"\n[bold]PDF Downloader[/bold]  "
        f"{len(papers)} papers total · "
        f"[green]{downloaded} already cached[/green] · "
        f"[yellow]{to_fetch} to download[/yellow]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading PDFs…", total=len(papers))

        for i, paper in enumerate(papers):
            progress.update(
                task,
                description=f"[{paper.arxiv_id}] {paper.title[:45]}…",
                advance=0,
            )
            download_single(paper, output_dir=output_dir, force=force)
            progress.advance(task)

            # Rate-limit: only sleep before the next actual download
            is_last = i == len(papers) - 1
            if not is_last and paper.download_status == "downloaded" and to_fetch > 0:
                time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok    = [p for p in papers if p.download_status == "downloaded"]
    failed = [p for p in papers if p.download_status == "failed"]

    console.print(
        f"\n[bold green]✓ {len(ok)} downloaded[/bold green]  "
        f"[bold red]✗ {len(failed)} failed[/bold red]"
    )
    if failed:
        for p in failed:
            console.print(f"  [red]✗[/red] {p.arxiv_id}: {p.title[:60]}")

    return papers


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_metadata_batch(json_path: Path) -> list[PaperMetadata]:
    """Load a list of PaperMetadata from a JSON file saved by stage 1."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    # Support both raw list and the dict-with-papers format
    records = raw if isinstance(raw, list) else raw.get("papers", [])
    return [PaperMetadata(**p) for p in records]


def save_metadata_batch(papers: list[PaperMetadata], output_path: Path) -> None:
    """Persist updated PaperMetadata (with download status) back to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "count": len(papers),
        "downloaded": sum(1 for p in papers if p.download_status == "downloaded"),
        "failed": sum(1 for p in papers if p.download_status == "failed"),
        "papers": [p.to_dict() for p in papers],
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Saved updated metadata → {output_path}")


# ── Download directory report ─────────────────────────────────────────────────

def report_downloads(pdf_dir: Optional[Path] = None) -> None:
    """Print a summary of what's in the PDF directory."""
    pdf_dir = pdf_dir or settings.pdf_dir
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    total_mb = sum(p.stat().st_size for p in pdfs) / 1_048_576

    console.print(
        f"\n[bold]PDF Cache:[/bold] {pdf_dir}\n"
        f"  {len(pdfs)} files · {total_mb:.1f} MB total"
    )
    for pdf in pdfs[:10]:
        size_kb = pdf.stat().st_size / 1024
        console.print(f"  [dim]{pdf.name}[/dim]  {size_kb:.0f} KB")
    if len(pdfs) > 10:
        console.print(f"  [dim]… and {len(pdfs)-10} more[/dim]")


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs for arXiv papers from a stage-1 metadata JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/02_pdf_downloader.py --input data/extracted/search_xyz.json
  python pipeline/02_pdf_downloader.py --id 1706.03762
  python pipeline/02_pdf_downloader.py --input search.json --force --delay 5
  python pipeline/02_pdf_downloader.py --report
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  "-i", type=str, help="Path to stage-1 JSON metadata file")
    group.add_argument("--id",           type=str, help="Download a single paper by arXiv ID")
    group.add_argument("--report", "-r", action="store_true", help="Report what's in the PDF cache")

    parser.add_argument("--output", "-o", type=str, default=None, help="Save updated metadata here")
    parser.add_argument("--delay",  "-d", type=float, default=None, help="Seconds between downloads")
    parser.add_argument("--force",  "-f", action="store_true", help="Re-download existing files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.report:
        report_downloads()
        return

    if args.id:
        # Fetch metadata on the fly, then download
        from pipeline.arxiv_search import fetch_paper_by_id
        paper = fetch_paper_by_id(args.id)
        if not paper:
            console.print(f"[red]Paper not found: {args.id}[/red]")
            return
        papers = [paper]
    else:
        papers = load_metadata_batch(Path(args.input))
        console.print(f"Loaded {len(papers)} paper records from {args.input}")

    papers = download_papers(papers, delay=args.delay, force=args.force)

    # Save updated metadata
    if args.output:
        save_metadata_batch(papers, Path(args.output))
    elif args.input:
        out = Path(args.input).with_stem(Path(args.input).stem + "_downloaded")
        save_metadata_batch(papers, out)
        console.print(f"\n[dim]Updated metadata → {out}[/dim]")


if __name__ == "__main__":
    main()