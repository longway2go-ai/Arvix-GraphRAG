"""
graph/graph_builder.py
───────────────────────
The core of Milestone 3.
Takes PaperExtractionResult objects (output of Milestone 2) and
populates the Neo4j knowledge graph with nodes and relationships.

Node creation order matters:
  1. Topic node       (the user's search query)
  2. Paper node       (one per arXiv paper)
  3. Author nodes     (from paper metadata)
  4. Method nodes     (from text entity extraction)
  5. Dataset nodes    (from text entity extraction)
  6. Concept nodes    (from text entity extraction)
  7. Table nodes      (from table extractor)
  8. Equation nodes   (from equation extractor)
  9. Figure nodes     (from figure extractor)

Then relationships are created between them all.

Usage:
    from graph.graph_builder import GraphBuilder
    builder = GraphBuilder()
    builder.ingest_topic("transformer attention", papers, extraction_results)
    builder.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.schema import NodeLabel, RelType, MERGE_KEYS
from graph.neo4j_client import Neo4jClient
from pipeline.models import ExtractedElement, PaperExtractionResult, ElementType

console = Console()


class GraphBuilder:
    """
    Builds the Neo4j knowledge graph from extracted paper data.
    One GraphBuilder instance manages one session with Neo4j.
    """

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Args:
            client: An existing Neo4jClient. If None, creates a new one.
        """
        self.client = client or Neo4jClient()
        self.client.create_indexes()

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Top-level entry point ─────────────────────────────────────────────────

    def ingest_topic(
        self,
        topic:       str,
        papers:      list[dict],
        results:     list[PaperExtractionResult],
    ) -> None:
        """
        Ingest an entire topic (search query + all its papers) into Neo4j.

        Args:
            topic:   The user's search query e.g. "transformer attention"
            papers:  List of PaperMetadata dicts from stage 1
            results: List of PaperExtractionResult from stage 2

        This is the main function called by the pipeline orchestrator.
        """
        console.print(f"\n[bold cyan]Graph Builder[/bold cyan] — topic: '{topic}'")
        console.print(f"  {len(papers)} papers  ·  {len(results)} extraction results\n")

        # 1. Create the Topic node
        self._create_topic_node(topic)

        # Build a lookup: paper_id → extraction result
        result_map = {r.paper_id: r for r in results}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting papers…", total=len(papers))

            for paper_dict in papers:
                paper_id = paper_dict.get("arxiv_id") or paper_dict.get("paper_id", "")
                progress.update(task, description=f"Ingesting [{paper_id}]…")

                try:
                    # 2. Paper + Author nodes + relationships
                    self._ingest_paper(paper_dict, topic)

                    # 3. Extracted elements (text entities, tables, equations, figures)
                    result = result_map.get(paper_id)
                    if result:
                        self._ingest_extraction_result(result)
                    else:
                        logger.warning(f"No extraction result for {paper_id}")

                except Exception as e:
                    logger.error(f"Failed to ingest {paper_id}: {e}")

                progress.advance(task)

        # Print final graph stats
        self.client.test_connection()

    # ── Paper ingestion ───────────────────────────────────────────────────────

    def _create_topic_node(self, topic: str) -> None:
        """Create or update the Topic node for this search query."""
        self.client.merge_node(
            label      = NodeLabel.TOPIC,
            merge_key  = MERGE_KEYS[NodeLabel.TOPIC],
            merge_val  = topic.lower().strip(),
            properties = {"name": topic.lower().strip(), "display_name": topic},
        )
        logger.debug(f"Topic node: '{topic}'")

    def _ingest_paper(self, paper: dict, topic: str) -> None:
        """
        Create Paper node, Author nodes, and all their relationships.
        Uses MERGE so re-running never creates duplicates.
        """
        paper_id = paper.get("arxiv_id") or paper.get("paper_id", "")
        if not paper_id:
            return

        # ── Paper node ────────────────────────────────────────────────────────
        self.client.merge_node(
            label     = NodeLabel.PAPER,
            merge_key = MERGE_KEYS[NodeLabel.PAPER],
            merge_val = paper_id,
            properties = {
                "arxiv_id":        paper_id,
                "title":           paper.get("title", ""),
                "abstract":        paper.get("abstract", "")[:2000],  # cap length
                "published":       paper.get("published", ""),
                "pdf_url":         paper.get("pdf_url", ""),
                "primary_category": paper.get("primary_category", ""),
                "authors_str":     ", ".join(paper.get("authors", [])),
            },
        )

        # ── Topic → INCLUDES → Paper ──────────────────────────────────────────
        self.client.merge_relationship(
            from_label = NodeLabel.TOPIC,
            from_key   = "name",
            from_val   = topic.lower().strip(),
            rel_type   = RelType.INCLUDES,
            to_label   = NodeLabel.PAPER,
            to_key     = "arxiv_id",
            to_val     = paper_id,
        )

        # ── Author nodes + WROTE relationships ───────────────────────────────
        for author_name in paper.get("authors", []):
            author_name = author_name.strip()
            if not author_name:
                continue

            self.client.merge_node(
                label     = NodeLabel.AUTHOR,
                merge_key = MERGE_KEYS[NodeLabel.AUTHOR],
                merge_val = author_name,
                properties = {"name": author_name},
            )
            self.client.merge_relationship(
                from_label = NodeLabel.AUTHOR,
                from_key   = "name",
                from_val   = author_name,
                rel_type   = RelType.WROTE,
                to_label   = NodeLabel.PAPER,
                to_key     = "arxiv_id",
                to_val     = paper_id,
            )

    # ── Extraction result ingestion ───────────────────────────────────────────

    def _ingest_extraction_result(self, result: PaperExtractionResult) -> None:
        """Route each ExtractedElement to its specific ingestion method."""
        for element in result.elements:
            try:
                if element.element_type in (ElementType.TEXT, ElementType.TITLE):
                    self._ingest_text_element(element)
                elif element.element_type == ElementType.TABLE:
                    self._ingest_table_element(element)
                elif element.element_type == ElementType.EQUATION:
                    self._ingest_equation_element(element)
                elif element.element_type == ElementType.FIGURE:
                    self._ingest_figure_element(element)
            except Exception as e:
                logger.debug(f"Element ingest failed ({element.element_type}): {e}")

    # ── Text element → Method / Dataset / Concept nodes ───────────────────────

    def _ingest_text_element(self, element: ExtractedElement) -> None:
        """
        Extract entities from text elements and create graph nodes.
        Text itself is NOT stored as a node — only the entities it contains.
        This keeps the graph lean and queryable.
        """
        paper_id = element.paper_id
        entities = element.metadata.get("entities", {})

        # ── Method nodes ──────────────────────────────────────────────────────
        for method_name in entities.get("methods", []):
            self.client.merge_node(
                label     = NodeLabel.METHOD,
                merge_key = MERGE_KEYS[NodeLabel.METHOD],
                merge_val = method_name,
                properties = {"name": method_name},
            )
            self.client.merge_relationship(
                from_label = NodeLabel.PAPER,
                from_key   = "arxiv_id",
                from_val   = paper_id,
                rel_type   = RelType.PROPOSES,
                to_label   = NodeLabel.METHOD,
                to_key     = "name",
                to_val     = method_name,
            )

        # ── Dataset nodes ─────────────────────────────────────────────────────
        for dataset_name in entities.get("datasets", []):
            self.client.merge_node(
                label     = NodeLabel.DATASET,
                merge_key = MERGE_KEYS[NodeLabel.DATASET],
                merge_val = dataset_name,
                properties = {"name": dataset_name},
            )
            self.client.merge_relationship(
                from_label = NodeLabel.PAPER,
                from_key   = "arxiv_id",
                from_val   = paper_id,
                rel_type   = RelType.USES_DATASET,
                to_label   = NodeLabel.DATASET,
                to_key     = "name",
                to_val     = dataset_name,
            )

        # ── Concept nodes (from ORG / WORK_OF_ART entities) ──────────────────
        for concept in entities.get("concepts", []):
            if len(concept) < 3:
                continue
            self.client.merge_node(
                label     = NodeLabel.CONCEPT,
                merge_key = MERGE_KEYS[NodeLabel.CONCEPT],
                merge_val = concept.lower(),
                properties = {"name": concept.lower(), "display_name": concept},
            )
            self.client.merge_relationship(
                from_label = NodeLabel.PAPER,
                from_key   = "arxiv_id",
                from_val   = paper_id,
                rel_type   = RelType.MENTIONS,
                to_label   = NodeLabel.CONCEPT,
                to_key     = "name",
                to_val     = concept.lower(),
            )

    # ── Table element → Table node ────────────────────────────────────────────

    def _ingest_table_element(self, element: ExtractedElement) -> None:
        """Create a Table node and link it to its Paper."""
        self.client.merge_node(
            label     = NodeLabel.TABLE,
            merge_key = MERGE_KEYS[NodeLabel.TABLE],
            merge_val = element.element_id,
            properties = {
                "element_id":   element.element_id,
                "paper_id":     element.paper_id,
                "page_number":  element.page_number,
                "table_index":  element.metadata.get("table_index", 0),
                "content":      element.content[:1000],   # truncate for storage
                "col_count":    element.metadata.get("col_count", 0),
                "row_count":    element.metadata.get("row_count", 0),
                "is_benchmark": element.metadata.get("is_benchmark", False),
                "headers":      json.dumps(element.metadata.get("headers", [])),
            },
        )
        self.client.merge_relationship(
            from_label = NodeLabel.PAPER,
            from_key   = "arxiv_id",
            from_val   = element.paper_id,
            rel_type   = RelType.HAS_TABLE,
            to_label   = NodeLabel.TABLE,
            to_key     = "element_id",
            to_val     = element.element_id,
        )

        # If it's a benchmark table, also link methods to datasets via EVALUATED_ON
        if element.metadata.get("is_benchmark"):
            self._link_benchmark_table(element)

    def _link_benchmark_table(self, element: ExtractedElement) -> None:
        """
        For benchmark tables, try to link Method → EVALUATED_ON → Dataset.
        Uses header names as hints for dataset/method columns.
        """
        headers = element.metadata.get("headers", [])
        rows    = element.metadata.get("rows", [])

        if not headers or not rows:
            return

        # Heuristic: first column = model/method names, other cols = metrics/datasets
        for row in rows[:20]:   # cap at 20 rows
            if not row:
                continue
            method_name = str(row[0]).strip()
            if not method_name or len(method_name) < 2:
                continue

            # Create a method node for each row entry
            try:
                self.client.merge_node(
                    label     = NodeLabel.METHOD,
                    merge_key = "name",
                    merge_val = method_name,
                    properties = {"name": method_name, "from_table": True},
                )
                self.client.merge_relationship(
                    from_label = NodeLabel.PAPER,
                    from_key   = "arxiv_id",
                    from_val   = element.paper_id,
                    rel_type   = RelType.PROPOSES,
                    to_label   = NodeLabel.METHOD,
                    to_key     = "name",
                    to_val     = method_name,
                )
            except Exception:
                pass

    # ── Equation element → Equation node ─────────────────────────────────────

    def _ingest_equation_element(self, element: ExtractedElement) -> None:
        """Create an Equation node and link it to its Paper."""
        latex = element.metadata.get("latex") or element.content
        if not latex or len(latex) < 3:
            return

        self.client.merge_node(
            label     = NodeLabel.EQUATION,
            merge_key = MERGE_KEYS[NodeLabel.EQUATION],
            merge_val = element.element_id,
            properties = {
                "element_id":  element.element_id,
                "paper_id":    element.paper_id,
                "latex":       latex[:500],
                "page_number": element.page_number,
                "source":      element.metadata.get("source", "unknown"),
            },
        )
        self.client.merge_relationship(
            from_label = NodeLabel.PAPER,
            from_key   = "arxiv_id",
            from_val   = element.paper_id,
            rel_type   = RelType.HAS_EQUATION,
            to_label   = NodeLabel.EQUATION,
            to_key     = "element_id",
            to_val     = element.element_id,
        )

    # ── Figure element → Figure node ──────────────────────────────────────────

    def _ingest_figure_element(self, element: ExtractedElement) -> None:
        """Create a Figure node and link it to its Paper."""
        self.client.merge_node(
            label     = NodeLabel.FIGURE,
            merge_key = MERGE_KEYS[NodeLabel.FIGURE],
            merge_val = element.element_id,
            properties = {
                "element_id":  element.element_id,
                "paper_id":    element.paper_id,
                "fig_number":  element.metadata.get("fig_number", ""),
                "caption":     element.content[:500],
                "image_path":  element.metadata.get("image_path", ""),
                "page_number": element.page_number,
            },
        )
        self.client.merge_relationship(
            from_label = NodeLabel.PAPER,
            from_key   = "arxiv_id",
            from_val   = element.paper_id,
            rel_type   = RelType.HAS_FIGURE,
            to_label   = NodeLabel.FIGURE,
            to_key     = "element_id",
            to_val     = element.element_id,
        )


# ── Convenience loader ────────────────────────────────────────────────────────

def load_extraction_results(extracted_dir: Path) -> list[PaperExtractionResult]:
    """
    Load all extraction.json files from data/extracted/<paper_id>/
    Returns a list of PaperExtractionResult objects.
    """
    results = []
    for json_file in extracted_dir.rglob("extraction.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            results.append(PaperExtractionResult.from_dict(data))
            logger.debug(f"Loaded: {json_file.parent.name}")
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")
    logger.info(f"Loaded {len(results)} extraction results")
    return results