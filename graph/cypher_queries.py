"""
graph/cypher_queries.py
────────────────────────
Named Cypher queries used by the Graph RAG engine (Milestone 4)
and the Streamlit UI.

All queries are pure Cypher strings with $param placeholders.
Execute them via:
    client.run(CypherQueries.PAPERS_BY_TOPIC, {"topic": "transformer attention"})
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.neo4j_client import Neo4jClient


class CypherQueries:
    """
    All reusable Cypher queries as class-level string constants.
    Organised by what question they answer.
    """

    # ── Topic queries ─────────────────────────────────────────────────────────

    # All papers under a topic
    PAPERS_BY_TOPIC = """
        MATCH (t:Topic {name: $topic})-[:INCLUDES]->(p:Paper)
        RETURN p.arxiv_id   AS arxiv_id,
               p.title      AS title,
               p.abstract   AS abstract,
               p.published  AS published,
               p.authors_str AS authors
        ORDER BY p.published DESC
    """

    # All topics in the graph
    ALL_TOPICS = """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)-[:INCLUDES]->(p:Paper)
        RETURN t.name AS topic, count(p) AS paper_count
        ORDER BY paper_count DESC
    """

    # ── Paper queries ─────────────────────────────────────────────────────────

    # Full paper detail
    PAPER_BY_ID = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        RETURN p
    """

    # Papers that use a specific dataset
    PAPERS_BY_DATASET = """
        MATCH (p:Paper)-[:USES_DATASET]->(d:Dataset {name: $dataset_name})
        RETURN p.arxiv_id AS arxiv_id,
               p.title    AS title,
               p.published AS published
        ORDER BY p.published DESC
    """

    # Papers that propose a specific method
    PAPERS_BY_METHOD = """
        MATCH (p:Paper)-[:PROPOSES]->(m:Method {name: $method_name})
        RETURN p.arxiv_id AS arxiv_id,
               p.title    AS title,
               p.published AS published
    """

    # Papers by a specific author
    PAPERS_BY_AUTHOR = """
        MATCH (a:Author {name: $author_name})-[:WROTE]->(p:Paper)
        RETURN p.arxiv_id AS arxiv_id,
               p.title    AS title,
               p.published AS published
        ORDER BY p.published DESC
    """

    # ── Method queries ────────────────────────────────────────────────────────

    # All methods proposed in a topic
    METHODS_BY_TOPIC = """
        MATCH (t:Topic {name: $topic})-[:INCLUDES]->(p:Paper)-[:PROPOSES]->(m:Method)
        RETURN DISTINCT m.name AS method,
               collect(DISTINCT p.arxiv_id) AS papers,
               count(DISTINCT p) AS paper_count
        ORDER BY paper_count DESC
    """

    # Datasets a method was evaluated on
    DATASETS_FOR_METHOD = """
        MATCH (m:Method {name: $method_name})-[:EVALUATED_ON]->(d:Dataset)
        RETURN d.name AS dataset
    """

    # ── Table queries ─────────────────────────────────────────────────────────

    # All tables in a paper
    TABLES_BY_PAPER = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})-[:HAS_TABLE]->(t:Table)
        RETURN t.element_id  AS id,
               t.page_number AS page,
               t.is_benchmark AS is_benchmark,
               t.headers     AS headers,
               t.content     AS content,
               t.row_count   AS rows,
               t.col_count   AS cols
        ORDER BY t.page_number
    """

    # All benchmark tables in a topic
    BENCHMARK_TABLES_BY_TOPIC = """
        MATCH (t:Topic {name: $topic})-[:INCLUDES]->(p:Paper)-[:HAS_TABLE]->(tbl:Table)
        WHERE tbl.is_benchmark = true
        RETURN p.title       AS paper,
               p.arxiv_id   AS arxiv_id,
               tbl.headers  AS headers,
               tbl.content  AS content,
               tbl.page_number AS page
        ORDER BY p.published DESC
    """

    # ── Equation queries ──────────────────────────────────────────────────────

    # All equations in a paper
    EQUATIONS_BY_PAPER = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})-[:HAS_EQUATION]->(e:Equation)
        RETURN e.element_id  AS id,
               e.latex       AS latex,
               e.page_number AS page,
               e.source      AS source
        ORDER BY e.page_number
    """

    # ── Figure queries ────────────────────────────────────────────────────────

    # All figures in a paper
    FIGURES_BY_PAPER = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})-[:HAS_FIGURE]->(f:Figure)
        RETURN f.element_id  AS id,
               f.fig_number  AS fig_number,
               f.caption     AS caption,
               f.image_path  AS image_path,
               f.page_number AS page
        ORDER BY f.page_number
    """

    # ── Graph stats ───────────────────────────────────────────────────────────

    GRAPH_STATS = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """

    RELATIONSHIP_STATS = """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY count DESC
    """

    # ── RAG context builder ───────────────────────────────────────────────────
    # This query builds the context paragraph passed to GPT-4o-mini.
    # It retrieves the full neighbourhood of relevant nodes for a paper.

    RAG_CONTEXT_FOR_PAPER = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        OPTIONAL MATCH (p)-[:PROPOSES]->(m:Method)
        OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
        OPTIONAL MATCH (p)-[:HAS_EQUATION]->(e:Equation)
        OPTIONAL MATCH (p)-[:HAS_TABLE]->(t:Table)
        OPTIONAL MATCH (p)-[:HAS_FIGURE]->(f:Figure)
        OPTIONAL MATCH (p)-[:MENTIONS]->(c:Concept)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        RETURN
            p.title       AS title,
            p.abstract    AS abstract,
            p.published   AS published,
            collect(DISTINCT a.name)  AS authors,
            collect(DISTINCT m.name)  AS methods,
            collect(DISTINCT d.name)  AS datasets,
            collect(DISTINCT e.latex) AS equations,
            collect(DISTINCT t.content)[..3] AS tables,
            collect(DISTINCT f.caption)[..5] AS figure_captions,
            collect(DISTINCT c.name)  AS concepts
    """

    # Multi-paper context (used when the question spans multiple papers)
    RAG_CONTEXT_FOR_TOPIC = """
        MATCH (t:Topic {name: $topic})-[:INCLUDES]->(p:Paper)
        OPTIONAL MATCH (p)-[:PROPOSES]->(m:Method)
        OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        RETURN
            p.arxiv_id    AS arxiv_id,
            p.title       AS title,
            p.abstract    AS abstract,
            collect(DISTINCT m.name) AS methods,
            collect(DISTINCT d.name) AS datasets,
            collect(DISTINCT a.name) AS authors
        ORDER BY p.published DESC
        LIMIT $limit
    """


# ── Query runner helper ───────────────────────────────────────────────────────

def run_query(
    query:  str,
    params: dict,
    client: Neo4jClient | None = None,
) -> list[dict]:
    """
    Execute a named query and return results.
    Creates a temporary client if none provided.

    Example:
        results = run_query(
            CypherQueries.PAPERS_BY_DATASET,
            {"dataset_name": "ImageNet"}
        )
    """
    if client:
        return client.run(query, params)

    with Neo4jClient() as c:
        return c.run(query, params)


# ── CLI for quick graph inspection ────────────────────────────────────────────

if __name__ == "__main__":
    from rich.table import Table
    from rich.console import Console
    console = Console()

    with Neo4jClient() as client:
        console.print("\n[bold cyan]Graph Statistics[/bold cyan]")

        # Node counts
        node_table = Table(title="Nodes", show_lines=True)
        node_table.add_column("Label", style="cyan")
        node_table.add_column("Count", style="green", justify="right")
        for row in client.run(CypherQueries.GRAPH_STATS):
            node_table.add_row(str(row["label"]), str(row["count"]))
        console.print(node_table)

        # Relationship counts
        rel_table = Table(title="Relationships", show_lines=True)
        rel_table.add_column("Type", style="yellow")
        rel_table.add_column("Count", style="green", justify="right")
        for row in client.run(CypherQueries.RELATIONSHIP_STATS):
            rel_table.add_row(str(row["relationship"]), str(row["count"]))
        console.print(rel_table)

        # Topics
        topic_table = Table(title="Topics in Graph", show_lines=True)
        topic_table.add_column("Topic", style="white")
        topic_table.add_column("Papers", style="green", justify="right")
        for row in client.run(CypherQueries.ALL_TOPICS):
            topic_table.add_row(str(row["topic"]), str(row["paper_count"]))
        console.print(topic_table)