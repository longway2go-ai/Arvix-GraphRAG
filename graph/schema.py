"""
graph/schema.py
────────────────
Defines all Neo4j node labels, relationship types, and their
required properties. Think of this as the "create table" equivalent
for the knowledge graph.

No Neo4j connection needed here — pure Python constants.
Imported by graph_builder.py and cypher_queries.py.
"""

from __future__ import annotations


# ── Node Labels ───────────────────────────────────────────────────────────────

class NodeLabel:
    PAPER    = "Paper"
    AUTHOR   = "Author"
    METHOD   = "Method"
    DATASET  = "Dataset"
    EQUATION = "Equation"
    TABLE    = "Table"
    FIGURE   = "Figure"
    CONCEPT  = "Concept"
    TOPIC    = "Topic"       # the user's search topic (e.g. "transformer attention")


# ── Relationship Types ────────────────────────────────────────────────────────

class RelType:
    # Author → Paper
    WROTE          = "WROTE"

    # Topic → Paper
    INCLUDES       = "INCLUDES"

    # Paper → Method / Dataset / Concept
    PROPOSES       = "PROPOSES"
    USES_DATASET   = "USES_DATASET"
    MENTIONS       = "MENTIONS"

    # Method → Dataset / Equation
    EVALUATED_ON   = "EVALUATED_ON"
    DEFINED_BY     = "DEFINED_BY"

    # Paper → Table / Figure / Equation
    HAS_TABLE      = "HAS_TABLE"
    HAS_FIGURE     = "HAS_FIGURE"
    HAS_EQUATION   = "HAS_EQUATION"

    # Paper → Paper
    CITES          = "CITES"


# ── Node property schemas ─────────────────────────────────────────────────────
# These are the properties each node type MUST have.
# Optional properties can be added freely in graph_builder.py.

NODE_SCHEMAS: dict[str, list[str]] = {
    NodeLabel.PAPER: [
        "arxiv_id",       # unique identifier  e.g. "1706.03762"
        "title",
        "abstract",
        "published",
        "pdf_url",
        "primary_category",
    ],
    NodeLabel.AUTHOR: [
        "name",           # unique identifier
    ],
    NodeLabel.METHOD: [
        "name",           # unique identifier  e.g. "Transformer"
    ],
    NodeLabel.DATASET: [
        "name",           # unique identifier  e.g. "ImageNet"
    ],
    NodeLabel.EQUATION: [
        "latex",          # LaTeX string  e.g. "\\frac{QK^T}{\\sqrt{d_k}}"
        "paper_id",
    ],
    NodeLabel.TABLE: [
        "paper_id",
        "page_number",
        "table_index",
    ],
    NodeLabel.FIGURE: [
        "paper_id",
        "fig_number",
    ],
    NodeLabel.CONCEPT: [
        "name",           # unique identifier  e.g. "self-attention"
    ],
    NodeLabel.TOPIC: [
        "name",           # user search topic  e.g. "transformer attention"
    ],
}


# ── Cypher MERGE keys ─────────────────────────────────────────────────────────
# The property used in MERGE to avoid duplicates.
# MERGE on this key means re-running the pipeline never creates duplicate nodes.

MERGE_KEYS: dict[str, str] = {
    NodeLabel.PAPER:    "arxiv_id",
    NodeLabel.AUTHOR:   "name",
    NodeLabel.METHOD:   "name",
    NodeLabel.DATASET:  "name",
    NodeLabel.EQUATION: "element_id",
    NodeLabel.TABLE:    "element_id",
    NodeLabel.FIGURE:   "element_id",
    NodeLabel.CONCEPT:  "name",
    NodeLabel.TOPIC:    "name",
}


# ── Neo4j index definitions ───────────────────────────────────────────────────
# These indexes are created once on first run to speed up MERGE operations.

INDEXES: list[tuple[str, str]] = [
    (NodeLabel.PAPER,    "arxiv_id"),
    (NodeLabel.AUTHOR,   "name"),
    (NodeLabel.METHOD,   "name"),
    (NodeLabel.DATASET,  "name"),
    (NodeLabel.CONCEPT,  "name"),
    (NodeLabel.TOPIC,    "name"),
    (NodeLabel.EQUATION, "element_id"),
    (NodeLabel.TABLE,    "element_id"),
    (NodeLabel.FIGURE,   "element_id"),
]