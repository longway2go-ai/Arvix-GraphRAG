"""
pipeline/models.py
───────────────────
Shared data models that flow between ALL pipeline stages.
Every extractor (text, table, equation, figure) must return
an ExtractedElement — this is the contract that keeps the
graph builder simple.

Import anywhere:
    from pipeline.models import ExtractedElement, PaperExtractionResult, ElementType
"""

from __future__ import annotations

import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


# ── Element type constants ────────────────────────────────────────────────────

class ElementType(str, Enum):
    TEXT      = "text"
    TABLE     = "table"
    EQUATION  = "equation"
    FIGURE    = "figure"
    CAPTION   = "caption"
    TITLE     = "title"
    UNKNOWN   = "unknown"


# ── Core data models ──────────────────────────────────────────────────────────

@dataclass
class ExtractedElement:
    """
    The universal output unit from every specialized extractor.
    One of these is created for every text block, table, equation,
    figure, or caption detected in a PDF.

    This object flows into the graph builder (Milestone 3) which
    turns it into a Neo4j node.
    """

    # Required fields
    element_type: str          # one of ElementType values
    content:      str          # main extracted content (text / LaTeX / caption)
    paper_id:     str          # arXiv ID of the source paper

    # Auto-generated
    element_id:   str = field(
        default_factory=lambda: str(uuid.uuid4())[:12]
    )

    # Location in the PDF
    page_number:  int          = 0
    bbox:         Optional[list[float]] = None   # [x0, y0, x1, y1] in pts

    # Type-specific metadata (populated by each extractor)
    metadata:     dict[str, Any] = field(default_factory=dict)
    # Examples of what goes in metadata per type:
    #   text:     {"section": "Introduction", "entities": [...], "word_count": 120}
    #   table:    {"headers": [...], "rows": [[...]], "col_count": 5, "row_count": 8}
    #   equation: {"latex": "\\frac{...}", "confidence": 0.91, "symbol_hint": "attention"}
    #   figure:   {"caption": "...", "image_path": "...", "fig_number": "Figure 3"}
    #   caption:  {"figure_ref": "Figure 3", "associated_element_id": "..."}

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """An element is valid if it has non-empty content."""
        return bool(self.content and self.content.strip())

    def short_repr(self) -> str:
        snippet = self.content[:60].replace("\n", " ")
        return f"[{self.element_type}|p{self.page_number}] {snippet}…"


@dataclass
class PaperExtractionResult:
    """
    All extracted elements for a single paper, plus run statistics.
    This is persisted to data/extracted/<arxiv_id>/extraction.json
    and consumed by the graph builder.
    """
    paper_id:   str
    pdf_path:   str
    elements:   list[ExtractedElement] = field(default_factory=list)
    stats:      dict[str, int]         = field(default_factory=dict)
    timestamp:  str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    errors:     list[str] = field(default_factory=list)

    def add(self, element: ExtractedElement) -> None:
        """Add an element and keep stats current."""
        if element.is_valid:
            self.elements.append(element)
            key = element.element_type
            self.stats[key] = self.stats.get(key, 0) + 1

    def get_by_type(self, element_type: str) -> list[ExtractedElement]:
        return [e for e in self.elements if e.element_type == element_type]

    def summary(self) -> str:
        parts = [f"{v} {k}" for k, v in self.stats.items()]
        return f"[{self.paper_id}] " + ", ".join(parts) if parts else "empty"

    def to_dict(self) -> dict:
        return {
            "paper_id":  self.paper_id,
            "pdf_path":  self.pdf_path,
            "timestamp": self.timestamp,
            "stats":     self.stats,
            "errors":    self.errors,
            "elements":  [e.to_dict() for e in self.elements],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperExtractionResult":
        elements = [ExtractedElement(**e) for e in data.get("elements", [])]
        return cls(
            paper_id  = data["paper_id"],
            pdf_path  = data["pdf_path"],
            elements  = elements,
            stats     = data.get("stats", {}),
            timestamp = data.get("timestamp", ""),
            errors    = data.get("errors", []),
        )