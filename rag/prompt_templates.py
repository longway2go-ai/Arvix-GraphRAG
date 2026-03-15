"""
rag/prompt_templates.py
────────────────────────
All prompt templates used by the Graph RAG engine.
Keeping prompts here makes them easy to tune without touching logic.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge of machine learning, computer vision, and NLP.

You are given structured knowledge extracted from arXiv research papers — including abstracts, methods, datasets, equations, tables, figures, and author information from a Neo4j knowledge graph.

Your job is to answer the user's question with HIGH QUALITY, DETAILED responses. Follow these rules strictly:

1. ALWAYS cite the specific paper title and arXiv ID when referencing results
2. STRUCTURE your answer with clear sections when the question has multiple parts
3. For methods — explain what the method does, not just its name
4. For datasets — mention what domain they cover and why they matter
5. For benchmark results — present numbers clearly and explain what they mean
6. For equations — explain what each symbol represents after showing the formula
7. COMPARE across papers when multiple papers are relevant
8. If something is not in the context, say exactly what is missing rather than giving a vague answer
9. End with a 1-line "Key Takeaway" summarising the most important finding

Context comes from a knowledge graph — trust it completely."""


# ── RAG prompt template ───────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """Here is the knowledge graph context retrieved for your question:

{context}

---

User question: {question}

Answer based on the context above:"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context_string(graph_results: list[dict], query_type: str = "general") -> str:
    if not graph_results:
        return "No relevant information found in the knowledge graph."

    lines = []
    for i, row in enumerate(graph_results, 1):
        lines.append(f"\n{'='*50}")
        lines.append(f"PAPER {i}")
        lines.append(f"{'='*50}")

        if row.get("title"):
            lines.append(f"Title: {row['title']}")
        if row.get("arxiv_id"):
            lines.append(f"ArXiv ID: {row['arxiv_id']}")
        if row.get("published"):
            lines.append(f"Published: {row['published'][:10]}")
        if row.get("authors"):
            authors = row["authors"]
            if isinstance(authors, list) and authors:
                lines.append(f"Authors: {', '.join(a for a in authors if a)}")

        # Full abstract — don't truncate
        if row.get("abstract"):
            lines.append(f"\nAbstract:\n{row['abstract']}")

        if row.get("methods"):
            methods = [m for m in row["methods"] if m]
            if methods:
                lines.append(f"\nProposed Methods/Models: {', '.join(methods)}")

        if row.get("datasets"):
            datasets = [d for d in row["datasets"] if d]
            if datasets:
                lines.append(f"Datasets Used: {', '.join(datasets)}")

        if row.get("equations"):
            eqs = [e for e in row["equations"] if e]
            if eqs:
                lines.append(f"\nKey Equations:")
                for eq in eqs:
                    lines.append(f"  - {eq}")

        if row.get("tables"):
            tables = [t for t in row["tables"] if t]
            if tables:
                lines.append(f"\nTables/Results:")
                for t in tables:
                    lines.append(t[:600])

        if row.get("figure_captions"):
            caps = [c for c in row["figure_captions"] if c]
            if caps:
                lines.append(f"\nFigures: {' | '.join(caps[:5])}")

        if row.get("concepts"):
            concepts = [c for c in row["concepts"] if c]
            if concepts:
                lines.append(f"Key Concepts: {', '.join(concepts)}")

    return "\n".join(lines)


# ── Query type detector ───────────────────────────────────────────────────────

def detect_query_type(question: str) -> str:
    """
    Detect what kind of question the user is asking so we can
    pick the most relevant Cypher query.
    """
    q = question.lower()

    if any(w in q for w in ["dataset", "trained on", "evaluated on", "benchmark"]):
        return "dataset"
    elif any(w in q for w in ["equation", "formula", "math", "defined", "attention score"]):
        return "equation"
    elif any(w in q for w in ["table", "results", "performance", "accuracy", "score", "comparison"]):
        return "table"
    elif any(w in q for w in ["figure", "diagram", "architecture", "image", "visualization"]):
        return "figure"
    elif any(w in q for w in ["author", "who wrote", "researcher", "proposed by"]):
        return "author"
    elif any(w in q for w in ["method", "model", "approach", "technique", "algorithm"]):
        return "method"
    else:
        return "general"