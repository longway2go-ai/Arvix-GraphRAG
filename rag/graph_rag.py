"""
rag/graph_rag.py - Fixed version
"""
from __future__ import annotations
from typing import Optional, Generator
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph.neo4j_client import Neo4jClient
from rag.prompt_templates import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, build_context_string, detect_query_type

QUERY_ALL_PAPERS = """
MATCH (p:Paper)
OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
OPTIONAL MATCH (p)-[:PROPOSES]->(m:Method)
OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
OPTIONAL MATCH (p)-[:HAS_EQUATION]->(e:Equation)
OPTIONAL MATCH (p)-[:HAS_TABLE]->(tbl:Table)
OPTIONAL MATCH (p)-[:HAS_FIGURE]->(f:Figure)
OPTIONAL MATCH (p)-[:MENTIONS]->(c:Concept)
RETURN
    p.arxiv_id    AS arxiv_id,
    p.title       AS title,
    p.abstract    AS abstract,
    p.published   AS published,
    collect(DISTINCT a.name)           AS authors,
    collect(DISTINCT m.name)           AS methods,
    collect(DISTINCT d.name)           AS datasets,
    collect(DISTINCT e.latex)[..3]     AS equations,
    collect(DISTINCT tbl.content)[..2] AS tables,
    collect(DISTINCT f.caption)[..4]   AS figure_captions,
    collect(DISTINCT c.name)           AS concepts
ORDER BY p.published DESC
LIMIT $limit
"""

QUERY_BENCHMARK_TABLES = """
MATCH (p:Paper)-[:HAS_TABLE]->(t:Table)
WHERE t.is_benchmark = true
RETURN p.title AS title, p.arxiv_id AS arxiv_id,
       t.headers AS headers, t.content AS content, t.page_number AS page
"""

QUERY_STATS     = "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
QUERY_REL_STATS = "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
QUERY_TOPIC_PAPERS = "MATCH (t:Topic {name: $topic})-[:INCLUDES]->(p:Paper) RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract, p.published AS published"

class GraphRAG:
    def __init__(self, topic: str, model: Optional[str] = None, neo4j_client: Optional[Neo4jClient] = None):
        from config.settings import settings
        self.topic  = topic.lower().strip()
        self.model  = model or settings.openai_model
        self.client = neo4j_client or Neo4jClient()
        self._openai_client = None
        logger.info(f"GraphRAG initialized — topic: '{self.topic}', model: {self.model}")

    def _get_openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            from config.settings import settings
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def close(self):
        if self.client: self.client.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def retrieve_context(self, question: str) -> tuple[list[dict], str]:
        query_type = detect_query_type(question)
        logger.debug(f"Query type: {query_type}")
        if query_type == "table":
            results = self.client.run(QUERY_BENCHMARK_TABLES, {})
            if results:
                return results, query_type
        results = self.client.run(QUERY_ALL_PAPERS, {"limit": 10})
        logger.debug(f"Retrieved {len(results)} papers from graph")
        if not results:
            logger.warning("No papers found — make sure pipeline ran successfully")
        return results, query_type

    def ask(self, question: str) -> dict:
        raw_results, query_type = self.retrieve_context(question)
        context_str = build_context_string(raw_results, query_type)
        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)
        openai = self._get_openai()
        logger.info(f"Calling {self.model} with {len(raw_results)} context records…")
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
        )
        answer  = response.choices[0].message.content
        sources = [r.get("title") or r.get("arxiv_id", "Unknown") for r in raw_results]
        return {
            "answer": answer, "context": raw_results, "query_type": query_type,
            "model": self.model, "sources": sources,
            "tokens_used": response.usage.total_tokens,
        }

    def ask_stream(self, question: str) -> Generator[str, None, None]:
        raw_results, query_type = self.retrieve_context(question)
        context_str = build_context_string(raw_results, query_type)
        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)
        openai = self._get_openai()
        stream = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2, max_tokens=1500, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def get_graph_stats(self) -> dict:
        try:
            node_counts  = self.client.run(QUERY_STATS, {})
            rel_counts   = self.client.run(QUERY_REL_STATS, {})
            topic_papers = self.client.run(QUERY_TOPIC_PAPERS, {"topic": self.topic})
            if not topic_papers:
                topic_papers = self.client.run(
                    "MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id, p.title AS title, "
                    "p.abstract AS abstract, p.published AS published LIMIT 20", {}
                )
            return {
                "nodes":       {r["label"]: r["count"] for r in node_counts if r["label"]},
                "rels":        {r["type"]:  r["count"] for r in rel_counts  if r["type"]},
                "papers":      topic_papers,
                "total_nodes": sum(r["count"] for r in node_counts),
                "total_rels":  sum(r["count"] for r in rel_counts),
            }
        except Exception as e:
            logger.error(f"Graph stats failed: {e}")
            return {"nodes": {}, "rels": {}, "papers": [], "total_nodes": 0, "total_rels": 0}