"""
graph/neo4j_client.py
──────────────────────
Neo4j AuraDB connection manager.
Wraps the official neo4j Python driver with:
  - Automatic connection + retry
  - Index creation on first run
  - Safe MERGE helpers (no duplicate nodes)
  - Connection test utility

Usage:
    from graph.neo4j_client import Neo4jClient
    client = Neo4jClient()
    client.test_connection()
    client.close()

    # Or use as context manager:
    with Neo4jClient() as client:
        client.run("RETURN 1")
"""

from __future__ import annotations

from typing import Any, Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.schema import INDEXES


class Neo4jClient:
    """
    Thin wrapper around the neo4j Python driver.
    Handles connection to AuraDB (neo4j+s://) and local bolt://.
    """

    def __init__(
        self,
        uri:      Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Connect to Neo4j. Reads credentials from .env if not provided.

        Args:
            uri:      e.g. "neo4j+s://xxxx.databases.neo4j.io" (AuraDB)
                      or   "bolt://localhost:7687" (local Docker)
            username: default "neo4j"
            password: your AuraDB or Docker password
            database: default "neo4j"
        """
        from config.settings import settings

        self._uri      = uri      or settings.neo4j_uri
        self._username = username or settings.neo4j_username
        self._password = password or settings.neo4j_password
        self._database = database or settings.neo4j_database
        self._driver   = None

        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Open the driver connection. Called automatically on __init__."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
            )
            logger.success(f"Connected to Neo4j: {self._uri}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def close(self) -> None:
        """Close the driver. Always call this when done."""
        if self._driver:
            self._driver.close()
            logger.debug("Neo4j driver closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Query execution ───────────────────────────────────────────────────────

    def run(
        self,
        cypher: str,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """
        Execute a Cypher query and return results as a list of dicts.

        Args:
            cypher: Cypher query string
            params: Query parameters (use $param_name in query)

        Returns:
            List of result records as dicts

        Example:
            results = client.run(
                "MATCH (p:Paper {arxiv_id: $id}) RETURN p.title AS title",
                {"id": "1706.03762"}
            )
        """
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def run_write(
        self,
        cypher: str,
        params: Optional[dict] = None,
    ) -> Any:
        """
        Execute a write Cypher query (CREATE / MERGE / SET).
        Uses an explicit write transaction for AuraDB compatibility.
        """
        with self._driver.session() as session:
            return session.execute_write(
                lambda tx: tx.run(cypher, params or {}).consume()
            )

    def run_batch(
        self,
        cypher: str,
        params_list: list[dict],
        batch_size: int = 100,
    ) -> int:
        """
        Execute the same Cypher query for a list of parameter dicts.
        Uses UNWIND for efficiency — much faster than running one query per item.

        Args:
            cypher:      Cypher that uses UNWIND $rows AS row
            params_list: List of dicts, one per item
            batch_size:  How many items to send per transaction

        Returns:
            Total number of items processed

        Example:
            client.run_batch(
                "UNWIND $rows AS row MERGE (p:Paper {arxiv_id: row.arxiv_id}) SET p += row",
                [{"arxiv_id": "1706.03762", "title": "Attention Is All You Need"}, ...]
            )
        """
        total = 0
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i : i + batch_size]
            self.run_write(cypher, {"rows": batch})
            total += len(batch)
        return total

    # ── Schema setup ──────────────────────────────────────────────────────────

    def create_indexes(self) -> None:
        """
        Create all indexes defined in schema.py.
        Safe to call multiple times — uses IF NOT EXISTS.
        Run this once before populating the graph.
        """
        logger.info("Creating Neo4j indexes…")
        for label, prop in INDEXES:
            cypher = (
                f"CREATE INDEX {label.lower()}_{prop}_idx IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{prop})"
            )
            try:
                self.run_write(cypher)
                logger.debug(f"  Index: {label}.{prop}")
            except Exception as e:
                logger.warning(f"  Index {label}.{prop} skipped: {e}")
        logger.success("Indexes ready")

    def clear_graph(self, confirm: bool = False) -> None:
        """
        Delete ALL nodes and relationships.
        Requires confirm=True to prevent accidents.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to clear the entire graph.")
        logger.warning("Clearing entire graph…")
        self.run_write("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared.")

    # ── Connection test ───────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """
        Verify the connection is alive and print database info.
        Returns True if connected, False otherwise.
        """
        try:
            result = self.run("RETURN 'connected' AS status")
            status = result[0]["status"] if result else "unknown"
            logger.success(f"Neo4j connection test: {status}")

            # Print node counts if graph has data
            counts = self.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
                "ORDER BY count DESC"
            )
            if counts:
                logger.info("Current graph contents:")
                for row in counts:
                    logger.info(f"  {row['label']:<15} {row['count']} nodes")
            else:
                logger.info("Graph is empty — ready to populate")

            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    # ── MERGE helpers ─────────────────────────────────────────────────────────

    def merge_node(
        self,
        label:      str,
        merge_key:  str,
        merge_val:  Any,
        properties: dict,
    ) -> None:
        """
        MERGE a single node — creates it if it doesn't exist, updates if it does.
        This is the key operation that prevents duplicate nodes on re-runs.

        Args:
            label:      Node label e.g. "Paper"
            merge_key:  Property to merge on e.g. "arxiv_id"
            merge_val:  Value of the merge key e.g. "1706.03762"
            properties: All properties to set on the node

        Example:
            client.merge_node("Paper", "arxiv_id", "1706.03762",
                              {"title": "Attention Is All You Need", ...})
        """
        cypher = (
            f"MERGE (n:{label} {{{merge_key}: $merge_val}}) "
            f"SET n += $props"
        )
        self.run_write(cypher, {"merge_val": merge_val, "props": properties})

    def merge_relationship(
        self,
        from_label:     str,
        from_key:       str,
        from_val:       Any,
        rel_type:       str,
        to_label:       str,
        to_key:         str,
        to_val:         Any,
        rel_properties: Optional[dict] = None,
    ) -> None:
        """
        MERGE a relationship between two existing nodes.

        Example:
            client.merge_relationship(
                "Author", "name", "Ashish Vaswani",
                "WROTE",
                "Paper", "arxiv_id", "1706.03762"
            )
        """
        props_clause = "SET r += $rel_props" if rel_properties else ""
        cypher = (
            f"MATCH (a:{from_label} {{{from_key}: $from_val}}) "
            f"MATCH (b:{to_label}   {{{to_key}:   $to_val}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"{props_clause}"
        )
        params = {
            "from_val":  from_val,
            "to_val":    to_val,
            "rel_props": rel_properties or {},
        }
        self.run_write(cypher, params)