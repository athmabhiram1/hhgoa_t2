"""Lightweight graph layer for retrieval.

Not LightRAG-style LLM entity extraction — a cheap metadata/co-occurrence
graph (Query / Language / QueryType / Chunk) that powers:
  1. sibling expansion: given a top passage, pull its query's other passages
     (co-occurrence boost, ~0 extra embedding cost), and
  2. the knowledge-graph visual for the demo.
"""

from __future__ import annotations

from ..config import Settings
from ..core.models import RetrievedPassage
from .neo4j_store import LABEL_BY_NS, Neo4jStore

EXPANSION_NAMESPACE = "passage_natural"


async def sibling_expand(store: Neo4jStore, seed: RetrievedPassage, *, topk: int = 4) -> list[RetrievedPassage]:
    """Fetch other passages belonging to the seed passage's query."""
    if not seed.query_id:
        return []
    label = LABEL_BY_NS[EXPANSION_NAMESPACE]
    query = (
        f"MATCH (n:`{label}`) WHERE n.query_id = $qid AND n.chunk_id <> $cid "
        f"RETURN n, 0.0 AS score LIMIT $topk"
    )
    rows = await store._run(query, {"qid": int(seed.query_id), "cid": seed.id, "topk": topk})
    out = []
    for r in rows:
        node = r["n"]
        p = store._node_to_passage(node, 0.0, "graph", EXPANSION_NAMESPACE)
        p.score = max(0.0, seed.score * 0.9)  # slightly below the seed
        out.append(p)
    return out