"""Batch indexer: chunks -> embeddings (Qwen3 via Ollama) -> Neo4j (vector + fulltext + graph).

Idempotent (content-stable chunk ids, MERGE upserts) and resumable (skip
namespaces already populated when `--skip-done`). Embeddings are computed in
batches; for large builds use the GPU (`EMBED_DEVICE=cuda`).
"""

from __future__ import annotations

import logging
import time

from ..config import Settings
from ..retrieval.embeddings import EmbeddingService, set_embedding_service
from ..retrieval.neo4j_store import Neo4jStore
from .chunking import NAMESPACES, chunk_queries
from .dataset import QueryRecord, load_sample

logger = logging.getLogger(__name__)


async def index_sample(
    queries: list[QueryRecord],
    store: Neo4jStore,
    cfg: Settings,
    *,
    namespaces: list[str] | None = None,
    skip_done: bool = True,
    limit_passages: int | None = None,
) -> dict[str, int]:
    embed_svc = EmbeddingService(cfg)
    set_embedding_service(embed_svc)
    await store.ensure_schema()

    # Warm the embed runner before the main loop: Ollama's Windows runner can
    # drop its tokenize endpoint during cold model load (intermittent 400s).
    embed_svc.embed_one("warmup")

    enabled = [ns for ns in (namespaces or NAMESPACES) if ns in NAMESPACES]
    chunks = chunk_queries(queries)
    if limit_passages:
        # Cheap fast-iteration mode: restrict to the first N passages per query.
        kept: list[QueryRecord] = []
        for q in queries:
            q.passages = q.passages[:limit_passages]
            kept.append(q)
        chunks = chunk_queries(kept, strategies=enabled)

    counts: dict[str, int] = {}
    start = time.perf_counter()

    # --- chunk namespaces -------------------------------------------------
    for ns in enabled:
        ns_chunks = [c for c in chunks if c.namespace == ns]
        if skip_done and await store.count_chunks(ns) >= len(ns_chunks):
            counts[ns] = await store.count_chunks(ns)
            logger.info("Skipping %s (already %d chunks)", ns, counts[ns])
            continue
        rows: list[dict] = []
        batch = 0
        for i in range(0, len(ns_chunks), cfg.index_batch_size):
            batch_chunks = ns_chunks[i : i + cfg.index_batch_size]
            # Resume-safe: skip embedding + MERGE for chunks already stored in a
            # previous partial run instead of re-embedding them from zero.
            existing = await store.existing_chunk_ids(ns, [c.chunk_id for c in batch_chunks])
            pending = [c for c in batch_chunks if c.chunk_id not in existing]
            if existing:
                logger.info(
                    "  %s: batch %d — %d/%d chunks already stored, skipping",
                    ns, batch + 1, len(existing), len(batch_chunks),
                )
            if not pending:
                batch += 1
                continue
            embeddings = embed_svc.embed_batch([c.text for c in pending])
            for c, vec in zip(pending, embeddings):
                rows.append(
                    {
                        "chunk_id": c.chunk_id,
                        "namespace": c.namespace,
                        "text": c.text,
                        "lang": c.lang,
                        "query_id": int(c.query_id),
                        "query_type": c.query_type,
                        "position": c.position,
                        "is_selected": c.is_selected,
                        "passage_pos": c.passage_pos,
                        "doc_key": c.doc_key,
                        "embedding": vec,
                    }
                )
            await store.upsert_chunks(rows, ns)
            rows.clear()
            batch += 1
            if batch % 5 == 0:
                logger.info(
                    "  %s: %d/%d chunks embedded",
                    ns, min((i + cfg.index_batch_size), len(ns_chunks)), len(ns_chunks),
                )
        stored = await store.count_chunks(ns)
        counts[ns] = stored
        generated = len(ns_chunks)
        if stored != generated:
            logger.info(
                "Indexed %s: %d stored (of %d generated — %d dedup collisions) in %.1fs",
                ns, stored, generated, generated - stored, time.perf_counter() - start,
            )
        else:
            logger.info("Indexed %s: %d chunks in %.1fs", ns, stored, time.perf_counter() - start)

    # --- query graph (once) -----------------------------------------------
    graph_ns = enabled[0] if enabled else "passage_natural"
    chunk_ids = [{"query_id": int(c.query_id), "chunk_id": c.chunk_id} for c in chunks if c.namespace == graph_ns]
    await store.upsert_query_graph(
        [{"query_id": q.query_id, "query": q.query, "answer": q.answer, "query_type": q.query_type, "lang": q.lang} for q in queries],
        graph_ns,
        chunk_ids,
    )
    counts["queries"] = await store.count_queries()
    counts["graph_chunks"] = await store.count_has_passage(graph_ns)
    logger.info("Query graph: %d queries, %d chunks linked", counts["queries"], counts["graph_chunks"])
    logger.info("Indexing complete in %.1fs — %s", time.perf_counter() - start, counts)
    return counts