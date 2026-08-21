"""Batch indexer: chunks -> embeddings (Qwen3 via Ollama) -> Neo4j (vector + fulltext + graph).

Idempotent (content-stable chunk ids, MERGE upserts) and resumable (skip
namespaces already populated when `--skip-done`). Embeddings are computed in
batches; for large builds use the GPU (`EMBED_DEVICE=cuda`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from ..config import Settings
from ..retrieval.embeddings import EmbeddingService, set_embedding_service
from ..retrieval.neo4j_store import Neo4jStore
from .chunking import NAMESPACES, chunk_queries
from .dataset import QueryRecord, load_sample
from ..harness.eval_mrr import gold_texts, is_hit

logger = logging.getLogger(__name__)


async def index_sample(
    queries: list[QueryRecord],
    store: Neo4jStore,
    cfg: Settings,
    *,
    namespaces: list[str] | None = None,
    skip_done: bool = True,
    force: bool = False,
    limit_passages: int | None = None,
) -> dict[str, int]:
    embed_svc = EmbeddingService(cfg)
    set_embedding_service(embed_svc)
    await store.ensure_schema()

    # Warm the embed runner before the main loop: Ollama's Windows runner can
    # drop its tokenize endpoint during cold model load (intermittent 400s).
    embed_svc.embed_one("warmup")
    backend_name = embed_svc.backend_name
    if backend_name != cfg.embed_backend:
        raise RuntimeError(
            f"embedding backend mismatch: service resolved {backend_name!r} "
            f"but config requires {cfg.embed_backend!r} — refusing to index"
        )

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
    stall_batches = 0
    upsert_seconds = 0.0

    # --- chunk namespaces -------------------------------------------------
    for ns in enabled:
        ns_chunks = [c for c in chunks if c.namespace == ns]
        # passage_en generates one chunk per (query, lang) for the SAME English
        # text — ~14x duplicate ids that MERGE-dedup in Neo4j. Embedding each
        # unique id once is byte-identical DB state with 14x fewer API calls.
        seen: set[str] = set()
        unique: list = []
        for c in ns_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique.append(c)
        if len(unique) != len(ns_chunks):
            logger.info("  %s: deduped %d duplicate chunk ids before embedding", ns, len(ns_chunks) - len(unique))
        ns_chunks = unique
        # Namespace-level completion is EXACT: we ask Neo4j which of THIS job's
        # target chunk IDs already carry the current backend's vectors. A global
        # count (count_chunks with embed_backend) is NOT a valid completion
        # signal for subset jobs — e.g. a namespace with 22,624 Vertex chunks
        # and a target subset of 2,040 Bengali IDs would pass a
        # `global_vertex >= len(subset)` check even when those Bengali IDs are
        # NULL. Chunked by index_batch_size to keep IN-lists bounded.
        if skip_done and not force:
            target_ids = [c.chunk_id for c in ns_chunks]
            existing_ids: set[str] = set()
            for i in range(0, len(target_ids), cfg.index_batch_size):
                existing_ids |= await store.existing_chunk_ids(
                    ns, target_ids[i : i + cfg.index_batch_size], embed_backend=backend_name
                )
            if len(existing_ids) == len(target_ids):
                counts[ns] = len(existing_ids)
                logger.info("Skipping %s (all %d target %s chunks already stored)", ns, len(existing_ids), backend_name)
                continue
        rows: list[dict] = []
        batch = 0
        for i in range(0, len(ns_chunks), cfg.index_batch_size):
            batch_chunks = ns_chunks[i : i + cfg.index_batch_size]
            # Resume-safe: skip embedding + MERGE for chunks already stored in a
            # previous partial run instead of re-embedding them from zero.
            # `force` disables that so a full re-embed overwrites every vector
            # (upsert_chunks MERGE + SET replaces the embedding property).
            existing: set[str] = set()
            if not force:
                existing = await store.existing_chunk_ids(ns, [c.chunk_id for c in batch_chunks], embed_backend=backend_name)
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
                        "embed_backend": backend_name,
                    }
                )
            t0 = time.perf_counter()
            await store.upsert_chunks(rows, ns)
            ups = time.perf_counter() - t0
            upsert_seconds += ups
            if ups > 30.0:
                stall_batches += 1
                logger.warning("  %s: upsert batch took %.1fs — AuraDB stall burst (driver retry recovered)", ns, ups)
            if cfg.index_pace_s > 0:
                await asyncio.sleep(cfg.index_pace_s)
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

    counts["upsert_seconds"] = round(upsert_seconds, 1)
    counts["stall_batches"] = stall_batches
    logger.info(
        "Upsert timing: %.1fs total, %d stall batches (>30s) across %d namespaces",
        upsert_seconds, stall_batches, len(enabled),
    )

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


async def progressive_index(
    queries: list[QueryRecord],
    store: Neo4jStore,
    cfg: Settings,
    *,
    gate_batch: int,
    gate_threshold: float,
    namespaces: list[str] | None = None,
    force: bool = False,
    limit_passages: int | None = None,
) -> dict:
    """Index a small pilot subset, measure retrieval quality against the gold
    `is_selected` passages, and only continue the full build if the gate passes.

    Pattern (research-backed: staging index + golden-set gate + promote on
    pass — ranjankumar.in staging-index, newline.co "Why RAG Systems Fail at
    Scale", Weaviate retrieval-quality overview):
      1. Pick a pilot: `gate_batch` queries per language (deterministic).
      2. Index the pilot into the `passage_natural` namespace only.
      3. Run a mini eval over the just-indexed pilot pool (embed the pilot
         query, hybrid vector+BM25 + RRF, score whether the gold `is_selected`
         passage is hit in the top-10 — exactly `in_index_mrr`, but over the
         pilot pool so gold-in-pool is guaranteed by construction).
      4. Gate: if Recall@10 >= gate_threshold -> CONTINUE, index the rest
         (all requested namespaces). If below -> STOP; return a report the CLI
         turns into a non-zero exit and improvement hints (chunking strategy,
         embed backend, threshold), never silently burn hours of embed time.

    Returns a report dict; `report["gate"]["passed"]` decides continuation.
    The remaining (non-pilot) queries are indexed in-place by this function
    when the gate passes. The pilot's chunks are already stored, so the second
    `index_sample` pass skips them via existing-chunk-id resume.
    """
    embed_svc = EmbeddingService(cfg)
    set_embedding_service(embed_svc)
    await store.ensure_schema()

    # --- 1. deterministic pilot subset (first N queries per language) ------
    by_lang: dict[str, list[QueryRecord]] = {}
    for q in queries:
        by_lang.setdefault(q.lang, []).append(q)
    pilot: list[QueryRecord] = []
    for lang in sorted(by_lang):
        pilot.extend(by_lang[lang][:gate_batch])
    pilot_ids = {q.query_id for q in pilot}
    remaining = [q for q in queries if q.query_id not in pilot_ids]
    logger.info(
        "Progressive gate: pilot %d queries (%d/lang, %d languages) indexed first; %d remain",
        len(pilot), gate_batch, len(by_lang), len(remaining),
    )

    # --- 2. index only the pilot into passage_natural ----------------------
    pilot_counts = await index_sample(
        pilot, store, cfg,
        namespaces=["passage_natural"], force=force, limit_passages=limit_passages,
    )

    # --- 3. mini eval over the just-indexed pilot pool ----------------------
    from ..harness.eval_mrr import EvalRunner
    runner = EvalRunner(cfg, store, topk=10)
    stats = await runner.eval_queries(pilot, "passage_natural")
    overall = stats["overall"]
    recall = overall.get("recall10", 0.0)
    mrr = overall.get("mrr10", 0.0)
    ndcg = overall.get("ndcg10", 0.0)
    gold_hits = 0
    for q in pilot:
        if await _pilot_hit(runner, q):
            gold_hits += 1
    report = {
        "namespace": "passage_natural",
        "gate_batch": gate_batch,
        "gate_threshold": gate_threshold,
        "pilot_queries": len(pilot),
        "remaining_queries": len(remaining),
        "pilot_chunks": pilot_counts.get("passage_natural", 0),
        "recall10": round(recall, 4),
        "mrr10": round(mrr, 4),
        "ndcg10": round(ndcg, 4),
        "gold_hits": gold_hits,
    }
    report["gate"] = {"passed": recall >= gate_threshold, "reason": None}
    if not report["gate"]["passed"]:
        report["gate"]["reason"] = (
            f"Recall@10 {recall:.3f} < gate threshold {gate_threshold}. Stop and improve: "
            "re-check the embed backend (Ollama/Vertex up? dim 1024? index populated?), "
            "chunking strategy, RRF weights, or lower the gate only after a measured "
            "decision — never to silence a failing index."
        )
        logger.error("PROGRESSIVE GATE FAILED — %s", report["gate"]["reason"])
        _write_gate_report(report, cfg)
        return report

    logger.info("Progressive gate PASSED: Recall@10 %.3f (MRR@10 %.3f, nDCG@10 %.3f) — continuing with %d queries",
                recall, mrr, ndcg, len(remaining))

    # --- 4. gate passed -> index the rest (all requested namespaces) --------
    final = await index_sample(
        remaining, store, cfg,
        namespaces=namespaces, force=force, limit_passages=limit_passages,
    )
    report["final_counts"] = final
    _write_gate_report(report, cfg)
    return report


async def _pilot_hit(runner, q) -> bool:
    """Whether any gold `is_selected` passage of q is hit in the top-10 pilot pool."""
    golds = gold_texts(q)
    if not golds:
        return False
    cands = await runner._search(q.query, "passage_natural", None)
    return any(is_hit(c.text, golds) for c in cands[:10])


def _write_gate_report(report: dict, cfg: Settings) -> None:
    import time as _time
    from pathlib import Path

    evaldir = Path(__file__).resolve().parent.parent.parent / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    path = evaldir / f"gate_{_time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Gate report written to %s", path)