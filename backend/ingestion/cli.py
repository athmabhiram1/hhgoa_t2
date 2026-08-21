"""CLI for sampling + indexing the dataset.

Examples:
  python -m backend.ingestion.cli --sample                 # stream & save sample JSONL
  python -m backend.ingestion.cli --sample --holdout        # also reserve eval holdout (never indexed)
  python -m backend.ingestion.cli --index                   # index from saved sample
  python -m backend.ingestion.cli --sample --index --limit-passages 3   # fast iteration
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from ..config import get_settings
from .dataset import StratifiedSampler, load_sample, save_sample
from .indexer import index_sample
from ..retrieval.neo4j_store import Neo4jStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
HOLDOUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "holdout"


async def _do_index(cfg, args) -> None:
    queries = load_sample(SAMPLE_DIR, cfg.langs)
    if not queries:
        logger.error("No sample found at %s — run --sample first", SAMPLE_DIR)
        return
    if args.queries_per_lang:
        kept: list[QueryRecord] = []
        by_lang: dict[str, int] = {}
        for q in queries:
            if by_lang.get(q.lang, 0) < args.queries_per_lang:
                kept.append(q)
                by_lang[q.lang] = by_lang.get(q.lang, 0) + 1
        logger.info(
            "Curated subset: %d queries (per-lang cap %d) from full sample %d",
            len(kept),
            args.queries_per_lang,
            len(queries),
        )
        queries = kept
    store = Neo4jStore(cfg)
    try:
        if args.progressive:
            from .indexer import progressive_index
            report = await progressive_index(
                queries, store, cfg,
                gate_batch=args.gate_batch or cfg.index_gate_batch,
                gate_threshold=args.gate_threshold if args.gate_threshold is not None else cfg.index_gate_threshold,
                namespaces=args.namespaces,
                force=args.force,
                limit_passages=args.limit_passages,
            )
            passed = bool(report.get("gate", {}).get("passed"))
            logger.info("Progressive index gate %s: Recall@10 %.3f vs %.3f (MRR@10 %.3f, nDCG@10 %.3f, pilot %d queries)",
                        "PASSED" if passed else "FAILED",
                        report.get("recall10", 0.0),
                        report.get("gate_threshold", 0.0),
                        report.get("mrr10", 0.0),
                        report.get("ndcg10", 0.0),
                        report.get("pilot_queries", 0))
            if passed:
                logger.info("Index stats: %s", report.get("final_counts", {}))
            else:
                logger.error("GATE FAILED — index stopped. See eval/gate_*.json and the improvement hints above.")
                raise SystemExit(2)
            return
        counts = await index_sample(queries, store, cfg, namespaces=args.namespaces, force=args.force, limit_passages=args.limit_passages)
        logger.info("Index stats: %s", counts)
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VakRAG ingestion")
    parser.add_argument("--sample", action="store_true", help="Stream dataset and save a stratified sample")
    parser.add_argument("--holdout", action="store_true", help="Reserve an eval holdout split (never indexed)")
    parser.add_argument("--holdout-size", type=int, default=None, help="Holdout queries per language (default: cfg.dataset_holdout_per_lang)")
    parser.add_argument("--index", action="store_true", help="Build Neo4j index from the saved sample")
    parser.add_argument("--limit-passages", type=int, default=None, help="Cap passages per query (fast iteration)")
    parser.add_argument("--namespaces", nargs="*", default=None, help="Chunk namespaces to index (default: all six)")
    parser.add_argument("--queries-per-lang", type=int, default=None, help="Curated deploy subset: cap queries per language")
    parser.add_argument("--progressive", action="store_true", help="Index a small pilot, eval Recall@10 vs gold, only continue if the gate passes")
    parser.add_argument("--gate-batch", type=int, default=None, help="Pilot queries per language for the progressive gate (default: cfg.index_gate_batch)")
    parser.add_argument("--gate-threshold", type=float, default=None, help="Recall@10 floor to continue the full index (default: cfg.index_gate_threshold)")
    parser.add_argument("--force", action="store_true", help="Re-embed everything, overwriting existing vectors (e.g. after a backend switch)")
    args = parser.parse_args()

    cfg = get_settings()
    if args.sample:
        holdout_size = args.holdout_size if args.holdout else 0
        if args.holdout and args.holdout_size is None:
            holdout_size = cfg.dataset_holdout_per_lang
        sampler = StratifiedSampler(cfg)
        queries, holdout = sampler.run(holdout_per_lang=holdout_size)
        save_sample(queries, SAMPLE_DIR)
        if holdout:
            save_sample(holdout, HOLDOUT_DIR)
            logger.info("Holdout saved to %s (%d queries)", HOLDOUT_DIR, len(holdout))
    if args.index:
        asyncio.run(_do_index(cfg, args))
    if not args.sample and not args.index:
        parser.print_help()


if __name__ == "__main__":
    main()