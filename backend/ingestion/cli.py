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
        counts = await index_sample(queries, store, cfg, namespaces=args.namespaces, limit_passages=args.limit_passages)
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