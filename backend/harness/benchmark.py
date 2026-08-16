"""Latency benchmark harness.

Runs N queries from the saved sample through the full pipeline (transcript
mode, STT excluded — reported separately) and writes:

  benchmarks/benchmark_<ts>.json   — raw per-query results + stage percentiles
  benchmarks/benchmark_<ts>.md     — P50 / P70 / P100 + cache & warm/cold splits

Warm/cold: the first run populates the embedding LRU + retrieval cache (cold);
a second pass over the same queries measures warm + cache-hit latency.

Run with: python -m backend.harness.benchmark
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ..config import Settings, get_settings
from ..core.models import PipelineResult
from ..ingestion.dataset import load_sample
from ..retrieval.embeddings import EmbeddingService, get_embedding_service, set_embedding_service
from ..retrieval.neo4j_store import Neo4jStore
from .pipeline import VakRagPipeline

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (s[c] - s[f]) * (k - f)


def _pcts(values: list[float]) -> dict:
    return {
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p100": round(percentile(values, 100), 2),
        "mean": round(statistics.mean(values), 2) if values else 0.0,
        "n": len(values),
    }


class BenchmarkRunner:
    def __init__(self, cfg: Settings, pipeline: VakRagPipeline) -> None:
        self.cfg = cfg
        self.pipeline = pipeline

    async def run(self, queries: list[tuple[str, str | None]], *, concurrency: int = 8) -> dict:
        start = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)

        async def one(pair: tuple[str, str | None]) -> dict:
            q, lang = pair
            async with sem:
                t0 = time.perf_counter()
                result: PipelineResult = await self.pipeline.run_transcript(q, lang=lang, mode="extractive")
                stages = {s.name: s.duration_ms for s in result.spans}
                return {
                    "query": q,
                    "total_ms": result.total_ms,
                    "answer_latency_ms": result.answer.latency_ms,
                    "retrieval_ms": result.retrieval.latency_ms if result.retrieval else None,
                    "cache": (result.retrieval.breakdown_ms.get("cache") if result.retrieval else None),
                    "stages": stages,
                    "mode": result.answer.mode,
                    "refusal_reason": result.answer.refusal_reason,
                    "grounding_score": result.answer.grounding_score,
                    "answered": result.answer.mode != "refusal",
                }

        results = await asyncio.gather(*[one(p) for p in queries])
        wall = time.perf_counter() - start
        return {"results": results, "wall_seconds": wall}

    def stats(self, results: list[dict]) -> dict:
        totals = [r["total_ms"] for r in results]
        answered = [r for r in results if r["answered"]]
        retrieval_ms = [r["retrieval_ms"] for r in results if r["retrieval_ms"] is not None]
        cache_hits = [r["total_ms"] for r in results if r.get("cache") == "hit"]
        cache_miss = [r["total_ms"] for r in results if r.get("cache") == "miss"]
        stages: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for name, ms in r["stages"].items():
                stages[name].append(ms)
        return {
            "n": len(results),
            "answered": len(answered),
            "answer_rate": round(len(answered) / len(results), 3) if results else 0.0,
            "total_ms": _pcts(totals),
            "answered_total_ms": _pcts([r["total_ms"] for r in answered]),
            "retrieval_ms": _pcts(retrieval_ms),
            "cache_hit_ms": _pcts(cache_hits),
            "cache_miss_ms": _pcts(cache_miss),
            "cache_hit_rate": round(len(cache_hits) / len(results), 3) if results else 0.0,
            "stages": {name: _pcts(ms) for name, ms in stages.items()},
        }


def _mixed_queries(raw: list, n: int) -> list[tuple[str, str | None]]:
    """Pick n queries balanced across languages and query_types (the sample file
    order is per-language sorted, so a plain slice would be single-language)."""
    by_lang: dict[str, list] = collections.defaultdict(list)
    for q in raw:
        by_lang[q.lang].append(q)
    langs = sorted(by_lang)
    out: list = []
    per_lang = max(1, n // len(langs))
    for lang in langs:
        bucket = by_lang[lang]
        types = sorted({q.query_type for q in bucket})
        per_type = max(1, per_lang // len(types))
        taken = []
        for t in types:
            pool = [q for q in bucket if q.query_type == t]
            taken.extend(pool[:per_type])
        if len(taken) < per_lang:
            rest = [q for q in bucket if q not in taken]
            taken.extend(rest[: per_lang - len(taken)])
        out.extend((q.query, q.lang) for q in taken[:per_lang])
    return out[:n]


async def _main(cfg: Settings) -> None:
    queries_raw = load_sample(SAMPLE_DIR, cfg.langs)
    if not queries_raw:
        logger.error("No sample at %s — run `python -m backend.ingestion.cli --sample` first", SAMPLE_DIR)
        return
    queries = _mixed_queries(queries_raw, cfg.benchmark_n_queries)
    langs = collections.Counter(lang for _, lang in queries)
    logger.info("benchmark queries: n=%d langs=%s", len(queries), dict(langs))

    set_embedding_service(EmbeddingService(cfg))
    get_embedding_service().warm()

    store = Neo4jStore(cfg)
    pipeline = VakRagPipeline(cfg, client=None)  # extractive mode, no external LLM needed
    pipeline.bind_retrieval(store)
    try:
        report = {}
        report["cold"] = await BenchmarkRunner(cfg, pipeline).run(queries, concurrency=cfg.benchmark_concurrency)
        if cfg.benchmark_warm:
            report["warm"] = await BenchmarkRunner(cfg, pipeline).run(queries, concurrency=cfg.benchmark_concurrency)
    finally:
        await store.close()

    stats = {name: BenchmarkRunner(cfg, pipeline).stats(run["results"]) for name, run in report.items()}
    combined = [r for run in report.values() for r in run["results"]]
    stats["all"] = BenchmarkRunner(cfg, pipeline).stats(combined)

    outdir = Path(cfg.benchmark_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    full = {
        "run_ts": ts,
        "config": {"n_queries": len(queries), "concurrency": cfg.benchmark_concurrency, "warm_pass": cfg.benchmark_warm},
        "stats": stats,
        "results": combined,
    }
    (outdir / f"benchmark_{ts}.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    md = ["# VakRAG latency benchmark", "", f"run: {ts}  queries: {len(queries)}  concurrency: {cfg.benchmark_concurrency}",
          "", "## all passes (P50/P70/P100, ms)", "", "| metric | p50 | p70 | p100 | mean | n |", "|---|---|---|---|---|---|"]
    for key, label in [("total_ms", "total"), ("answered_total_ms", "answered only"), ("retrieval_ms", "retrieval"),
                       ("cache_hit_ms", "cache hit"), ("cache_miss_ms", "cache miss")]:
        s = stats["all"].get(key) or {}
        if not s:
            continue
        md.append(f"| {label} | {s['p50']} | {s['p70']} | {s['p100']} | {s['mean']} | {s['n']} |")
    md.append("")
    for name in ["cold", "warm"]:
        s = stats.get(name) or {}
        if not s:
            continue
        md += [f"## {name} pass", "", "| metric | p50 | p70 | p100 | mean | n |", "|---|---|---|---|---|---|"]
        for key, label in [("total_ms", "total"), ("retrieval_ms", "retrieval"), ("cache_hit_ms", "cache hit"),
                           ("cache_miss_ms", "cache miss")]:
            st = s.get(key) or {}
            if not st:
                continue
            md.append(f"| {label} | {st['p50']} | {st['p70']} | {st['p100']} | {st['mean']} | {st['n']} |")
        md += ["", f"cache hit rate: {s.get('cache_hit_rate', 0.0)}  ·  answer rate: {s.get('answer_rate', 0.0)}", ""]
    (outdir / f"benchmark_{ts}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(stats["all"], indent=2))
    print(f"\nwrote benchmarks/benchmark_{ts}.json + .md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(_main(get_settings()))