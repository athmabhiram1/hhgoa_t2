"""Concurrent load test (Phase 5) — P100 under saturation.

Runs the SAME mixed-language query set at higher concurrency than the latency
benchmark and reports P50/P70/P100 for cold (uncached, embed-queued) and warm
(cache-hit) traffic plus throughput (queries/sec). This is what keeps the P100
claim defensible: the worst-case tail is measured under load, not guessed.

Writes benchmarks/loadtest_<ts>.json + .md

Run with: python -m backend.harness.loadtest
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import statistics
import time
from pathlib import Path

from ..config import Settings, get_settings
from ..ingestion.dataset import load_sample
from ..retrieval.embeddings import EmbeddingService, get_embedding_service, set_embedding_service
from ..retrieval.neo4j_store import Neo4jStore
from .benchmark import BenchmarkRunner, _mixed_queries, percentile
from .pipeline import VakRagPipeline

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
OUTDIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"


def _pcts(values: list[float]) -> dict:
    return {
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p100": round(percentile(values, 100), 2),
        "mean": round(statistics.mean(values), 2) if values else 0.0,
        "n": len(values),
    }


async def main(cfg: Settings) -> None:
    queries_raw = load_sample(SAMPLE_DIR, cfg.langs)
    queries = _mixed_queries(queries_raw, cfg.benchmark_n_queries)

    set_embedding_service(EmbeddingService(cfg))
    get_embedding_service().warm()

    store = Neo4jStore(cfg)
    pipeline = VakRagPipeline(cfg, client=None)
    pipeline.bind_retrieval(store)
    runner = BenchmarkRunner(cfg, pipeline)

    report = {"run_ts": time.strftime("%Y%m%d-%H%M%S")}
    try:
        # Cold under load: each query is distinct -> embed requests queue on
        # Ollama's serial endpoint. Concurrency = loadtest_concurrency.
        cold = await runner.run(queries, concurrency=cfg.loadtest_concurrency)
        report["cold"] = cold
        report["cold"]["stats"] = runner.stats(cold["results"])
        report["cold"]["throughput_qps"] = round(len(queries) / cold["wall_seconds"], 2)

        # Warm under load: the SAME queries, now all cache hits -> measures the
        # cache-served tail at full concurrency (the demo repeat-query flow).
        warm = await runner.run(queries, concurrency=cfg.loadtest_concurrency)
        report["warm"] = warm
        report["warm"]["stats"] = runner.stats(warm["results"])
        report["warm"]["throughput_qps"] = round(len(queries) / warm["wall_seconds"], 2)
    finally:
        await store.close()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"loadtest_{report['run_ts']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# VakRAG concurrent load test", "",
          f"run: {report['run_ts']}  queries: {len(queries)}  concurrency: {cfg.loadtest_concurrency}", "",
          "| pass | p50 (ms) | p70 (ms) | p100 (ms) | mean (ms) | n | throughput (qps) |",
          "|---|---|---|---|---|---|---|"]
    for name in ["cold", "warm"]:
        s = report[name]["stats"]
        tot = s["total_ms"]
        md.append(f"| {name} | {tot['p50']} | {tot['p70']} | {tot['p100']} | {tot['mean']} | {tot['n']} | {report[name]['throughput_qps']} |")
    md += ["", "## per-pass detail", ""]
    for name in ["cold", "warm"]:
        s = report[name]["stats"]
        md += [f"### {name} pass", "", f"answer rate: {s['answer_rate']}  cache hit rate: {s['cache_hit_rate']}", "",
               "| metric | p50 | p70 | p100 | mean | n |", "|---|---|---|---|---|---|"]
        for key, label in [("total_ms", "total"), ("retrieval_ms", "retrieval"), ("cache_hit_ms", "cache hit"),
                           ("cache_miss_ms", "cache miss")]:
            st = s.get(key) or {}
            if not st:
                continue
            md.append(f"| {label} | {st['p50']} | {st['p70']} | {st['p100']} | {st['mean']} | {st['n']} |")
        md.append("")
    (OUTDIR / f"loadtest_{report['run_ts']}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for name in ["cold", "warm"]:
        s = report[name]["stats"]["total_ms"]
        print(f"{name}: total p50={s['p50']} p70={s['p70']} p100={s['p100']} mean={s['mean']}  "
              f"qps={report[name]['throughput_qps']}  answer_rate={report[name]['stats']['answer_rate']}")
    print(f"\nwrote benchmarks/loadtest_{report['run_ts']}.json + .md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main(get_settings()))