"""Fast-path latency benchmark — local ANN, no network.

Loads 70 queries (5 per lang) from the Pool A gold set and measures
local embed+search latency end-to-end. Reports P50/P70/P100 for the
fast path (the 200ms-compliant extractive output) and per-stage breakdown.

Usage:
  python -m backend.harness.fastpath_bench --n-per-lang 5
  python -m backend.harness.fastpath_bench --queries 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from ..retrieval.local_index import LocalFastIndex


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-lang", type=int, default=5, help="queries per language (14 langs)")
    ap.add_argument("--queries", type=int, default=0, help="override total queries (0 = use n-per-lang)")
    ap.add_argument("--pool", type=str, default="eval/mrr_unbiased_pool_a_20260819-205521.json")
    ap.add_argument("--outdir", type=str, default="eval")
    args = ap.parse_args()

    cfg = get_settings()
    # load pool
    pool_path = Path(args.pool)
    if not pool_path.exists():
        # fallback to any pool file
        cands = sorted(Path("eval").glob("mrr_unbiased_pool_a_*.json"))
        if not cands:
            print("no pool file found", file=sys.stderr)
            return 2
        pool_path = cands[-1]
    with pool_path.open(encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results") or []
    if not results:
        print("pool has no results", file=sys.stderr)
        return 2

    # group by lang
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_lang[r.get("lang","")].append(r)

    if args.queries and args.queries > 0:
        # flat sample across langs round-robin
        total = args.queries
        per_lang = max(1, total // len(by_lang))
        args.n_per_lang = per_lang

    queries: list[dict] = []
    for lang in sorted(by_lang):
        # prefer hits (grounded queries) for meaningful latency
        hits = [r for r in by_lang[lang] if r.get("hit")]
        pool = hits if len(hits) >= args.n_per_lang else by_lang[lang]
        queries.extend(pool[: args.n_per_lang])

    idx = LocalFastIndex(cfg)
    if not idx.load():
        print(f"fast-path index not found at {cfg.fast_path_index_dir} — run python -m backend.harness.build_fastpath first", file=sys.stderr)
        return 2

    # warm the model (first encode loads weights; exclude from p50)
    idx.search("warmup query for bge-m3 on cuda")

    latencies: list[float] = []
    per_lang_lat: dict[str, list[float]] = defaultdict(list)

    print(f"fast-path bench: {len(queries)} queries ({args.n_per_lang} per lang, {len(by_lang)} langs)", flush=True)
    for row in queries:
        q = row["query"]
        lang = row.get("lang","")
        t0 = time.perf_counter()
        res = idx.search(q)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        per_lang_lat[lang].append(dt)

    latencies_sorted = sorted(latencies)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pool": str(pool_path),
        "n_queries": len(queries),
        "n_per_lang": args.n_per_lang,
        "langs": sorted(per_lang_lat.keys()),
        "fast_path": {
            "p50": round(_pct(latencies, 50), 2),
            "p70": round(_pct(latencies, 70), 2),
            "p100": round(_pct(latencies, 100), 2),
            "mean": round(statistics.mean(latencies), 2),
            "n": len(latencies),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "per_lang_p50": {lang: round(_pct(v, 50), 2) for lang, v in per_lang_lat.items()},
        "config": {
            "model": cfg.fast_path_model,
            "device": cfg.fast_path_device,
            "topk": cfg.fast_path_topk,
            "index_size": idx.size,
        },
        "all_latencies_ms": [round(x, 2) for x in latencies_sorted],
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = outdir / f"fastpath_latency_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfast-path latency (local, no network, {len(queries)} queries):")
    print(f"  P50  {report['fast_path']['p50']} ms")
    print(f"  P70  {report['fast_path']['p70']} ms")
    print(f"  P100 {report['fast_path']['p100']} ms  (mean {report['fast_path']['mean']} ms)")
    print(f"  per-lang P50: {json.dumps(report['per_lang_p50'], ensure_ascii=False)}")
    print(f"  report: {out_path}")

    # also emit a markdown table for README
    md_path = outdir / f"fastpath_latency_{ts}.md"
    md_path.write_text(
        f"# Fast-path latency — {report['ts']}\n\n"
        f"Local bge-m3 ({cfg.fast_path_device}) + brute-force cosine over {idx.size} chunks. No Vertex, no Neo4j.\n\n"
        f"| metric | ms |\n|---|---|\n"
        f"| P50 | {report['fast_path']['p50']} |\n"
        f"| P70 | {report['fast_path']['p70']} |\n"
        f"| P100 | {report['fast_path']['p100']} |\n"
        f"| mean | {report['fast_path']['mean']} |\n"
        f"| n | {len(queries)} |\n"
        + "\nPer-lang P50 (ms):\n\n| lang | P50 |\n|---|---|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in sorted(report["per_lang_p50"].items()))
        + "\n",
        encoding="utf-8",
    )

    # pass/fail gate: P50 must be <= 200ms for the claim
    if report["fast_path"]["p50"] > 200:
        print(f"FAIL: P50 {report['fast_path']['p50']}ms exceeds 200ms budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
