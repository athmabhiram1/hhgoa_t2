"""Offline retrieval eval — the three locked metrics.

Metric 1 — in_index_mrr (PRIMARY / headline):
    For every INDEXED query (from data/sample) whose is_selected passage is in
    the passage_natural pool, embed the query and retrieve over the full
    passage_natural pool, scoring MRR@10 / Recall@10 / nDCG@10. Legitimate
    because passage_natural nodes carry no query text — the pool does not reveal
    which passage is the gold. Structurally analogous to the paper's Table 2
    (query paired with a known gold, retrieved from a pool).

Metric 2 — query_anchored_mrr (LEAKY, reported separately):
    Same computation over the query_anchored namespace. That namespace embeds
    query+passage together, so the score is inflated by construction. It is
    NEVER presented next to in_index_mrr without the leakage caveat attached
    in both the JSON and the Markdown report.

Metric 3 — held_out_mrr (SECONDARY, small-N, genuinely disjoint):
    The holdout queries whose gold passage text matches one of the recurring
    passages (present in >= 2 indexed queries). These are true unseen queries
    with gold-in-pool = True by construction. This is the ONLY number in the
    whole report that is a genuine held-out eval; N is expected to be small
    (0 in the current data — the recurring-passage set is ~342 texts and no
    holdout gold matched them). If N == 0, the metric is reported as N=0 and
    [not measured], never filled with an estimate.

Run with: python -m backend.harness.eval_mrr
Writes:  eval/mrr_<ts>.json + eval/mrr_<ts>.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ..config import get_settings
from ..ingestion.dataset import LANG_NAMES, QueryRecord, load_sample
from ..retrieval.embeddings import EmbeddingService, get_embedding_service, query_embedding, set_embedding_service
from ..retrieval.fusion import dedupe, rrf_fuse
from ..retrieval.neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
HOLDOUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "holdout"
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"

# Paper Table 2 (IndicRAGSuite) bge-m3 MRR@10 baselines, where published.
# Sanskrit has NO baseline row in the paper's Table 2 — it appears only in the
# paper's training-data table. The leaderboard shows "no published baseline".
PAPER_MRR = {
    "hi": 0.52, "bn": 0.49, "ta": 0.49, "te": 0.50, "or": 0.45, "as": 0.46,
}
NOT_MEASURED = "[not measured]"


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _tokens(s: str) -> set[str]:
    return set(_norm(s).split())


def gold_texts(q: QueryRecord) -> list[str]:
    return [p.text for p in q.passages if p.is_selected]


def is_hit(chunk_text: str, golds: list[str]) -> bool:
    """A chunk hits when any gold passage text is contained in it (normalized)
    or the gold's token set overlaps the chunk's by >= 0.7 (chunked passages)."""
    cn = _norm(chunk_text)
    for g in golds:
        gn = _norm(g)
        if not gn:
            continue
        if gn in cn or cn in gn:
            return True
        gs = _tokens(g)
        if gs and len(gs & set(cn.split())) / len(gs) >= 0.7:
            return True
    return False


def mrr_ndcg(ranks: list[int], k: int) -> tuple[float, float, float]:
    mr = 1.0 / min(ranks) if ranks else 0.0
    rec = 1.0 if ranks else 0.0
    if not ranks:
        return mr, rec, 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ranks), k) + 1))
    return mr, rec, dcg / idcg


def recurring_passages(sample: list[QueryRecord]) -> dict[str, list[int]]:
    """Passage texts present in >= 2 indexed queries -> {norm_text: [query_ids]}."""
    text_to_queries: dict[str, set[int]] = defaultdict(set)
    for q in sample:
        for p in q.passages:
            text_to_queries[_norm(p.text)].add(q.query_id)
    return {t: sorted(qs) for t, qs in text_to_queries.items() if len(qs) >= 2}


def held_out_eligible(holdout: list[QueryRecord], recurring: dict[str, list[int]]) -> list[QueryRecord]:
    """Holdout queries whose gold passage text matches a recurring passage."""
    out = []
    for q in holdout:
        for g in gold_texts(q):
            if _norm(g) in recurring:
                out.append(q)
                break
    return out


class EvalRunner:
    def __init__(self, cfg, store: Neo4jStore, *, topk: int = 10) -> None:
        self.cfg = cfg
        self.store = store
        self.topk = topk
        self._cache: dict[tuple, list] = {}

    async def _search(self, query: str, namespace: str, lang: str | None) -> list:
        key = (namespace, query, lang or "")
        if key not in self._cache:
            vec = query_embedding(query)
            vk, bk = self.cfg.retrieval_vector_k, self.cfg.retrieval_bm25_k
            vector = await self.store.vector_search(namespace, vec, vk, lang=lang)
            bm25 = await self.store.bm25_search(namespace, query, bk, lang=lang)
            self._cache[key] = dedupe(rrf_fuse([vector, bm25], k=self.cfg.retrieval_rrf_k, topk=self.cfg.retrieval_fusion_topk), by_lang=True)
        return self._cache[key][: self.topk]

    async def eval_queries(self, queries: list[QueryRecord], namespace: str) -> dict:
        """MRR/Recall/nDCG over one namespace. Gold = query's is_selected passage.
        Retrieval is over the FULL pool of that namespace (no language filter),
        matching the spec for in_index_mrr.

        NOTE: there is deliberately NO "coverage" field here. gold_found counts
        top-10 hits, which is exactly Recall@10 restated — naming it "coverage"
        conflated it with pool-membership (does the gold exist anywhere in the
        pool, ~1.0 by construction). Only held_out_mrr carries a coverage field,
        and there it means gold-in-pool by recurrence construction."""
        by_lang: dict[str, list[list[int]]] = {}
        gold_found = 0
        for q in queries:
            golds = gold_texts(q)
            if not golds:
                continue
            cands = await self._search(q.query, namespace, None)
            ranks = [i + 1 for i, c in enumerate(cands) if is_hit(c.text, golds)]
            if ranks:
                gold_found += 1
            by_lang.setdefault(q.lang, []).append(ranks)
        n_queried = sum(len(v) for v in by_lang.values())
        all_ranks = [r for ranks in by_lang.values() for r in ranks]
        return {
            "namespace": namespace,
            "leaky": namespace == "query_anchored",
            "n": n_queried,
            "gold_found_top10": gold_found,
            "overall": self._stats(all_ranks),
            "by_lang": {lang: self._stats(ranks) for lang, ranks in by_lang.items()},
        }

    def _stats(self, lists: list[list[int]]) -> dict:
        if not lists:
            return {"n": 0, "mrr10": 0.0, "recall10": 0.0, "ndcg10": 0.0}
        mrr = [mrr_ndcg(r, self.topk)[0] for r in lists]
        rec = [mrr_ndcg(r, self.topk)[1] for r in lists]
        ndcg = [mrr_ndcg(r, self.topk)[2] for r in lists]
        return {
            "n": len(lists),
            "mrr10": round(statistics.mean(mrr), 4),
            "recall10": round(statistics.mean(rec), 4),
            "ndcg10": round(statistics.mean(ndcg), 4),
        }


async def _main(args: argparse.Namespace) -> None:
    cfg = get_settings()
    store = Neo4jStore(cfg)
    await store.ensure_schema()
    set_embedding_service(EmbeddingService(cfg))
    get_embedding_service().warm()

    sample = load_sample(SAMPLE_DIR, args.langs or None)
    holdout = load_sample(HOLDOUT_DIR, args.langs or None)
    if args.limit:
        sample = sample[: args.limit]

    recurring = recurring_passages(sample)
    eligible = held_out_eligible(holdout, recurring)

    report: dict[str, object] = {
        "run_ts": time.strftime("%Y%m%d-%H%M%S"),
        "eval_pool": args.namespace,
        "topk": args.topk,
        "n_index_queries": len(sample),
        "n_holdout": len(holdout),
        "n_recurring_passages": len(recurring),
        "n_held_out_eligible": len(eligible),
        "metrics": {},
    }

    if args.namespace == "passage_natural":
        runner = EvalRunner(cfg, store, topk=args.topk)
        report["metrics"]["in_index_mrr"] = await runner.eval_queries(sample, "passage_natural")
    elif args.namespace == "query_anchored":
        runner = EvalRunner(cfg, store, topk=args.topk)
        report["metrics"]["query_anchored_mrr"] = await runner.eval_queries(sample, "query_anchored")
    elif args.namespace == "all":
        runner = EvalRunner(cfg, store, topk=args.topk)
        report["metrics"]["in_index_mrr"] = await runner.eval_queries(sample, "passage_natural")
        report["metrics"]["query_anchored_mrr"] = await runner.eval_queries(sample, "query_anchored")

    qa = report["metrics"].get("query_anchored_mrr")
    if qa is not None:
        qa["note"] = ("POOL-SIZE MISMATCH: the query_anchored namespace holds ONLY the curated "
                      "subset (28,518 chunks; 204 distinct query_ids per language), NOT the full "
                      "sample pool (213,928 chunks; 1,530 distinct query_ids). Gold passages from "
                      "the full sample are largely absent from this smaller pool, so scores here are "
                      "both leaky (query+passage embedded together) AND measured against a "
                      "non-representative pool. Do not compare with in_index_mrr.")

    # held_out_mrr — genuinely disjoint, gold-in-pool by construction.
    if args.namespace in ("all", "passage_natural"):
        held: dict = {"namespace": "passage_natural", "leaky": False, "n": len(eligible),
                      "gold_in_pool": len(eligible), "coverage": 1.0 if eligible else 0.0,
                      "overall": None, "by_lang": {}, "note": "true unseen-query eval; gold-in-pool by recurrence construction"}
        if eligible:
            runner = EvalRunner(cfg, store, topk=args.topk)
            held = await runner.eval_queries(eligible, "passage_natural")
            held["note"] = "true unseen-query eval; gold-in-pool by recurrence construction"
            held["coverage"] = 1.0
            held["gold_in_pool"] = len(eligible)
        else:
            held["overall"] = {"n": 0, "mrr10": NOT_MEASURED, "recall10": NOT_MEASURED, "ndcg10": NOT_MEASURED}
            held["mrr10"] = NOT_MEASURED
            held["recall10"] = NOT_MEASURED
            held["ndcg10"] = NOT_MEASURED
        report["metrics"]["held_out_mrr"] = held

    await store.close()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"mrr_{report['run_ts']}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# VakRAG retrieval eval (locked three metrics)",
          "", f"run: {report['run_ts']}  pool: {report['eval_pool']}  topk: {report['topk']}",
          f"index queries: {report['n_index_queries']}  holdout: {report['n_holdout']}  "
          f"recurring passages: {report['n_recurring_passages']}  held-out eligible: {report['n_held_out_eligible']}",
          ""]
    for name in ["in_index_mrr", "query_anchored_mrr", "held_out_mrr"]:
        m = report["metrics"].get(name)
        if m is None:
            continue
        md += [f"## {name}", ""]
        if m.get("leaky"):
            md += ["> **LEAKY — do not compare with in_index_mrr.** This namespace embeds query+passage "
                   "together, so retrieval scores are inflated by construction.", ""]
        if m.get("note"):
            md += [f"> {m['note']}", ""]
        if "coverage" in m:
            md += [f"coverage: {m['coverage']}  (gold-in-pool {m['gold_in_pool']}/{m['n']})", ""]
        md += ["| lang | MRR@10 | Recall@10 | nDCG@10 | n | paper MRR |", "|---|---|---|---|---|---|"]
        for lang in sorted(m.get("by_lang", {})):
            s = m["by_lang"][lang]
            paper = PAPER_MRR.get(lang, "no published baseline" if lang == "sa" else "—")
            md.append(f"| {LANG_NAMES.get(lang, lang)} | {s['mrr10']} | {s['recall10']} | {s['ndcg10']} | {s['n']} | {paper} |")
        ov = m.get("overall") or {}
        if ov and ov.get("n", 0) > 0:
            tail = f" · coverage {m['coverage']}" if "coverage" in m else ""
            md += ["", f"**overall:** MRR@10 {ov.get('mrr10', NOT_MEASURED)} · Recall@10 {ov.get('recall10', NOT_MEASURED)} · nDCG@10 {ov.get('ndcg10', NOT_MEASURED)}{tail}", ""]
        else:
            md += ["", f"**overall:** N=0 — {NOT_MEASURED}", ""]
    md += ["> Paper column: IndicRAGSuite Table 2 (arXiv:2506.01615, bge-m3) where published. "
           "Sanskrit has no baseline row in Table 2. Our eval pool is machine-translated MSMARCO-XI "
           "(no human verification) — same metric, not the same benchmark as the paper's 1,000-query "
           "hand-verified IndicMSMarco.", ""]
    (EVAL_DIR / f"mrr_{report['run_ts']}.md").write_text("\n".join(md), encoding="utf-8")

    for name in ["in_index_mrr", "query_anchored_mrr", "held_out_mrr"]:
        m = report["metrics"].get(name)
        if m is None:
            continue
        ov = m.get("overall") or {}
        tag = " [LEAKY]" if m.get("leaky") else ""
        if ov and ov.get("n", 0) > 0:
            tail = f"  coverage {m['coverage']}" if "coverage" in m else ""
            print(f"{name}{tag}: MRR@10 {ov.get('mrr10', NOT_MEASURED)}  Recall@10 {ov.get('recall10', NOT_MEASURED)}  nDCG@10 {ov.get('ndcg10', NOT_MEASURED)}{tail} (N={m['n']})")
        else:
            tail = f"  coverage {m['coverage']}" if "coverage" in m else ""
            print(f"{name}{tag}: N=0 — {NOT_MEASURED}{tail} (N={m['n']})")
    print(f"\nwrote eval/mrr_{report['run_ts']}.json + .md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked three-metric retrieval eval")
    parser.add_argument("--namespace", choices=["passage_natural", "query_anchored", "all"], default="all",
                        help="Which eval pool to score (default: all)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Cap indexed queries (smoke run)")
    parser.add_argument("--langs", default=None, help="Comma list of languages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()