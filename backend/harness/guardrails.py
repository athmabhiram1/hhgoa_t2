"""Guardrail validation sets (Phase 5).

Runs three labeled sets through the pipeline and reports the refusal/answer
matrix:

  off_topic:    chit-chat / non-knowledge -> expect refusal (off_topic)
  low_grounding:on-topic but ungrounded  -> expect refusal (low_grounding)
  on_topic:     real sample queries       -> expect answer (not refusal)

Writes:
  benchmarks/guardrails_<ts>.json + .md

Run with: python -m backend.harness.guardrails
"""

from __future__ import annotations

import asyncio
import json
import logging
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
OUTDIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"

OFF_TOPIC_SET = [
    "hi",
    "hello",
    "how are you?",
    "kaise ho",
    "what is your opinion on cricket?",
    "tell me a joke",
    "who are you?",
    "good morning",
    "what do you think about pizza?",
    "thanks!",
]

LOW_GROUNDING_SET = [
    "What was the GDP of the fictional nation of Zorpia in 1987?",
    "Who won the 1967 annual pumpkin-carving championship in a village called Khenpur?",
    "What is the chemical formula of the recently discovered element kryptonite-9?",
    "Describe the mating ritual of the purple-spotted hornswoggle beetle found only on a private island?",
]


def _on_topic_sample(n: int = 5) -> list[str]:
    """Real sample queries whose is_selected gold is in the pool — the honest
    on-topic proxy. Generic English trivia (capital of india etc.) grounds at
    0.70-0.74 with qwen3 — below the recalibrated 0.78 threshold — because it
    is not actually in the index, so it is correctly refused as low_grounding."""
    raw = load_sample(SAMPLE_DIR, None)
    out: list[str] = []
    for q in raw:
        if not any(p.is_selected for p in q.passages):
            continue
        out.append(q.query)
        if len(out) >= n:
            break
    return out


async def main(cfg: Settings) -> None:
    set_embedding_service(EmbeddingService(cfg))
    get_embedding_service().warm()

    store = Neo4jStore(cfg)
    pipeline = VakRagPipeline(cfg, client=None)
    pipeline.bind_retrieval(store)
    try:
        results: dict[str, list[dict]] = {}

        on_topic = _on_topic_sample()
        for label, qs in [("off_topic", OFF_TOPIC_SET), ("on_topic", on_topic)]:
            results[label] = []
            for q in qs:
                r: PipelineResult = await pipeline.run_transcript(q, mode="extractive")
                results[label].append({
                    "query": q,
                    "mode": r.answer.mode,
                    "refusal_reason": r.answer.refusal_reason,
                    "grounding_score": r.answer.grounding_score,
                    "total_ms": r.total_ms,
                })

        # low_grounding: on-topic questions with NO supporting passage in the
        # index (fabricated/private facts). Grounding must fall below threshold.
        results["low_grounding"] = []
        for q in LOW_GROUNDING_SET:
            r = await pipeline.run_transcript(q, mode="extractive")
            results["low_grounding"].append({
                "query": q, "mode": r.answer.mode, "refusal_reason": r.answer.refusal_reason,
                "grounding_score": r.answer.grounding_score, "total_ms": r.total_ms,
            })
    finally:
        await store.close()

    def rate(label: str, want_mode: str) -> dict:
        items = results[label]
        good = sum(1 for x in items if x["mode"] == want_mode)
        return {
            "n": len(items),
            "pass": good,
            "rate": round(good / max(1, len(items)), 3),
            "modes": {m: sum(1 for x in items if x["mode"] == m) for m in {x["mode"] for x in items}},
        }

    report = {
        "run_ts": time.strftime("%Y%m%d-%H%M%S"),
        "summary": {
            "off_topic_refusal_rate": rate("off_topic", "refusal"),
            "low_grounding_refusal_rate": rate("low_grounding", "refusal"),
            "on_topic_answer_rate": rate("on_topic", "extractive"),
        },
        "sets": results,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"guardrails_{report['run_ts']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# VakRAG guardrail validation", "", f"run: {report['run_ts']}", "",
          "| set | expected | n | pass | rate | mode breakdown |", "|---|---|---|---|---|---|"]
    for key, label, want in [("off_topic", "off-topic", "refusal"), ("low_grounding", "low-grounding", "refusal"),
                             ("on_topic", "on-topic", "extractive")]:
        s = report["summary"][key + ("_refusal_rate" if key != "on_topic" else "_answer_rate")]
        md.append(f"| {label} | {want} | {s['n']} | {s['pass']} | {s['rate']} | {s['modes']} |")
    md += ["", "## per-query", ""]
    for label in ["off_topic", "low_grounding", "on_topic"]:
        md += [f"### {label}", ""]
        for x in results[label]:
            md.append(f"- `{x['query']}` → mode={x['mode']} · reason={x['refusal_reason']} · grounding={x['grounding_score']} · {x['total_ms']}ms")
        md.append("")
    (OUTDIR / f"guardrails_{report['run_ts']}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for key, label, want in [("off_topic", "off-topic", "refusal"), ("low_grounding", "low-grounding", "refusal"),
                             ("on_topic", "on-topic", "extractive")]:
        s = report["summary"][key + ("_refusal_rate" if key != "on_topic" else "_answer_rate")]
        print(f"{label}: {s['pass']}/{s['n']} -> {want} ({s['rate']})  modes={s['modes']}")
    print(f"\nwrote benchmarks/guardrails_{report['run_ts']}.json + .md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main(get_settings()))