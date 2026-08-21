"""Regression test for the grounding-timing bug (service.py).

Root cause: service.py computed grounding from `item.score` AFTER rrf_fuse
mutated it in place (fusion.py:43) to the RRF rank value (~0.016 = 1/(k+1)).
When every high-cosine candidate survived the fused top-k, the gate read
~0.016 instead of the true raw vector cosine (~0.80) and spuriously refused
the query at the 0.78 threshold.

Fix: raw_cosine is captured at retrieval time (source == "vector", before
fusion mutates score) and carried untouched; the gate now reads raw_cosine.

This test constructs the exact failing shape: a vector arm with a high raw
cosine (0.80) whose `score` gets mutated to ~0.016 by fusion, and asserts
the gate reads 0.80.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.core.models import RetrievedPassage
from backend.guardrails.grounding import GroundingGuard
from backend.retrieval.fusion import dedupe, rrf_fuse
from backend.retrieval.service import RetrievalService
from backend.retrieval.embeddings import set_embedding_service


def run(coro):
    return asyncio.run(coro)


class FakeEmbeddings:
    def embed_one(self, text: str) -> list[float]:
        return [0.1] * 16


class FakeStore:
    """Vector arm returns high-cosine candidates; every one survives the fused
    top-k, so rrf_fuse overwrites each `score` with its RRF rank value. The
    buggy gate then read ~0.016; the fixed gate must read the raw cosine."""

    def __init__(self, n_vector: int = 12) -> None:
        self.n_vector = n_vector

    async def vector_search(self, namespace: str, vector, k, *, lang=None, query_type=None):
        cos = [0.80 - i * 0.001 for i in range(self.n_vector)]
        return [
            RetrievedPassage(id=f"v-{namespace}-{i}", text=f"vector passage {i}", score=c, source="vector", raw_cosine=c)
            for i, c in enumerate(cos)
        ]

    async def bm25_search(self, namespace: str, text, k, *, lang=None, query_type=None):
        return [RetrievedPassage(id=f"b-{namespace}", text="bm25 passage", score=5.0, source="bm25", raw_cosine=None)]


def test_grounding_reads_raw_cosine_not_rrf_score():
    cfg = get_settings()
    set_embedding_service(FakeEmbeddings())
    svc = RetrievalService(cfg, FakeStore(n_vector=12))
    result = run(svc.retrieve("ভারতের রাজধানী কী?", lang="bn", graph_expand=False))

    # Confirm the bug shape: every vector candidate's `score` was mutated by
    # fusion to its RRF value (~0.016). With RRF k=60 and fusion_topk=12, all
    # 12 high-cosine items land in the fused top-k.
    lst = run(FakeStore(n_vector=12).vector_search("passage_natural", [0.1] * 16, 12))
    fused = rrf_fuse([lst], k=cfg.retrieval_rrf_k, topk=cfg.retrieval_fusion_topk)
    assert len(fused) == 12, f"expected all 12 to survive fusion, got {len(fused)}"
    assert all(c.score < 0.05 for c in fused), f"expected RRF-mutated scores, got {[round(c.score, 4) for c in fused]}"

    # The gate must read the raw cosine (~0.80), not the RRF score (~0.016).
    assert result.grounding_score >= 0.78, f"grounding {result.grounding_score} must read raw cosine"
    assert abs(result.grounding_score - 0.80) < 0.02

    # And the gate must ALLOW the answer (no spurious refusal).
    verdict = run(GroundingGuard(cfg).check(result))
    assert verdict.allow, f"gate must allow, got {verdict.category} ({verdict.reason})"


def test_raw_cosine_survives_fusion_untouched():
    # Candidates beyond the fused top-k keep their raw_cosine readable, and
    # survivors keep raw_cosine even though `score` was overwritten.
    cfg = get_settings()
    lst = run(FakeStore(n_vector=20).vector_search("passage_natural", [0.1] * 16, 20))
    fused = rrf_fuse([lst], k=cfg.retrieval_rrf_k, topk=cfg.retrieval_fusion_topk)
    assert len(fused) == 12

    survivors = {c.id for c in fused}
    dropped = [c for c in lst if c.id not in survivors]
    assert dropped, "expected at least one non-surviving candidate to check"
    for c in dropped:
        assert c.raw_cosine is not None and c.raw_cosine > 0.75
        assert c.score == c.raw_cosine  # never mutated
    for c in fused:
        assert c.raw_cosine is not None and c.raw_cosine > 0.75  # carried through
        assert c.score < 0.05  # score was overwritten — raw_cosine is the source of truth