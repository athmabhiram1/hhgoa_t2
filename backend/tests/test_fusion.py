import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.models import RetrievedPassage
from backend.retrieval.fusion import dedupe, rrf_fuse


def _p(pid: str, score: float, qid: str = "q1", pos: int = 0, lang: str = "hi") -> RetrievedPassage:
    return RetrievedPassage(id=pid, text=f"text {pid}", score=score, source="vector", query_id=qid, position=pos, language_code=lang)


def test_rrf_orders_by_rank_not_score():
    vec = [_p("a", 0.9), _p("b", 0.8)]
    bm = [_p("b", 9.0), _p("c", 8.0)]  # bm25 scores are unbounded
    fused = rrf_fuse([vec, bm], topk=3)
    assert [p.id for p in fused] == ["b", "a", "c"]
    assert fused[0].score > fused[1].score > fused[2].score


def test_rrf_deduplicates_same_id():
    vec = [_p("a", 0.9)]
    bm = [_p("a", 7.0)]
    fused = rrf_fuse([vec, bm], topk=10)
    assert len(fused) == 1


def test_rrf_topk_limits():
    vec = [_p(str(i), 1.0) for i in range(10)]
    fused = rrf_fuse([vec], topk=4)
    assert len(fused) == 4


def test_dedupe_by_query_position():
    a = _p("x1", 1.0, qid="q1", pos=0)
    b = _p("x2", 1.0, qid="q1", pos=0)  # same query, same position, different id
    c = _p("x3", 1.0, qid="q1", pos=1)
    out = dedupe([a, b, c], by_lang=True)
    assert [p.id for p in out] == ["x1", "x3"]