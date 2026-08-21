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


def test_rrf_default_weights_are_all_one():
    vec = [_p("a", 0.9), _p("b", 0.8)]
    bm = [_p("b", 9.0), _p("c", 8.0)]
    fused_default = rrf_fuse([vec, bm], k=60, topk=3)
    fused_explicit = rrf_fuse([vec, bm], k=60, topk=3, weights=[1.0, 1.0])
    assert [p.id for p in fused_default] == [p.id for p in fused_explicit]
    assert abs(fused_default[0].score - fused_explicit[0].score) < 1e-12


def test_rrf_b2_weights_vector_bm25():
    vec = [_p("v", 0.9), _p("s", 0.8)]
    bm = [_p("s", 9.0), _p("b", 8.0)]
    fused = rrf_fuse([vec, bm], k=60, topk=3, weights=[1.0, 0.5])
    # "s": vector rank2 -> 1.0/62, bm25 rank1 -> 0.5/61
    s_score = 1.0 / 62 + 0.5 / 61
    # "v": vector rank1 -> 1.0/61
    v_score = 1.0 / 61
    # "b": bm25 rank2 -> 0.5/62
    b_score = 0.5 / 62
    assert [p.id for p in fused] == ["s", "v", "b"]
    assert abs(fused[0].score - s_score) < 1e-12
    assert abs(fused[1].score - v_score) < 1e-12
    assert abs(fused[2].score - b_score) < 1e-12


def test_rrf_weights_are_effective_vector_wins_over_bm25():
    # Identical rank-1 arms; the vector arm (w=1.0) must outrank the bm25 arm
    # (w=0.5) under B2, and tie exactly under all-1.0.
    vec = [_p("v", 1.0)]
    bm = [_p("v", 2.0)]
    fused_w = rrf_fuse([vec, bm], k=60, weights=[1.0, 0.5])
    fused_flat = rrf_fuse([vec, bm], k=60, weights=[1.0, 1.0])
    assert len(fused_w) == 1
    assert abs(fused_w[0].score - (1.0 / 61 + 0.5 / 61)) < 1e-12
    assert abs(fused_flat[0].score - 2.0 / 61) < 1e-12


def test_rrf_weights_length_mismatch_raises():
    vec = [_p("a", 0.9)]
    bm = [_p("b", 9.0)]
    try:
        rrf_fuse([vec, bm], k=60, weights=[1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched weights length")


def test_dedupe_by_query_position():
    a = _p("x1", 1.0, qid="q1", pos=0)
    b = _p("x2", 1.0, qid="q1", pos=0)  # same query, same position, different id
    c = _p("x3", 1.0, qid="q1", pos=1)
    out = dedupe([a, b, c], by_lang=True)
    assert [p.id for p in out] == ["x1", "x3"]