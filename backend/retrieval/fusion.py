"""Hybrid fusion: Reciprocal Rank Fusion over vector + BM25 candidate lists.

RRF is parameter-light and robust to the fact that BM25 scores are unbounded
while vector cosine scores are [-1, 1]. Rank-based, so no score normalization
is needed.
"""

from __future__ import annotations

from collections import OrderedDict

from ..core.models import RetrievedPassage


def rrf_fuse(lists: list[list[RetrievedPassage]], *, k: int = 60, topk: int | None = None) -> list[RetrievedPassage]:
    """Merge ranked lists; each list must be pre-sorted best-first."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedPassage] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + rank)
            if item.id not in best or item.score > best[item.id].score:
                best[item.id] = item
    ordered = OrderedDict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    fused = []
    for cid, score in ordered.items():
        item = best[cid]
        # Preserve the strongest per-source score on the fused item.
        item.score = score
        fused.append(item)
        if topk and len(fused) >= topk:
            break
    return fused


def dedupe(passages: list[RetrievedPassage], *, by_lang: bool = False) -> list[RetrievedPassage]:
    """Drop near-duplicate passages sharing the same query_id+position."""
    seen: set[tuple] = set()
    out: list[RetrievedPassage] = []
    for p in passages:
        key = (p.query_id, p.position, p.language_code if by_lang else "")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out