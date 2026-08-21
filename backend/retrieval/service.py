"""Retrieval service — wires embedding + Neo4j hybrid search + RRF fusion +
optional rerank + light graph expansion into one call used by the fast path.

Runs the two main namespaces (natural + anchored) plus BM25 in parallel to
keep wall-clock low; the other namespaces stay indexable but the query path
favors speed (measured in the benchmark).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from ..config import Settings
from ..core.models import RetrievedPassage, RetrievalResult
from ..core.tracing import span
from .embeddings import query_embedding_async
from .fusion import dedupe, rrf_fuse
from .graph import sibling_expand
from .neo4j_store import Neo4jStore
from .reranker import Reranker

logger = logging.getLogger(__name__)

_PRIMARY_NS = ["passage_natural", "query_anchored"]


def _search_arm(namespace: str, lang: str | None) -> str | None:
    """Guard against the passage_en silent-break class of bug.

    passage_en chunks carry NO lang property (English is language-agnostic by
    design — any Indic query language retrieves into it). Passing a lang filter
    to a passage_en search would silently return nothing, so that combination
    is made structurally impossible: if this arm is ever wired in, it MUST be
    searched with lang=None. Returns the namespace to search, or raises.
    """
    if namespace == "passage_en" and lang is not None:
        raise ValueError("passage_en must be searched without a lang filter (language-agnostic arm)")
    return namespace


class RetrievalService:
    def __init__(self, cfg: Settings, store: Neo4jStore) -> None:
        self.cfg = cfg
        self.store = store
        self.reranker = Reranker(cfg) if cfg.retrieval_rerank == "local" else None
        self._cache: OrderedDict[tuple, RetrievalResult] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_get(self, key: tuple) -> RetrievalResult | None:
        if key not in self._cache:
            self.cache_misses += 1
            return None
        self._cache.move_to_end(key)
        self.cache_hits += 1
        return self._cache[key]

    def _cache_put(self, key: tuple, result: RetrievalResult) -> None:
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self.cfg.benchmark_cache_size:
            self._cache.popitem(last=False)

    async def retrieve(
        self,
        query: str,
        *,
        lang: str | None = None,
        query_type: str | None = None,
        graph_expand: bool = True,
        on_embed_queued=None,
    ) -> RetrievalResult:
        cache_key = (query, lang, query_type, graph_expand)
        cached = self._cache_get(cache_key)
        if cached is not None:
            t_hit = time.perf_counter()
            result = cached.model_copy(deep=True)
            result.breakdown_ms = dict(result.breakdown_ms)
            result.breakdown_ms["cache"] = "hit"
            result.breakdown_ms["cache_latency_ms"] = int((time.perf_counter() - t_hit) * 1000)
            result.latency_ms = result.breakdown_ms["cache_latency_ms"]
            return result

        start = time.perf_counter()
        breakdown: dict[str, int] = {}

        with span("retrieve.embed"):
            vec = await query_embedding_async(query, bound=self.cfg.embed_concurrency, on_queued=on_embed_queued)
        breakdown["embed_ms"] = int((time.perf_counter() - start) * 1000)

        vk, bk = self.cfg.retrieval_vector_k, self.cfg.retrieval_bm25_k

        with span("retrieve.search"):
            vector_lists = await asyncio.gather(
                *[self.store.vector_search(_search_arm(ns, lang), vec, vk, lang=lang, query_type=query_type) for ns in _PRIMARY_NS]
            )
            bm25_lists = await asyncio.gather(
                *[self.store.bm25_search(_search_arm(ns, lang), query, bk, lang=lang, query_type=query_type) for ns in _PRIMARY_NS]
            )
        breakdown["search_ms"] = int((time.perf_counter() - start) * 1000) - breakdown.get("embed_ms", 0)

        with span("retrieve.fuse"):
            fused = rrf_fuse(
                vector_lists + bm25_lists,
                k=self.cfg.retrieval_rrf_k,
                topk=self.cfg.retrieval_fusion_topk,
                weights=[self.cfg.retrieval_vector_weight] * len(vector_lists)
                + [self.cfg.retrieval_bm25_weight] * len(bm25_lists),
            )
            fused = dedupe(fused, by_lang=True)

        # Grounding uses the best raw vector cosine from the vector arm.
        # Calibrated on qwen3-embedding cosine scale (threshold 0.78, see
        # CONTEXT.md) — NOT the RRF rank score (which maxes ~0.03) and NOT
        # BM25 (unbounded, and inverted for out-of-domain queries).
        #
        # raw_cosine is captured at retrieval time (source == "vector") BEFORE
        # rrf_fuse mutates `score` in place (fusion.py:43). Reading the mutated
        # score here was the grounding-timing bug: when every candidate survives
        # the fused top-k, `score` holds the RRF rank value (~0.016) instead of
        # the true cosine (~0.80), spuriously refusing queries at the 0.78 gate.
        grounding = max((c.raw_cosine or 0.0 for lst in vector_lists for c in lst), default=0.0)

        if self.reranker is not None and fused:
            with span("retrieve.rerank"):
                fused = self.reranker.rerank(query, fused, topk=6)

        if graph_expand and fused:
            with span("retrieve.graph_expand"):
                siblings = await asyncio.gather(*[sibling_expand(self.store, fused[0])])
                fused.extend(s for s in siblings[0])
                fused = dedupe(fused, by_lang=True)

        breakdown["fuse_ms"] = int((time.perf_counter() - start) * 1000) - breakdown.get("embed_ms", 0) - breakdown.get("search_ms", 0)
        breakdown["cache"] = "miss"

        result = RetrievalResult(
            query=query,
            candidates=fused,
            grounding_score=round(grounding, 4),
            latency_ms=int((time.perf_counter() - start) * 1000),
            breakdown_ms=breakdown,
        )
        self._cache_put(cache_key, result)
        return result