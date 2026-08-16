"""Embed concurrency gate + queued callback contract tests.

PHASE 6 (directive): the request-path embed call is bounded by
asyncio.Semaphore(embed_concurrency) so Ollama's serial /api/embed doesn't
pile up in-flight requests; when the gate is contended the SSE stream emits a
"queued" stage event instead of stalling silently. Cache hits must bypass the
gate entirely.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retrieval.embeddings import query_embedding, query_embedding_async, set_embedding_service


class SlowEmbeddings:
    """embeddings whose embed_one blocks for `delay` — simulates Ollama CPU."""

    def __init__(self, delay: float = 0.25) -> None:
        self.delay = delay
        self.active = 0
        self.peak_active = 0

    def embed_one(self, text: str) -> list[float]:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.delay)
        finally:
            self.active -= 1
        return [0.5] * 16


def test_embed_gate_bounds_concurrency():
    svc = SlowEmbeddings(delay=0.3)
    set_embedding_service(svc)

    async def go():
        await asyncio.gather(*(query_embedding_async(f"q{i}", bound=2) for i in range(6)))

    asyncio.run(go())
    assert svc.peak_active <= 2, f"expected at most 2 concurrent embeds, saw {svc.peak_active}"


def test_embed_gate_fires_queued_when_contended():
    svc = SlowEmbeddings(delay=0.3)
    set_embedding_service(svc)
    queued_calls = []

    async def go():
        async def on_queued():
            queued_calls.append(1)

        await asyncio.gather(*(query_embedding_async(f"qq{i}", bound=1, on_queued=on_queued) for i in range(3)))

    asyncio.run(go())
    assert len(queued_calls) >= 2, f"expected >=2 queued events under bound=1, saw {len(queued_calls)}"


def test_embed_cache_hit_bypasses_gate():
    svc = SlowEmbeddings(delay=0.3)
    set_embedding_service(svc)

    # First call populates the cache; second must not touch the backend.
    assert query_embedding("cached-query") == [0.5] * 16
    hits = svc.peak_active
    assert query_embedding("cached-query") == [0.5] * 16
    assert svc.peak_active == hits, "cache hit must not hit the embed backend"


def test_embed_async_cache_hit_bypasses_gate():
    svc = SlowEmbeddings(delay=0.3)
    set_embedding_service(svc)

    async def go():
        first = await query_embedding_async("async-cached", bound=1)
        second = await query_embedding_async("async-cached", bound=1, on_queued=lambda: None)
        return first, second

    first, second = asyncio.run(go())
    assert first == [0.5] * 16 and second == [0.5] * 16
    assert svc.peak_active == 1, "async cache hit must bypass the gate"