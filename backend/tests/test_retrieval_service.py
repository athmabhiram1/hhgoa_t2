"""RetrievalService contract tests (query_type opt-in + cache).

LOCKED DECISION (Phase 5): live retrieval is UNFILTERED by default; the
heuristic query_type classifier must NOT drive filtering. query_type only
filters when the caller passes it explicitly. This test pins that contract.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.core.models import RetrievedPassage
from backend.retrieval.service import RetrievalService
from backend.retrieval.embeddings import set_embedding_service


def run(coro):
    return asyncio.run(coro)


class FakeEmbeddings:
    def embed_one(self, text: str) -> list[float]:
        return [0.1] * 16


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def vector_search(self, namespace: str, vector, k, *, lang=None, query_type=None):
        self.calls.append({"arm": "vector", "ns": namespace, "lang": lang, "query_type": query_type})
        return [RetrievedPassage(id=f"v-{namespace}", text="vector passage", score=0.8, source="vector")]

    async def bm25_search(self, namespace: str, text, k, *, lang=None, query_type=None):
        self.calls.append({"arm": "bm25", "ns": namespace, "lang": lang, "query_type": query_type})
        return [RetrievedPassage(id=f"b-{namespace}", text="bm25 passage", score=0.6, source="bm25")]


def test_retrieval_unfiltered_by_default():
    cfg = get_settings()
    set_embedding_service(FakeEmbeddings())
    store = FakeStore()
    svc = RetrievalService(cfg, store)
    result = run(svc.retrieve("কৰ্পোৰেচন কি?", graph_expand=False))
    assert result.candidates
    for call in store.calls:
        assert call["query_type"] is None, f"expected no query_type filter, got {call}"


def test_retrieval_explicit_query_type_filters():
    cfg = get_settings()
    set_embedding_service(FakeEmbeddings())
    store = FakeStore()
    svc = RetrievalService(cfg, store)
    result = run(svc.retrieve("what is the capital?", query_type="ENTITY", graph_expand=False))
    assert result.candidates
    for call in store.calls:
        assert call["query_type"] == "ENTITY", f"expected ENTITY filter, got {call}"


def test_retrieval_cache_hit_marks_breakdown():
    cfg = get_settings()
    set_embedding_service(FakeEmbeddings())
    store = FakeStore()
    svc = RetrievalService(cfg, store)
    first = run(svc.retrieve("same query", graph_expand=False))
    second = run(svc.retrieve("same query", graph_expand=False))
    assert first.breakdown_ms.get("cache") == "miss"
    assert second.breakdown_ms.get("cache") == "hit"
    assert svc.cache_hits == 1 and svc.cache_misses == 1
    assert len(store.calls) == 4  # only the first call searches


def test_connect_config_rewrites_neo4j_plus_s():
    """Driver >=6 forbids trusted_certificates on +s/+ssc URI schemes; the
    connect config must rewrite neo4j+s:// -> neo4j:// with explicit encryption
    + certifi trust so the app boots on both driver 5.x (TrustStore) and 6.x
    (TrustCustomCAs). Regression for the restart crash after the driver upgrade."""
    from backend.config import Settings
    from backend.retrieval.neo4j_store import _connect_config

    cfg = Settings(neo4j_uri="neo4j+s://abc123.databases.neo4j.io")
    conf = _connect_config(cfg)
    assert conf["uri"] == "neo4j://abc123.databases.neo4j.io"
    assert conf["encrypted"] is True
    assert "trusted_certificates" in conf


def test_connect_config_passthrough_plain_uri():
    from backend.config import Settings
    from backend.retrieval.neo4j_store import _connect_config

    cfg = Settings(neo4j_uri="neo4j://localhost:7687")
    assert _connect_config(cfg) == {"uri": "neo4j://localhost:7687"}


def test_trust_resolves_to_custom_cas_on_installed_driver():
    from backend.retrieval.neo4j_store import _trust
    from neo4j import TrustCustomCAs

    assert isinstance(_trust(), TrustCustomCAs)