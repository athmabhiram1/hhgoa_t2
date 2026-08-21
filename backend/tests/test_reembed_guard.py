"""PHASE 7 — fresh-vector provenance guard tests.

Regression for the Aug 2026 contamination: the Phase-5/6 bench scripts read the
stored `embedding` property from Neo4j and re-upserted those SAME stale vectors
with embed_backend='vertex', tagging ~300 chunks as freshly Vertex without ever
calling the Vertex API. The guard added to `Neo4jStore.upsert_chunks` must make
that write fail, and `fetch_chunk_texts` must structurally never return a stored
embedding.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.config import Settings
from backend.retrieval.embeddings import (
    clear_fresh_vectors,
    fresh_backend_of,
    register_fresh_vectors,
)
from backend.retrieval.neo4j_store import Neo4jStore


def run(coro):
    return asyncio.run(coro)


def _store(run_write=None):
    s = Neo4jStore.__new__(Neo4jStore)
    s.cfg = Settings(neo4j_uri="bolt://localhost:7687")
    s._driver = None
    if run_write is not None:
        s._run_write = run_write
    return s


def _row(chunk_id="abc", embedding=None, backend="vertex"):
    return {
        "chunk_id": chunk_id,
        "namespace": "passage_natural",
        "text": "hello",
        "lang": "ta",
        "query_id": 1,
        "query_type": "DESCRIPTION",
        "position": 0,
        "is_selected": True,
        "passage_pos": 0,
        "doc_key": "d",
        "embedding": embedding or [0.1] * 1024,
        "embed_backend": backend,
    }


def test_fresh_registry_roundtrip():
    clear_fresh_vectors()
    vec = [0.42] * 1024
    assert fresh_backend_of(vec) is None
    register_fresh_vectors([vec], "vertex")
    assert fresh_backend_of(vec) == "vertex"


def test_fresh_registry_distinguishes_backends():
    clear_fresh_vectors()
    register_fresh_vectors([[1.0] * 1024], "ollama")
    register_fresh_vectors([[2.0] * 1024], "vertex")
    assert fresh_backend_of([1.0] * 1024) == "ollama"
    assert fresh_backend_of([2.0] * 1024) == "vertex"


def test_guard_rejects_stale_stored_vector_tagged_vertex():
    """The exact Phase-5/6 mechanism: a stored embedding (never freshly produced
    in this process) re-tagged as vertex must be REFUSED."""
    clear_fresh_vectors()
    written = []
    async def noop(query, params=None):
        written.append(query)
    s = _store(run_write=noop)
    with pytest.raises(RuntimeError, match="REFUSED embed write"):
        run(s.upsert_chunks([_row()], "passage_natural"))
    assert written == [], "guard must fail before any write"


def test_guard_rejects_vertex_tag_on_other_backend_vector():
    """A vector freshly produced by ollama must NOT be writable as vertex."""
    clear_fresh_vectors()
    register_fresh_vectors([[0.1] * 1024], "ollama")
    written = []
    async def noop(query, params=None):
        written.append(query)
    s = _store(run_write=noop)
    with pytest.raises(RuntimeError, match="REFUSED embed write"):
        run(s.upsert_chunks([_row()], "passage_natural"))
    assert written == []


def test_guard_accepts_fresh_vertex_vector():
    """The only sanctioned path: vector freshly produced by Vertex in this
    process may be upserted with embed_backend='vertex'."""
    clear_fresh_vectors()
    register_fresh_vectors([[0.1] * 1024], "vertex")
    written = []
    async def noop(query, params=None):
        written.append(query)
    s = _store(run_write=noop)
    run(s.upsert_chunks([_row()], "passage_natural"))
    assert len(written) == 1, "fresh vertex vector must be written"


def test_guard_accepts_ollama_vector_tagged_ollama():
    clear_fresh_vectors()
    register_fresh_vectors([[0.1] * 1024], "ollama")
    written = []
    async def noop(query, params=None):
        written.append(query)
    s = _store(run_write=noop)
    run(s.upsert_chunks([_row(backend="ollama")], "passage_natural"))
    assert len(written) == 1


def test_fetch_chunk_texts_never_returns_embedding():
    """The harness input source must structurally omit the stored embedding."""
    rows = [
        {
            "chunk_id": "abc",
            "namespace": "passage_natural",
            "text": "hello",
            "lang": "ta",
            "query_id": 1,
            "query_type": "DESCRIPTION",
            "position": 0,
            "is_selected": True,
            "passage_pos": 0,
            "doc_key": "d",
            "embedding": [9.9] * 1024,  # present in DB record; must NOT surface
        }
    ]
    async def fake_run(query, params=None):
        return rows
    s = _store()
    s._run = fake_run
    out = run(s.fetch_chunk_texts("passage_natural", ["abc"]))
    assert len(out) == 1
    assert "embedding" not in out[0], "fetch_chunk_texts must never expose embedding"
    assert out[0]["text"] == "hello"