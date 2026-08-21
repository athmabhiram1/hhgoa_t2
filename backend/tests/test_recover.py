"""PHASE 7B — clean Vertex recovery module tests.

Covers the 10 required scenarios for backend/harness/recover.py:
wrong-backend refusal, stale-embedding exclusion, exact target set, exact
resume ids, run_id completion, batch splitting, missing-node hard stop,
dimension hard stop, duplicate ids, and zero-write dry run.

All tests are hermetic (no Neo4j, no Vertex, no Ollama): the embedding service
and store are fakes, and any "fresh" vector is registered via the real
provenance registry (register_fresh_vectors) exactly as EmbeddingService does.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.config import Settings
from backend.harness.recover import (
    _batches,
    _build_rows,
    build_target_ids,
    completed_ids,
    recover_slice,
)
from backend.retrieval.embeddings import (
    clear_fresh_vectors,
    register_fresh_vectors,
)
from backend.retrieval.neo4j_store import Neo4jStore


def run(coro):
    return asyncio.run(coro)


def _cfg(**kw):
    base = dict(
        _env_file=None,
        dataset_langs="all",
        embed_backend="vertex",
        embed_dim=1024,
        neo4j_uri="bolt://localhost:7687",
    )
    base.update(kw)
    return Settings(**base)


class FakeEmbedSvc:
    def __init__(self, backend="vertex", dim=1024):
        self.backend_name = backend
        self.dim = dim
        self.embed_calls = 0

    def warm(self):
        pass

    def embed_batch(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        self.embed_calls += 1
        vecs = [[0.5] * self.dim for _ in texts]
        register_fresh_vectors(vecs, "vertex")
        return vecs


class FakeStore:
    def __init__(self, rows_by_id=None, completed=None):
        self.rows_by_id = rows_by_id or {}
        self.completed = set(completed or [])
        self.writes = 0
        self.upserted_rows = []
        self.completed_queries = 0

    async def fetch_chunk_texts(self, namespace, chunk_ids):
        return [self.rows_by_id[c] for c in chunk_ids if c in self.rows_by_id]

    async def existing_run_ids(self, namespace, chunk_ids, run_id):
        self.completed_queries += 1
        return {c for c in chunk_ids if c in self.completed}

    async def upsert_chunks(self, rows, namespace, *, run_id=None):
        self.writes += 1
        self.upserted_rows.extend(rows)


def _meta(chunk_id, text="hello"):
    return {
        "chunk_id": chunk_id,
        "namespace": "passage_natural",
        "text": text,
        "lang": "ta",
        "query_id": 1,
        "query_type": "DESCRIPTION",
        "position": 0,
        "is_selected": True,
        "passage_pos": 0,
        "doc_key": "d",
    }


RUN = "vertex-recovery-20260818-01"


def test_wrong_backend_refusal():
    store = FakeStore(rows_by_id={f"c{i}": _meta(f"c{i}") for i in range(2)})
    embed = FakeEmbedSvc(backend="ollama")
    with pytest.raises(RuntimeError, match="refusing recovery"):
        run(recover_slice(store, _cfg(), "passage_natural", RUN,
                          target_ids=["c0", "c1"], embed_svc=embed))
    assert store.writes == 0
    assert embed.embed_calls == 0


def test_stale_embedding_cannot_enter_recovery():
    stale = {**_meta("c0"), "embedding": [9.9] * 1024}
    with pytest.raises(RuntimeError, match="stale-vector guard"):
        _build_rows([stale], [[0.5] * 1024], RUN)
    fresh_out = _build_rows([_meta("c0")], [[0.7] * 1024], RUN)
    assert fresh_out[0]["embedding"] == [0.7] * 1024
    assert fresh_out[0]["embed_backend"] == "vertex"
    assert fresh_out[0]["reembed_run"] == RUN


def test_exact_target_set():
    cfg = _cfg()
    for ns, expected in (
        ("passage_natural", 28518),
        ("query_anchored", 28518),
        ("passage_en", 2037),
    ):
        ids = build_target_ids(cfg, ns)
        assert len(ids) == expected, ns
        assert len(set(ids)) == expected, ns


def test_exact_resume_ids():
    ids = [f"c{i}" for i in range(10)]
    store = FakeStore(rows_by_id={i: _meta(i) for i in ids}, completed={"c0", "c1", "c2"})
    embed = FakeEmbedSvc()
    report = run(recover_slice(store, _cfg(), "passage_natural", RUN,
                               target_ids=ids, embed_svc=embed, batch_size=3))
    upserted = [r["chunk_id"] for r in store.upserted_rows]
    assert upserted == ["c3", "c4", "c5", "c6", "c7", "c8", "c9"]
    assert report["completed_before"] == 3
    assert report["pending"] == 7
    assert all(i not in upserted for i in ("c0", "c1", "c2"))


def test_run_id_completion():
    ids = [f"c{i}" for i in range(4)]
    store = FakeStore(rows_by_id={i: _meta(i) for i in ids})
    embed = FakeEmbedSvc()
    run(recover_slice(store, _cfg(), "passage_natural", RUN,
                      target_ids=ids, embed_svc=embed))
    assert store.writes == 1
    for r in store.upserted_rows:
        assert r["reembed_run"] == RUN
        assert r["embed_backend"] == "vertex"


def test_batch_splitting():
    ids = [f"c{i}" for i in range(21)]
    batches = _batches(ids, 5)
    assert [len(b) for b in batches] == [5, 5, 5, 5, 1]
    assert _batches(ids) == [ids[i : i + 256] for i in range(0, 21, 256)]


def test_missing_db_node_hard_stop():
    ids = ["c0", "c1", "c2"]
    store = FakeStore(rows_by_id={"c0": _meta("c0"), "c1": _meta("c1")})
    embed = FakeEmbedSvc()
    with pytest.raises(RuntimeError, match="missing DB nodes"):
        run(recover_slice(store, _cfg(), "passage_natural", RUN,
                          target_ids=ids, embed_svc=embed))
    assert store.writes == 0


def test_dimension_mismatch_hard_stop():
    ids = ["c0", "c1"]
    store = FakeStore(rows_by_id={"c0": _meta("c0"), "c1": _meta("c1")})
    embed = FakeEmbedSvc(dim=768)
    with pytest.raises(RuntimeError, match="dim 768"):
        run(recover_slice(store, _cfg(), "passage_natural", RUN,
                          target_ids=ids, embed_svc=embed))
    assert store.writes == 0


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate ids"):
        _batches(["a", "a", "b"], 2)


def test_dry_run_zero_db_writes():
    ids = [f"c{i}" for i in range(5)]
    store = FakeStore(rows_by_id={i: _meta(i) for i in ids})
    embed = FakeEmbedSvc()
    report = run(recover_slice(store, _cfg(), "passage_natural", RUN,
                               target_ids=ids, embed_svc=embed, dry_run=True))
    assert store.writes == 0
    assert store.upserted_rows == []
    assert store.completed_queries == 0, "dry run must not even query completion"
    assert embed.embed_calls == 1
    assert report["processed"] == 5
    assert report["dry_run"] is True


def test_completed_ids_uses_run_only():
    store = FakeStore(completed={"a", "b"})
    got = run(completed_ids(store, "passage_natural", ["a", "b", "c"], RUN))
    assert got == {"a", "b"}


def test_upsert_query_run_tag():
    q = Neo4jStore._upsert_query("passage_natural", "ChunkNatural", run_id=RUN)
    assert "n.reembed_run = row.reembed_run" in q
    assert _no_trailing_comma_before_with(q)
    q0 = Neo4jStore._upsert_query("passage_natural", "ChunkNatural")
    assert "reembed_run" not in q0


def _no_trailing_comma_before_with(query):
    lines = [ln.strip() for ln in query.strip().splitlines()]
    idx = lines.index("WITH n, row")
    return not lines[idx - 1].endswith(",")


def test_upsert_query_no_trailing_comma_all_variants():
    for ns, label in (("passage_natural", "ChunkNatural"), ("query_anchored", "ChunkAnchored")):
        assert _no_trailing_comma_before_with(Neo4jStore._upsert_query(ns, label))
        assert _no_trailing_comma_before_with(Neo4jStore._upsert_query(ns, label, run_id=RUN))
    assert _no_trailing_comma_before_with(Neo4jStore._upsert_query("passage_en", "ChunkEnglish"))
    assert _no_trailing_comma_before_with(Neo4jStore._upsert_query("passage_en", "ChunkEnglish", run_id=RUN))


def _no_dangling_comma_before_clause(query):
    lines = [ln.strip() for ln in query.strip().splitlines()]
    for target in ("REMOVE n.lang", "WITH n, row"):
        if target in lines:
            i = lines.index(target)
            if lines[i - 1].endswith(","):
                return False
    return True


def test_upsert_query_no_dangling_comma_before_remove_or_with():
    for ns, label in (("passage_natural", "ChunkNatural"), ("query_anchored", "ChunkAnchored"), ("passage_en", "ChunkEnglish")):
        assert _no_dangling_comma_before_clause(Neo4jStore._upsert_query(ns, label))
        assert _no_dangling_comma_before_clause(Neo4jStore._upsert_query(ns, label, run_id=RUN))


def test_upsert_query_passage_en_comma_fix():
    q = Neo4jStore._upsert_query("passage_en", "ChunkEnglish", run_id=RUN)
    assert "REMOVE n.lang" in q
    assert "n.embed_backend = row.embed_backend," in q


def test_upsert_rejects_run_id_row_mismatch():
    clear_fresh_vectors()
    register_fresh_vectors([[0.1] * 1024], "vertex")
    s = Neo4jStore.__new__(Neo4jStore)
    s.cfg = _cfg()
    written = []

    async def noop(query, params=None):
        written.append(query)

    s._run_write = noop
    row = {
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
        "embedding": [0.1] * 1024,
        "embed_backend": "vertex",
    }
    with pytest.raises(RuntimeError, match="run_id given but a row does not carry"):
        run(s.upsert_chunks([row], "passage_natural", run_id=RUN))
    assert written == []


def test_upsert_accepts_fresh_vertex_with_run():
    clear_fresh_vectors()
    register_fresh_vectors([[0.1] * 1024], "vertex")
    s = Neo4jStore.__new__(Neo4jStore)
    s.cfg = _cfg()
    written = []

    async def noop(query, params=None):
        written.append(query)

    s._run_write = noop
    row = {
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
        "embedding": [0.1] * 1024,
        "embed_backend": "vertex",
        "reembed_run": RUN,
    }
    run(s.upsert_chunks([row], "passage_natural", run_id=RUN))
    assert len(written) == 1
    assert "n.reembed_run = row.reembed_run" in written[0]