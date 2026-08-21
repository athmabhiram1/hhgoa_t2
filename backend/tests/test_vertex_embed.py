"""Vertex AI embedding backend unit tests (offline — no network / OAuth).

Covers the `:predict` endpoint URL shape, the thread-pool parallelism and
input-order preservation of `_VertexBackend.embed()`, transient vs permanent
error classification, and task_type plumbing through EmbeddingService
(indexed chunks = RETRIEVAL_DOCUMENT, queries = RETRIEVAL_QUERY).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Settings
from backend.retrieval.embeddings import EmbeddingService, _VertexBackend


def _vertex(cfg: Settings | None = None):
    """Build a _VertexBackend without touching credentials/network."""
    c = cfg or Settings(_env_file=None)
    b = object.__new__(_VertexBackend)
    b.cfg = c
    b.project = c.vertex_project or "proj-123"
    b.location = c.vertex_location
    b.model = c.vertex_embed_model
    b.dim = c.gemini_embed_dim
    b.max_chars = c.gemini_embed_max_chars
    b._concurrency = c.vertex_embed_concurrency
    b._client = None
    return b


def test_vertex_endpoint_url():
    b = _vertex()
    assert b._endpoint() == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-123"
        "/locations/us-central1/publishers/google/models/gemini-embedding-001:predict"
    )


def test_vertex_embed_parallel_preserves_order():
    b = _vertex()
    seen: list[tuple[str, str]] = []

    def fake_one(text: str, task_type: str) -> list[float]:
        seen.append((text, task_type))
        return [0.0] * 1024

    b._embed_one = fake_one
    texts = ["a", "b", "c", "d"]
    out = b.embed(texts, task_type="RETRIEVAL_DOCUMENT")

    assert [t for t, _ in seen] == texts
    assert all(tt == "RETRIEVAL_DOCUMENT" for _, tt in seen)
    assert len(out) == 4 and all(len(v) == 1024 for v in out)
    assert [v[0] for v in out] == [0.0] * 4


def test_vertex_embed_truncates_long_inputs():
    b = _vertex()
    captured: dict = {}

    def fake_one(text: str, task_type: str) -> list[float]:
        captured["text"] = text
        return [0.0]

    b._embed_one = fake_one
    b.embed(["x" * 9000], task_type="RETRIEVAL_QUERY")
    assert len(captured["text"]) <= b.max_chars


class _Resp:
    def __init__(self, code: int) -> None:
        self.status_code = code


class _Exc(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"http {code}")
        self.response = _Resp(code)


def test_vertex_is_transient_classification():
    b = _vertex()
    assert b._is_transient(_Exc(429))  # quota — retry
    assert b._is_transient(_Exc(500))  # server error — retry
    assert not b._is_transient(_Exc(400))  # bad request — raise
    assert not b._is_transient(_Exc(401))  # bad credentials — raise
    assert not b._is_transient(_Exc(403))  # permission denied — raise
    assert not b._is_transient(_Exc(404))  # not found — raise


def test_embedding_service_task_type_plumbing():
    svc = EmbeddingService(Settings())
    svc._backend_name = "vertex"
    captured: dict = {}

    class FakeBackend:
        def embed(self, texts, batch_size, task_type):
            captured["task_type"] = task_type
            return [[1.0]] * len(texts)

    svc._backend = FakeBackend()

    svc.embed_batch(["doc1", "doc2"])
    assert captured["task_type"] == "RETRIEVAL_DOCUMENT"

    svc.embed_one("q1")
    assert captured["task_type"] == "RETRIEVAL_QUERY"