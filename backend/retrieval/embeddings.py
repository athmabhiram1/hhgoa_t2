"""Multilingual embeddings — local (Qwen3 via Ollama) or Gemini Embedding API.

Backend order is driven by `cfg.embed_backend`:
  - "ollama" (default): POST /api/embed on a local Ollama server. Dense vectors
    only — the lexical/BM25 arm is served by Neo4j's Lucene fulltext index.
  - "gemini": batchEmbedContents on the Gemini API (gemini-embedding-001,
    output_dimensionality=1024, L2-normalized). Uses the same GEMINI_API_KEY as
    the generation LLM — no new secrets.
  - "vertex": gemini-embedding-001 via Vertex AI `:predict` (aiplatform).
    One text per request (Vertex has no batchEmbedContents for this model), so
    embed() parallelizes with a thread pool. Billed to the GCP project, so the
    $300 Free Trial credit covers it — unlike the AI Studio Gemini API. Auth =
    service-account JSON (cfg.vertex_credentials). Uses task_type
    (RETRIEVAL_QUERY for queries, RETRIEVAL_DOCUMENT for indexed chunks).
  - "fastembed" / "sentence-transformers": ONNX / PyTorch fallbacks (bge-m3).
All backends are lazy-imported / lazily connected so the rest of the app runs
without them; a missing backend degrades health instead of crashing boot.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable

from ..config import Settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_SVC: "EmbeddingService | None" = None

# Query-text → vector cache. Thread-safe bounded dict (replaces the old
# lru_cache) so the async path can *peek* cache hits without acquiring the
# embed semaphore — cache hits must never wait on the network gate.
_QUERY_CACHE: OrderedDict[str, tuple[float, ...]] = OrderedDict()
_QUERY_CACHE_MAX = 2048
_CACHE_LOCK = threading.Lock()

# Request-path embed concurrency gate (see cfg.embed_concurrency). Keyed by
# event-loop id because asyncio primitives are loop-bound and the app/tests
# each spin up fresh loops.
_EMBED_SEMS: dict[int, asyncio.Semaphore] = {}
_EMBED_SEM_BOUND: int = 2

# --- fresh-vector provenance registry (Phase 7) ------------------------------
# A vector may carry embed_backend='vertex' ONLY if it was freshly produced by
# the Vertex backend in THIS process. Every vector returned by
# EmbeddingService.embed_batch() is registered here (digest -> producing
# backend); the store's upsert guard rejects any vertex-tagged row whose vector
# was not registered as produced by 'vertex'. This closes the Phase-6
# contamination hole where bench scripts read the stored `embedding` property
# from Neo4j and re-tagged stale Qwen3-era vectors as Vertex without ever
# calling the Vertex API. Keyed by a process-stable digest (not the raw 1024-dim
# tuple) so a full-corpus re-embed (~36k chunks) stays a few MB instead of ~1GB.
_FRESH_VECTORS: dict[int, str] = {}
_FRESH_VECTORS_MAX = 200_000
_FRESH_LOCK = threading.Lock()


def _vec_digest(vec) -> int:
    return hash(tuple(vec))


def register_fresh_vectors(vectors: list[list[float]], backend: str) -> None:
    """Record freshly produced vectors (digest -> producing backend name)."""
    with _FRESH_LOCK:
        if len(_FRESH_VECTORS) > _FRESH_VECTORS_MAX:
            _FRESH_VECTORS.clear()
        for vec in vectors:
            _FRESH_VECTORS[_vec_digest(vec)] = backend


def fresh_backend_of(vec) -> str | None:
    """Backend name that produced `vec` in this process, or None if unknown."""
    with _FRESH_LOCK:
        return _FRESH_VECTORS.get(_vec_digest(vec))


def clear_fresh_vectors() -> None:
    """Drop the provenance registry (e.g. at the start of a harness run)."""
    with _FRESH_LOCK:
        _FRESH_VECTORS.clear()


def set_embedding_service(svc: "EmbeddingService") -> None:
    global _SVC
    _SVC = svc


def get_embedding_service() -> "EmbeddingService":
    if _SVC is None:
        raise RuntimeError("Embedding service not initialized — call set_embedding_service() at startup")
    return _SVC


class _OllamaBackend:
    """Minimal client for `POST {base}/api/embed` (Ollama embeddings API)."""

    def __init__(self, cfg: Settings) -> None:
        import httpx

        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=5.0),
        )
        self.model = cfg.ollama_embed_model
        self.max_chars = cfg.embed_max_chars

    def embed(self, texts: list[str], batch_size: int = 128, task_type: str | None = None) -> list[list[float]]:
        cleaned = [t[: self.max_chars] for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        import httpx

        import time as _time
        import random as _random

        body = {
            "model": self.model,
            "input": batch,
            "truncate": True,
            "keep_alive": self.cfg.ollama_embed_keep_alive,
            "options": {"num_batch": self.cfg.ollama_embed_num_batch},
        }
        # Ollama's runner exposes a tokenize endpoint on a random port that
        # intermittently drops on Windows (400 "connection refused"); a short
        # retry recovers it. This is the embed path's counterpart to
        # retry_with_backoff for a synchronous call.
        attempts = 8
        base_delay = 0.3
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post("/api/embed", json=body)
                resp.raise_for_status()
                return [list(map(float, vec)) for vec in resp.json()["embeddings"]]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if not self._is_transient(exc):
                    raise
                if attempt >= attempts:
                    raise
                delay = min(8.0, base_delay * (2 ** (attempt - 1))) * (1.0 + _random.uniform(-0.2, 0.2))
                body_hint = ""
                resp = getattr(exc, "response", None)
                if resp is not None and getattr(resp, "text", ""):
                    body_hint = f" body={resp.text[:200]!r}"
                logger.warning("Ollama embed retry %d/%d in %.3fs: %s%s", attempt, attempts, delay, exc, body_hint)
                _time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """True for errors that are safe to retry on the embed endpoint.

        Embedding is idempotent (each request recomputes vectors from the same
        input), so we retry broadly: transport errors, 5xx/429, and 4xx other
        than 404. Ollama's Windows runner intermittently drops its tokenize
        endpoint and surfaces 400s whose body does not always mention
        "tokenize"; retrying recovers those without masking a missing model.
        """
        text = str(exc)
        body = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            body = getattr(resp, "text", "") or ""
        if "No connection could be made" in text or "No connection could be made" in body:
            return True
        status_code = getattr(resp, "status_code", None) or getattr(exc, "status_code", None)
        if status_code is None:
            return True
        if status_code == 404:
            return False
        return status_code == 429 or status_code >= 400


class _GeminiBackend:
    """Minimal client for `batchEmbedContents` on the Gemini API.

    Uses the same `GEMINI_API_KEY` and host as the generation LLM
    (`generativelanguage.googleapis.com`), so no new secrets are needed.
    Output dimensionality is pinned to `cfg.gemini_embed_dim` (1024) which is a
    supported value for gemini-embedding-001 (128-3072) — this keeps the Neo4j
    HNSW index dimension unchanged. Vectors are L2-normalized because
    gemini-embedding-001 does NOT auto-normalize for dimensions < 3072.
    """

    def __init__(self, cfg: Settings) -> None:
        import httpx

        self.cfg = cfg
        self._client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self.model = cfg.gemini_embed_model
        self.dim = cfg.gemini_embed_dim
        self.max_chars = cfg.gemini_embed_max_chars

    def embed(self, texts: list[str], batch_size: int = 100, task_type: str | None = None) -> list[list[float]]:
        batch_size = min(batch_size, self.cfg.gemini_embed_batch_size)
        cleaned = [t[: self.max_chars] for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        import httpx
        import time as _time
        import random as _random

        import numpy as np

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        )
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dim,
                }
                for t in batch
            ]
        }
        attempts = 8
        base_delay = 0.4
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(url, params={"key": self.cfg.gemini_api_key}, json=body)
                resp.raise_for_status()
                data = resp.json()
                embeddings = data["embeddings"]
                if len(embeddings) != len(batch):
                    raise ValueError(f"gemini returned {len(embeddings)} embeddings for {len(batch)} inputs")
                vecs = np.asarray([emb["values"] for emb in embeddings], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (vecs / norms).tolist()
            except (httpx.HTTPStatusError, httpx.TransportError, ValueError) as exc:
                if not self._is_transient(exc):
                    raise
                if attempt >= attempts:
                    raise
                delay = min(8.0, base_delay * (2 ** (attempt - 1))) * (1.0 + _random.uniform(-0.2, 0.2))
                body_hint = ""
                resp = getattr(exc, "response", None)
                if resp is not None and getattr(resp, "text", ""):
                    body_hint = f" body={resp.text[:200]!r}"
                logger.warning("Gemini embed retry %d/%d in %.3fs: %s%s", attempt, attempts, delay, exc, body_hint)
                _time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", None) or getattr(exc, "status_code", None)
        if status_code is None:
            return True
        if status_code == 400:
            body = getattr(resp, "text", "") or ""
            if "dimension" in body.lower() or "has been blocked" in body.lower():
                return False
        return status_code == 429 or status_code >= 400


class _VertexBackend:
    """gemini-embedding-001 via Vertex AI `:predict` (one text per request).

    Vertex AI exposes no batchEmbedContents for this model, so each text is a
    single predict call; embed() parallelizes with a thread pool (bounded by
    cfg.vertex_embed_concurrency). Usage is billed to the GCP project and
    covered by the $300 Free Trial credit — unlike the AI Studio Gemini API.
    NOTE (Aug 2026): NOT the deploy path — Render has no metadata server and
    the org policy blocks SA-key creation, so production uses `_GeminiBackend`
    (free-tier quota). Vertex remains a local-ADC convenience; its RETRIEVAL_
    QUERY vectors are byte-identical to the Gemini API backend (measured
    max_abs_diff = 0.0), but its RETRIEVAL_DOCUMENT vectors are NOT — never
    mix a Vertex-document index with Gemini API query vectors.
    Auth uses a service-account key (cfg.vertex_credentials: path to JSON or
    inline JSON); an access token is fetched on demand and cached until ~2min
    before expiry. task_type differentiates query vs document embeddings
    (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT), which Vertex trains for retrieval.
    """

    def __init__(self, cfg: Settings) -> None:
        import json as _json
        import pathlib
        import threading

        import httpx

        self.cfg = cfg
        self.project = cfg.vertex_project
        self.location = cfg.vertex_location
        self.model = cfg.vertex_embed_model
        self.dim = cfg.gemini_embed_dim
        self.max_chars = cfg.gemini_embed_max_chars
        self._concurrency = cfg.vertex_embed_concurrency

        self._creds, self.project = self._resolve_creds(cfg)
        if not self.project:
            raise RuntimeError(
                "VERTEX_PROJECT not set and Application Default Credentials did not "
                "resolve a project — set VERTEX_PROJECT or configure ADC "
                "(gcloud auth application-default login / GOOGLE_APPLICATION_CREDENTIALS)"
            )

        from google.auth.transport.requests import Request

        self._auth_request = Request()
        self._token_lock = threading.Lock()
        self._token: str | None = None
        self._token_expiry = 0.0
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    def _access_token(self) -> str:
        import time as _time

        now = _time.time()
        if self._token is None or now > self._token_expiry - 120:
            with self._token_lock:
                if self._token is None or now > self._token_expiry - 120:
                    self._creds.refresh(self._auth_request)
                    self._token = self._creds.token
                    self._token_expiry = self._creds.expiry.timestamp()
        return self._token

    def _resolve_creds(self, cfg: Settings) -> tuple:
        """Resolve (credentials, project) for Vertex AI.

        Priority:
          1. Explicit service-account JSON: cfg.vertex_credentials accepts an
             inline JSON string (starts with '{') or a path to a JSON file.
          2. Application Default Credentials (ADC): google.auth.default() walks
             the standard chain — GOOGLE_APPLICATION_CREDENTIALS env var, the
             gcloud ADC file (~/.config/gcloud/application_default_credentials.json),
             and finally the metadata server (Cloud Run / GCE attached SA).
             This is the recommended auth for local gcloud login + Cloud Run.
        Returns (creds, project_id); project_id may be None when the caller
        also supplied cfg.vertex_project.
        """
        import json as _json
        import pathlib

        from google.auth import default as adc_default
        from google.oauth2 import service_account

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = cfg.vertex_credentials
        if creds:
            if creds.lstrip().startswith("{"):
                info = _json.loads(creds)
            else:
                info = _json.loads(pathlib.Path(creds).read_text())
            project = cfg.vertex_project or info.get("project_id") or ""
            return service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            ), project
        creds, project = adc_default(scopes=scopes)
        return creds, project or cfg.vertex_project

    def _endpoint(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:predict"
        )

    def embed(self, texts: list[str], batch_size: int = 128, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        import concurrent.futures

        cleaned = [t[: self.max_chars] for t in texts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._concurrency) as ex:
            results = list(ex.map(lambda t: self._embed_one(t, task_type), cleaned))
        return results

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        import time as _time
        import random as _random

        import httpx
        import numpy as np

        body = {
            "instances": [{"content": text, "task_type": task_type}],
            "parameters": {"outputDimensionality": self.dim, "autoTruncate": True},
        }
        attempts = 8
        base_delay = 0.4
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(
                    self._endpoint(),
                    headers={"Authorization": f"Bearer {self._access_token()}"},
                    json=body,
                )
                resp.raise_for_status()
                values = resp.json()["predictions"][0]["embeddings"]["values"]
                vec = np.asarray(values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm == 0:
                    norm = 1.0
                return (vec / norm).tolist()
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, IndexError, ValueError) as exc:
                if not self._is_transient(exc):
                    raise
                if attempt >= attempts:
                    raise
                delay = min(8.0, base_delay * (2 ** (attempt - 1))) * (1.0 + _random.uniform(-0.2, 0.2))
                body_hint = ""
                resp = getattr(exc, "response", None)
                if resp is not None and getattr(resp, "text", ""):
                    body_hint = f" body={resp.text[:200]!r}"
                logger.warning("Vertex embed retry %d/%d in %.3fs: %s%s", attempt, attempts, delay, exc, body_hint)
                _time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """429 (quota) and 5xx are safe to retry; 4xx (bad request / auth /
        permissions / not found) will not resolve with retries and must raise."""
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", None) or getattr(exc, "status_code", None)
        if status_code is None:
            return True
        if status_code in (400, 401, 403, 404):
            return False
        return status_code == 429 or status_code >= 500


class EmbeddingService:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._backend = None
        self._backend_name: str | None = None

    def _ensure(self):
        if self._backend is not None:
            return self._backend
        with _lock:
            if self._backend is not None:
                return self._backend
            backend_kind = self.cfg.embed_backend
            if backend_kind == "ollama":
                try:
                    import httpx  # noqa: F401

                    self._backend = _OllamaBackend(self.cfg)
                    self._backend_name = "ollama"
                    logger.info("Embedding backend: ollama (%s)", self.cfg.ollama_embed_model)
                    return self._backend
                except ImportError:
                    raise RuntimeError(
                        "embedding backend 'ollama' requested but dependency 'httpx' is not installed"
                    ) from None
            if backend_kind == "gemini":
                try:
                    import httpx  # noqa: F401

                    self._backend = _GeminiBackend(self.cfg)
                    self._backend_name = "gemini"
                    logger.info("Embedding backend: gemini (%s, dim %d)", self.cfg.gemini_embed_model, self.cfg.gemini_embed_dim)
                    return self._backend
                except ImportError:
                    raise RuntimeError(
                        "embedding backend 'gemini' requested but dependency 'httpx' is not installed"
                    ) from None
            if backend_kind == "vertex":
                try:
                    import httpx  # noqa: F401
                    import google.auth  # noqa: F401

                    self._backend = _VertexBackend(self.cfg)
                    self._backend_name = "vertex"
                    logger.info(
                        "Embedding backend: vertex (%s, dim %d, concurrency %d)",
                        self.cfg.vertex_embed_model,
                        self.cfg.gemini_embed_dim,
                        self.cfg.vertex_embed_concurrency,
                    )
                    return self._backend
                except ImportError:
                    raise RuntimeError(
                        "embedding backend 'vertex' requested but dependency 'google-auth' is not installed"
                    ) from None
            try:
                from fastembed import TextEmbedding

                try:
                    self._backend = TextEmbedding(model_name=self.cfg.embed_model, cache_dir=str(self.cfg.embed_onnx_dir), threads=8)
                except TypeError:
                    self._backend = TextEmbedding(model_name=self.cfg.embed_model, cache_dir=str(self.cfg.embed_onnx_dir))
                self._backend_name = "fastembed"
                logger.info("Embedding backend: fastembed (%s)", self.cfg.embed_model)
            except ImportError:
                from sentence_transformers import SentenceTransformer

                device = "cuda" if self.cfg.embed_device == "cuda" else "cpu"
                self._backend = SentenceTransformer(self.cfg.embed_model, device=device)
                self._backend_name = "sentence-transformers"
                logger.info("Embedding backend: sentence-transformers (%s)", self.cfg.embed_model)
            return self._backend

    @property
    def backend_name(self) -> str:
        self._ensure()
        return self._backend_name or "unknown"

    def embed_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        backend = self._ensure()
        cleaned = [t[: self.cfg.embed_max_chars] for t in texts]
        if self._backend_name == "fastembed":
            vectors = [vec.tolist() for vec in backend.embed(cleaned, batch_size=self.cfg.embed_batch_size)]
        elif self._backend_name in ("ollama", "gemini", "vertex"):
            vectors = backend.embed(cleaned, batch_size=self.cfg.embed_batch_size, task_type=task_type)
        else:
            vectors = backend.encode(cleaned, batch_size=self.cfg.embed_batch_size, normalize_embeddings=True).tolist()
        # Phase 7 provenance: only vectors that EMBED BATCH just produced in
        # THIS process may be tagged with embed_backend. Registering them here
        # is what lets the store's upsert guard tell "fresh" from "read back
        # from Neo4j and re-tagged" — the exact contamination from Phase 5/6.
        register_fresh_vectors(vectors, self._backend_name or "unknown")
        return vectors

    def embed_one(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        return self.embed_batch([text], task_type=task_type)[0]

    def warm(self) -> None:
        self._ensure()


# --- async / concurrency-aware query embedding ------------------------------

def _cache_get(text: str) -> tuple[float, ...] | None:
    with _CACHE_LOCK:
        hit = _QUERY_CACHE.get(text)
        if hit is not None:
            _QUERY_CACHE.move_to_end(text)
        return hit


def _cache_put(text: str, vec: tuple[float, ...]) -> None:
    with _CACHE_LOCK:
        _QUERY_CACHE[text] = vec
        _QUERY_CACHE.move_to_end(text)
        while len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)


def _embed_uncached(text: str) -> tuple[float, ...]:
    return tuple(get_embedding_service().embed_one(text))


def _embed_semaphore(bound: int) -> asyncio.Semaphore:
    global _EMBED_SEM_BOUND
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _EMBED_SEMS.get(key)
    if sem is None or _EMBED_SEM_BOUND != bound:
        with _lock:
            sem = _EMBED_SEMS.get(key)
            if sem is None or _EMBED_SEM_BOUND != bound:
                sem = asyncio.Semaphore(bound)
                _EMBED_SEMS[key] = sem
                _EMBED_SEM_BOUND = bound
    return sem


def query_embedding(text: str) -> list[float]:
    cached = _cache_get(text)
    if cached is not None:
        return list(cached)
    vec = _embed_uncached(text)
    _cache_put(text, vec)
    return list(vec)


async def query_embedding_async(text: str, *, bound: int = 2, on_queued: Callable[[], object] | None = None) -> list[float]:
    """Concurrency-bounded query embedding for the request path.

    Cache hits return immediately (no semaphore). On a miss we gate the
    network call behind asyncio.Semaphore(bound); if the gate is contended we
    invoke `on_queued()` first so the SSE stream can emit a "queued" stage
    event instead of stalling silently. The network call itself runs in a
    thread (httpx.Client is sync) so the event loop stays free.
    """
    cached = _cache_get(text)
    if cached is not None:
        return list(cached)

    sem = _embed_semaphore(bound)
    if sem.locked() and on_queued is not None:
        await on_queued()  # type: ignore[misc]
    async with sem:
        cached = _cache_get(text)
        if cached is not None:
            return list(cached)
        vec = await asyncio.to_thread(_embed_uncached, text)
        _cache_put(text, vec)
        return list(vec)