"""Local multilingual embeddings — Qwen3-Embedding via Ollama (dense-only).

Backend order is driven by `cfg.embed_backend`:
  - "ollama" (default): POST /api/embed on a local Ollama server. Dense vectors
    only — the lexical/BM25 arm is served by Neo4j's Lucene fulltext index.
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

    def embed(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
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
                    logger.warning("httpx missing — falling back from ollama embedding backend")
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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        backend = self._ensure()
        cleaned = [t[: self.cfg.embed_max_chars] for t in texts]
        if self._backend_name == "fastembed":
            return [vec.tolist() for vec in backend.embed(cleaned, batch_size=self.cfg.embed_batch_size)]
        if self._backend_name == "ollama":
            return backend.embed(cleaned, batch_size=self.cfg.embed_batch_size)
        return backend.encode(cleaned, batch_size=self.cfg.embed_batch_size, normalize_embeddings=True).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

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