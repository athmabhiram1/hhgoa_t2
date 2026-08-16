"""LightRAG deep path (Tier 2).

LightRAG builds a knowledge graph via LLM entity-relation extraction and
answers with graph-aware query modes (naive/hybrid/mix). That costs LLM time at
QUERY time, so it is NOT the default — see CONTEXT.md. It is enabled for
relational/multi-hop questions and powers the knowledge-graph visual.

Indexing cost is index-time only (Ollama on the RTX 5050). We index a curated
slice (default 300 queries) as pseudo-documents; the full 150-200k-passage
corpus remains on the Tier-1 Neo4j index. If `lightrag-hku` is not installed or
Neo4j is unreachable, every method degrades gracefully and the pipeline uses
the fast path only — the app never crashes on a missing optional dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import Settings

logger = logging.getLogger(__name__)

_AVAILABLE = True
try:
    from lightrag import LightRAG, QueryParam  # noqa: F401
    from lightrag.kg.neo4j_impl import Neo4JStorage  # noqa: F401
except ImportError:  # pragma: no cover
    _AVAILABLE = False
    logger.warning("lightrag-hku not installed — LightRAG deep path disabled")


class LightRAGDeepEngine:
    def __init__(self, cfg: Settings, llm_func: Callable | None = None, embedding_func: Callable | None = None) -> None:
        self.cfg = cfg
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self._rag: Any = None
        self._ready = False
        self._disabled = not _AVAILABLE

    @property
    def available(self) -> bool:
        return not self._disabled and self._ready

    async def initialize(self) -> None:
        if self._disabled:
            return
        try:
            from lightrag import LightRAG
            from lightrag.kg.neo4j_impl import Neo4JStorage

            self._rag = LightRAG(
                working_dir=str(self.cfg.embed_onnx_dir.parent / "lightrag_storage"),
                workspace="deep",
                llm_model_func=self.llm_func or (lambda prompt, **kw: "ERROR"),
                llm_model_name=self.cfg.ollama_model,
                embedding_func=self.embedding_func or (lambda texts: [[0.0] * self.cfg.embed_dim] * len(texts)),
                graph_storage="Neo4JStorage",
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                doc_status_storage="JsonDocStatusStorage",
                entity_extract_max_gleaning=0,
            )
            await self._rag.initialize_storages()
            self._ready = True
            logger.info("LightRAG deep engine initialized (Neo4j graph + Ollama)")
        except Exception as exc:  # noqa: BLE001
            logger.error("LightRAG init failed — deep path disabled: %s", exc)
            self._disabled = True

    async def index_documents(self, documents: list[str]) -> int:
        if self._disabled or self._rag is None:
            return 0
        try:
            for doc in documents:
                await self._rag.ainsert(input=[doc])
            await self._rag.finalize_storages()
            return len(documents)
        except Exception as exc:  # noqa: BLE001
            logger.error("LightRAG indexing failed: %s", exc)
            return 0

    async def query(self, question: str, *, mode: str = "naive", top_k: int = 12) -> str:
        if self._disabled or self._rag is None:
            raise RuntimeError("LightRAG deep path unavailable")
        try:
            result = await self._rag.aquery(question, param=QueryParam(mode=mode, only_need_context=False, top_k=top_k))
            return str(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("LightRAG query failed: %s", exc)
            raise

    async def close(self) -> None:
        if self._rag is not None:
            try:
                await self._rag.finalize_storages()
            except Exception:  # noqa: BLE001
                pass