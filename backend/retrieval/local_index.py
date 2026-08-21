"""Local in-memory ANN index for the latency-first fast path (Tier 1).

The stored Neo4j vectors live in gemini-embedding-001 (Vertex) space — a
local query embedder cannot search against them, so the fast path must be
self-consistent: the corpus is re-embedded ONCE with a local model
(BAAI/bge-m3, which covers all 14 Indic languages) and held in RAM as a
normalized numpy matrix. Query time = local embed + brute-force cosine, which
for ~31k x 1024 dims is <15ms — well inside the 200ms budget without any ANN
dependency.

The index is built from a READ-ONLY Neo4j text export (fetch_namespace_texts),
never the stored embeddings (Phase-7 contamination contract). Persisted to
`data/fastpath/` (matrix.npy + meta.json + model.txt) so a restart does not
re-embed the corpus.

This is the 200ms-compliant output path; the Vertex+Neo4j+RRF+grounding+LLM
pipeline streams the full answer afterward as progressive enhancement.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..core.models import RetrievedPassage, RetrievalResult
from ..core.tracing import span
from .neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)


class LocalFastIndex:
    """In-memory brute-force cosine index over locally-re-embedded chunk texts."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None
        self._mat: np.ndarray | None = None
        self._meta: list[dict] = []

    # ------------------------------------------------------------------ build
    async def build(self, store: Neo4jStore) -> int:
        """Read-only Neo4j text export -> local re-embed -> in-memory matrix.

        Returns the number of chunks indexed. Persists to fast_path_index_dir
        so `load()` can restore it on the next process start.
        """
        rows: list[dict] = []
        for ns in self.cfg.fast_path_namespaces:
            rows.extend(await store.fetch_namespace_texts(ns))
        if not rows:
            raise RuntimeError("fast-path build: no chunk texts exported from Neo4j")
        rows.sort(key=lambda r: str(r["chunk_id"]))
        texts = [r["text"] for r in rows]
        model = self._load_model()
        t0 = time.perf_counter()
        with span("fastpath.build_embed"):
            vecs = model.encode(
                texts,
                batch_size=self.cfg.fast_path_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        logger.info(
            "fast-path: embedded %d chunks in %.1fs (model=%s device=%s)",
            len(texts),
            time.perf_counter() - t0,
            self.cfg.fast_path_model,
            self.cfg.fast_path_device,
        )
        self._mat = np.asarray(vecs, dtype=np.float32)
        self._meta = [
            {
                "chunk_id": str(r["chunk_id"]),
                "text": r["text"],
                "lang": r.get("lang") or "",
                "namespace": r["namespace"],
                "query_id": r.get("query_id"),
                "query_type": r.get("query_type"),
            }
            for r in rows
        ]
        self._save()
        return len(self._meta)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            t0 = time.perf_counter()
            dtype = torch.float16 if self.cfg.fast_path_device == "cuda" else torch.float32
            self._model = SentenceTransformer(
                self.cfg.fast_path_model,
                device=self.cfg.fast_path_device,
                model_kwargs={"torch_dtype": dtype},
            )
            logger.info(
                "fast-path: loaded %s on %s in %.1fs",
                self.cfg.fast_path_model,
                self.cfg.fast_path_device,
                time.perf_counter() - t0,
            )
        return self._model

    # ------------------------------------------------------------- persist
    def _save(self) -> None:
        if self._mat is None:
            return
        out = Path(self.cfg.fast_path_index_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "matrix.npy", self._mat)
        (out / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
        )
        (out / "model.txt").write_text(
            f"{self.cfg.fast_path_model}\n{self.cfg.fast_path_device}\n", encoding="utf-8"
        )

    def load(self) -> bool:
        """Load a previously-built index from disk. False if absent/mismatched."""
        out = Path(self.cfg.fast_path_index_dir)
        mat_path, meta_path, model_path = out / "matrix.npy", out / "meta.json", out / "model.txt"
        if not (mat_path.exists() and meta_path.exists()):
            return False
        if model_path.exists():
            saved_model = model_path.read_text(encoding="utf-8").strip().splitlines()
            if saved_model and saved_model[0] != self.cfg.fast_path_model:
                logger.warning(
                    "fast-path: disk index was built with %r but config wants %r — rebuild",
                    saved_model[0],
                    self.cfg.fast_path_model,
                )
                return False
        self._mat = np.load(mat_path)
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("fast-path: loaded index %d chunks from %s", len(self._meta), out)
        return True

    # ------------------------------------------------------------- query
    @property
    def ready(self) -> bool:
        return self._mat is not None and bool(self._meta)

    @property
    def size(self) -> int:
        return len(self._meta)

    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        """Local embed + brute-force cosine over the in-memory matrix.

        Returns a RetrievalResult whose candidates carry raw_cosine=score so
        the extractive answer and grounding semantics stay consistent (the
        0.78 gate is NOT applied here — the fast path has its own
        fast_path_grounding_floor, because the local cosine scale differs from
        the calibrated gemini scale).
        """
        if not self.ready:
            raise RuntimeError("fast-path index not loaded — run build_fastpath first")
        t0 = time.perf_counter()
        model = self._load_model()
        qv = model.encode([query], normalize_embeddings=True)[0]
        sims = self._mat @ qv
        topk = k or self.cfg.fast_path_topk
        topk = min(topk, len(sims))
        order = np.argsort(-sims)[:topk]
        candidates: list[RetrievedPassage] = []
        for i in order:
            m = self._meta[i]
            qid = m.get("query_id")
            candidates.append(
                RetrievedPassage(
                    id=m["chunk_id"],
                    text=m["text"],
                    language_code=m.get("lang") or "",
                    score=float(sims[i]),
                    raw_cosine=float(sims[i]),
                    source="fastpath",
                    query_id=str(qid) if qid is not None else None,
                    query_type=m.get("query_type"),
                    namespace=m.get("namespace"),
                )
            )
        grounding = max(float(sims[i]) for i in order) if len(order) else 0.0
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return RetrievalResult(
            query=query,
            candidates=candidates,
            grounding_score=round(grounding, 4),
            latency_ms=latency_ms,
            breakdown_ms={"fastpath": "local", "embed_search_ms": latency_ms},
        )