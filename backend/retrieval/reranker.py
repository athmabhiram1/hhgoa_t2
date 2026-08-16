"""Optional local reranker (BAAI/bge-reranker-v2-m3).

Reranking is off by default (`RETRIEVAL_RERANK=none`) because it adds ~20-80ms
on CPU. When enabled it runs on the fused top-k candidates only, keeping the
latency cost bounded. The reranker is a cross-encoder: query x passage -> score.
"""

from __future__ import annotations

import logging
import threading

from ..config import Settings
from ..core.models import RetrievedPassage

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class Reranker:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None

    def _ensure(self):
        if self._model is None:
            with _lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.cfg.rerank_model, device=self.cfg.rerank_device)
                    logger.info("Reranker loaded: %s", self.cfg.rerank_model)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedPassage], topk: int = 6) -> list[RetrievedPassage]:
        if not candidates:
            return candidates
        model = self._ensure()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        for c, s in ranked:
            c.rerank_score = float(s)
        return [c for c, _ in ranked[:topk]]