"""Grounding guard: the pipeline must KNOW WHEN NOT TO ANSWER.

If the best fused score (grounding_score) is below the threshold, the answer
path is a refusal ("I don't have enough context"). This is a pure numerical
gate — no LLM, sub-millisecond.
"""

from __future__ import annotations

from ..config import Settings
from ..core.models import GuardVerdict, RetrievalResult


class GroundingGuard:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

    async def check(self, retrieval: RetrievalResult) -> GuardVerdict:
        if not self.cfg.guard_grounding:
            return GuardVerdict(allow=True)
        if not retrieval.candidates:
            return GuardVerdict(
                allow=False,
                category="low_grounding",
                reason="no passages retrieved",
                is_refusal=True,
                score=1.0,
            )
        if retrieval.grounding_score < self.cfg.grounding_threshold:
            return GuardVerdict(
                allow=False,
                category="low_grounding",
                reason=f"best grounding {retrieval.grounding_score:.3f} below threshold {self.cfg.grounding_threshold}",
                is_refusal=True,
                score=round(1.0 - retrieval.grounding_score, 2),
            )
        return GuardVerdict(allow=True, score=round(retrieval.grounding_score, 2))