"""Faithfulness guard: is the generated answer fully supported by the retrieved
context?

Default is a fast, free-check heuristic: a refusal/LLM answer is faithful if its
citations exist and, when a passage is referenced, the answer shares terms with
it. With `GUARD_LLM_JUDGE=on` an LLM judge scores faithfulness precisely.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..core.models import Answer, GuardVerdict, RetrievedPassage
from ..core.providers import LLMClient
from .prompts import FAITHFULNESS_SCHEMA, SYSTEM_FAITHFULNESS

_TOKEN = re.compile(r"[\w\u0900-\u0FFF\u0A00-\u0D7F\u0B80-\u0BFF\u0C00-\u0CFF]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


class FaithfulnessGuard:
    def __init__(self, cfg: Settings, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client

    async def check(self, answer: Answer, retrieval_passages: list[RetrievedPassage]) -> GuardVerdict:
        if answer.mode == "refusal":
            return GuardVerdict(allow=True, score=1.0)
        if not self.cfg.guard_faithfulness:
            return GuardVerdict(allow=True)
        if not answer.citations:
            return GuardVerdict(allow=False, category="unfaithful", reason="no citations on a generated answer", score=1.0)

        cited = {c.passage_id for c in answer.citations}
        known = {p.id for p in retrieval_passages}
        if not cited.issubset(known):
            return GuardVerdict(allow=False, category="unfaithful", reason="cites passages outside retrieved set", score=0.9)

        # Extractive answers are verbatim sentences extracted from a retrieved
        # passage (citation[0]) — faithful by construction. The token-overlap
        # heuristic below is meaningless here: a short extracted sentence will
        # trivially share <2 tokens with its full passage and be false-refused.
        if answer.mode == "extractive":
            top_cite = answer.citations[0].text
            overlap = len(_tokens(answer.text) & _tokens(top_cite))
            return GuardVerdict(allow=True, score=min(1.0, 0.5 + 0.05 * overlap))

        if self.client is not None and self.cfg.guard_llm_judge:
            context = "\n".join(f"[{p.id}] {p.text[:500]}" for p in retrieval_passages[:5])
            data = await self.client.complete_json(
                f"ANSWER: {answer.text}\n\nCONTEXT:\n{context}",
                system=SYSTEM_FAITHFULNESS,
                schema_fields={"faithful", "score", "reason"},
            )
            faithful = bool(data.get("faithful", True))
            score = float(data.get("score", 1.0))
            if not faithful or score < 0.5:
                return GuardVerdict(allow=False, category="unfaithful", reason=str(data.get("reason", "LLM faithfulness check")), score=round(1.0 - score, 2))
            return GuardVerdict(allow=True, score=round(score, 2))

        # Heuristic: overlap between answer and its top citation.
        top_cite = answer.citations[0].text
        overlap = len(_tokens(answer.text) & _tokens(top_cite))
        if overlap < 2:
            return GuardVerdict(allow=False, category="unfaithful", reason="answer shares almost no tokens with its citation", score=0.8)
        return GuardVerdict(allow=True, score=min(1.0, 0.5 + 0.05 * overlap))