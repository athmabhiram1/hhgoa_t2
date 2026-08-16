"""Fast path generation.

Two answer strategies, both latency-aware:
  * extractive — no LLM; a sentence from the top passage that best matches the
    question's keywords is returned as the answer. Sub-10ms.
  * llm — streamed-free structured JSON from the provider chain (Gemini →
    Ollama → Groq → OpenAI), parsed defensively.

`mode` is decided by the caller (harness/pipeline): extractive when the user
asked for max speed, llm otherwise. Extractive is always computed first and can
be streamed instantly while the LLM answer warms up.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..config import Settings
from ..core.models import Answer, Citation, RetrievedPassage, RetrievalResult
from ..core.providers import LLMClient
from ..core.tracing import span
from .prompts import GENERATION_SCHEMA, SYSTEM_GENERATION, build_generation_prompt

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?।।॥؟])[\s\n]+")


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[\w\u0900-\u0FFF\u0A00-\u0D7F\u0B80-\u0BFF\u0C00-\u0CFF]+", text.lower()) if len(t) > 1}


def _best_sentence(question: str, passage: str) -> str:
    q_tokens = _tokenize(question)
    sentences = [s.strip() for s in _SENT_SPLIT.split(passage) if s.strip()]
    if not sentences:
        return passage[:600]
    if not q_tokens:
        return sentences[0]
    scored = [(len(_tokenize(s) & q_tokens), s) for s in sentences]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else sentences[0]


def extractive_answer(query: str, retrieval: RetrievalResult) -> Answer | None:
    if not retrieval.candidates:
        return None
    top = retrieval.candidates[0]
    text = _best_sentence(query, top.text)
    return Answer(
        text=text,
        mode="extractive",
        citations=[Citation(passage_id=top.id, text=top.text, language_code=top.language_code, score=top.score)],
        grounding_score=retrieval.grounding_score,
        faithfulness=1.0,
        confidence=min(0.99, retrieval.grounding_score + 0.15),
    )


class FastPathLLM:
    def __init__(self, cfg: Settings, client: LLMClient) -> None:
        self.cfg = cfg
        self.client = client

    async def generate(self, query: str, retrieval: RetrievalResult, *, lang_hint: str | None = None, top_n: int = 6) -> Answer:
        candidates = retrieval.candidates[:top_n]
        prompt = build_generation_prompt(
            query,
            [{"text": c.text, "language_code": c.language_code} for c in candidates],
            query_lang_hint=lang_hint,
        )
        with span("gen.llm"):
            data: dict[str, Any] = await self.client.complete_json(
                prompt, system=SYSTEM_GENERATION, schema_fields=set(GENERATION_SCHEMA.keys())
            )
        answer_text = str(data.get("answer", "")).strip()
        unsupported = bool(data.get("unsupported", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)

        citations = [
            Citation(passage_id=str(c.get("passage_id", "")), text=str(c.get("text", "")), language_code=candidates[0].language_code if candidates else "")
            for c in data.get("citations", [])[:3]
            if isinstance(c, dict)
        ]

        if unsupported or not answer_text:
            return Answer(
                text="मुझे इसका उत्तर देने के लिए पर्याप्त संदर्भ नहीं मिला।" if lang_hint and lang_hint.startswith("hi") else "I couldn't find enough grounded context to answer that.",
                mode="refusal",
                citations=citations,
                grounding_score=retrieval.grounding_score,
                confidence=0.0,
                refusal_reason="unsupported by retrieved context",
            )

        return Answer(
            text=answer_text,
            mode="llm",
            citations=citations,
            grounding_score=retrieval.grounding_score,
            confidence=confidence,
        )