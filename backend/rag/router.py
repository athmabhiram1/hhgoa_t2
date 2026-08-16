"""Query router — intent classification and fast/deep tier selection.

Heuristics run first (sub-millisecond, no network): language code, query_type
guess, and graph-need guess. An optional LLM judge refines uncertain cases when
`GUARD_LLM_JUDGE=on`. The router never blocks; it only routes.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..core.models import QueryIntent
from ..core.providers import LLMClient
from ..core.tracing import span
from .prompts import INTENT_SCHEMA, SYSTEM_INTENT

_GRAPH_HINTS = re.compile(
    r"(compare|comparison|difference between|relationship|relation between|how are .* and .* related|"
    r"what happened after|impact of .* on|tulna|antar|sambandh|प्रभाव|तुलना|அ) ",
    re.IGNORECASE,
)

_NUMBER_HINTS = re.compile(
    r"(how many|how much|when did|how old|what year|kitne|kitni|kitan|kitano|"
    r"कितने|कितना|எத்தனை|ఎంత|کتنے|কিমান|কত|কটি|کيترا|గా)",
    re.IGNORECASE,
)

_PERSON_HINTS = re.compile(r"(who invented|who founded|who is|who was|kisne|किसने|யார்|ఎవరు|کون|কে)", re.IGNORECASE)

_LOCATION_HINTS = re.compile(r"(where is|where did|kahan|कहाँ|எங்கே|ఎక్కడ|کہاں|কোথায়|ক'ত)", re.IGNORECASE)

_ENTITY_HINTS = re.compile(r"(what is a|what is the|kya hai|क्या है|எது|ఏమిటి|کیا|কি|কোন)", re.IGNORECASE)


def _guess_query_type(text: str) -> str:
    if _NUMBER_HINTS.search(text):
        return "NUMERIC"
    if _PERSON_HINTS.search(text):
        return "PERSON"
    if _LOCATION_HINTS.search(text):
        return "LOCATION"
    if _ENTITY_HINTS.search(text):
        return "ENTITY"
    if _GRAPH_HINTS.search(text):
        return "DESCRIPTION"
    return "DESCRIPTION"


def _detect_lang(text: str) -> str | None:
    # Unambiguous script -> ISO-639-1 (matching DB `lang` property storage).
    # Ambiguous script blocks are deliberately NOT mapped here:
    #   - U+0900-U+097F Devanagari -> hi/mr/ne/sa (indistinguishable by script)
    #   - U+0980-U+09FF Bengali block -> bn/as (shared script)
    # Returning None lets the caller fall back to the STT-provided lang or
    # "auto" (full-pool, language-agnostic search — same contract the eval uses).
    scripts: list[tuple[str, str]] = [
        ("[\u0B80-\u0BFF]", "ta"),
        ("[\u0C00-\u0C7F]", "te"),
        ("[\u0C80-\u0CFF]", "kn"),
        ("[\u0A00-\u0A7F]", "pa"),
        ("[\u0A80-\u0AFF]", "gu"),
        ("[\u0D00-\u0D7F]", "ml"),
        ("[\u0600-\u06FF]", "ur"),
    ]
    for pattern, code in scripts:
        if re.search(pattern, text):
            return code
    return None


class QueryRouter:
    def __init__(self, cfg: Settings, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client

    async def route(self, query: str, *, stt_lang: str | None = None) -> QueryIntent:
        qtype = _guess_query_type(query)
        lang = _detect_lang(query) or stt_lang or "auto"
        needs_graph = bool(_GRAPH_HINTS.search(query))

        if self.client is not None and self.cfg.guard_llm_judge:
            with span("router.llm"):
                try:
                    data = await self.client.complete_json(
                        f"QUESTION: {query}",
                        system=SYSTEM_INTENT,
                        schema_fields=set(INTENT_SCHEMA.keys()),
                        order=["gemini", "ollama"],
                    )
                    qtype = str(data.get("query_type", qtype)).upper()
                    needs_graph = bool(data.get("needs_graph", needs_graph))
                    if data.get("language_code") and lang == "auto":
                        lang = str(data["language_code"])
                except Exception:  # noqa: BLE001
                    pass

        return QueryIntent(text=query, language_code=lang, query_type=qtype, needs_graph=needs_graph, confidence=1.0 if not self.cfg.guard_llm_judge else 0.7)