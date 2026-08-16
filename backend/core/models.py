"""Typed pipeline models — the I/O contract between every stage.

No ad-hoc dicts cross stage boundaries. Every stage accepts and emits these
models, and every stage records a latency span (see harness/pipeline.py).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LanguageCode = str
QueryType = Literal["DESCRIPTION", "ENTITY", "NUMERIC", "PERSON", "LOCATION", "MISC", "UNKNOWN"]


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
class Transcript(BaseModel):
    text: str
    language_code: LanguageCode = "auto"
    provider: str = "unknown"          # sarvam | whisper
    confidence: float | None = None    # whisper language_probability / sarvam language_probability
    audio_duration_ms: int = 0
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
GuardCategory = Literal[
    "unsafe", "off_topic", "low_grounding", "unfaithful", "ok", "refusal_unknown"
]

class GuardVerdict(BaseModel):
    allow: bool
    category: GuardCategory = "ok"
    reason: str = ""
    score: float = 0.0                # 0..1 severity / relevance
    is_refusal: bool = False          # True => pipeline returns a refusal, not an answer


# ---------------------------------------------------------------------------
# Query understanding / routing
# ---------------------------------------------------------------------------
class QueryIntent(BaseModel):
    text: str
    language_code: LanguageCode = "auto"
    query_type: QueryType = "UNKNOWN"
    needs_graph: bool = False         # escalate to LightRAG deep path
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class RetrievedPassage(BaseModel):
    id: str
    text: str
    language_code: LanguageCode = ""
    score: float = 0.0
    rerank_score: float | None = None
    source: str = "vector"            # vector | bm25 | graph | hybrid
    query_id: str | None = None
    query_type: str | None = None
    position: int | None = None
    is_selected: int | None = None    # ground-truth relevance label (0/1) if known
    namespace: str = ""

    @property
    def display_source(self) -> str:
        return self.source


class RetrievalResult(BaseModel):
    query: str
    candidates: list[RetrievedPassage] = Field(default_factory=list)
    grounding_score: float = 0.0      # calibrated max similarity
    latency_ms: int = 0
    breakdown_ms: dict[str, int | str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    passage_id: str
    text: str
    language_code: LanguageCode = ""
    score: float = 0.0


class Answer(BaseModel):
    text: str
    mode: Literal["extractive", "llm", "refusal"] = "llm"
    citations: list[Citation] = Field(default_factory=list)
    grounding_score: float = 0.0
    faithfulness: float = 1.0         # 0..1
    confidence: float = 0.0
    refusal_reason: str | None = None
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Telemetry envelope
# ---------------------------------------------------------------------------
class StageSpan(BaseModel):
    name: str
    duration_ms: float
    ok: bool = True
    detail: str | None = None


class PipelineResult(BaseModel):
    request_id: str
    transcript: Transcript
    intent: QueryIntent
    verdict: GuardVerdict
    retrieval: RetrievalResult | None = None
    answer: Answer
    spans: list[StageSpan] = Field(default_factory=list)
    total_ms: float = 0.0

    @property
    def stage_latency(self) -> dict[str, float]:
        return {s.name: s.duration_ms for s in self.spans}