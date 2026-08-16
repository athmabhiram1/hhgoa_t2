"""The request pipeline — the single place every query flows through.

Stages (each instrumented with a span; latency is a measured contract):
  audio (optional) -> STT -> intent/router -> safety -> off-topic
    -> retrieval (fast) -> grounding -> generate -> faithfulness
  or, for relational queries, the LightRAG deep path instead of fast retrieval.

Guardrail refusals are ordinary PipelineResult outputs (mode="refusal") with a
reason — never exceptions.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from ..config import Settings
from ..core.models import Answer, GuardVerdict, PipelineResult, QueryIntent, StageSpan, Transcript
from ..core.providers import LLMClient
from ..guardrails.faithfulness import FaithfulnessGuard
from ..guardrails.grounding import GroundingGuard
from ..guardrails.off_topic import OffTopicGuard
from ..guardrails.safety import SafetyGuard
from ..rag.fast_path import FastPathLLM, extractive_answer
from ..rag.lightrag_engine import LightRAGDeepEngine
from ..rag.router import QueryRouter
from ..retrieval.neo4j_store import Neo4jStore
from ..retrieval.service import RetrievalService
from ..stt.providers import STTManager

logger = logging.getLogger(__name__)

REFUSAL_TEXTS = {
    "unsafe": "I can't help with that.",
    "off_topic": "That doesn't look like a question I can answer from the corpus.",
    "low_grounding": "मुझे इसका उत्तर देने के लिए पर्याप्त संदर्भ नहीं मिला।",
}


class VakRagPipeline:
    def __init__(self, cfg: Settings, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client
        self.store: Neo4jStore | None = None
        self.stt = STTManager(cfg)
        self.router = QueryRouter(cfg, client)
        self.retrieval: RetrievalService | None = None
        self.gen = FastPathLLM(cfg, client) if client is not None else None
        self.safety = SafetyGuard(cfg, client)
        self.offtopic = OffTopicGuard(cfg, client)
        self.grounding = GroundingGuard(cfg)
        self.faithfulness = FaithfulnessGuard(cfg, client)
        self.deep = LightRAGDeepEngine(cfg, llm_func=getattr(client, "lightrag_llm_func", None))

    def bind_retrieval(self, store: Neo4jStore) -> None:
        self.store = store
        self.retrieval = RetrievalService(self.cfg, store)

    # ------------------------------------------------------------------ run
    async def run_transcript(
        self,
        text: str,
        *,
        lang: str | None = None,
        mode: str = "auto",
        query_type: str | None = None,
        on_embed_queued=None,
    ) -> PipelineResult:
        request_id = uuid.uuid4().hex[:12]
        spans: list[StageSpan] = []
        t0 = time.perf_counter()

        async def record(name: str, coro, **meta) -> object:
            start = time.perf_counter()
            try:
                result = await coro
                spans.append(StageSpan(name=name, duration_ms=round((time.perf_counter() - start) * 1000, 2), ok=True, detail=meta.get("detail")))
                return result
            except Exception as exc:  # noqa: BLE001
                spans.append(StageSpan(name=name, duration_ms=round((time.perf_counter() - start) * 1000, 2), ok=False, detail=str(exc)))
                raise

        transcript = Transcript(text=text, language_code=lang or "auto", provider="text")

        intent = await record("intent", self.router.route(text, stt_lang=lang))

        if intent.needs_graph and self.deep.available:
            return await self._run_deep(request_id, transcript, intent, spans, t0)

        safety = await record("safety", self.safety.check(text))
        if not safety.allow:
            return self._refusal_result(request_id, transcript, intent, safety, spans, t0)

        off_topic = await record("off_topic", self.offtopic.check(text))
        if not off_topic.allow:
            return self._refusal_result(request_id, transcript, intent, off_topic, spans, t0)

        if self.retrieval is None or self.store is None:
            raise RuntimeError("retrieval not bound — call bind_retrieval() first")
        # LOCKED DECISION (Phase 5): live retrieval is UNFILTERED by default.
        # The heuristic query_type classifier (~42% accurate) must NOT drive
        # filtering — it zeroed results for ~1/3 of queries. query_type is now
        # an explicit opt-in from the caller (POST /v1/ask body).
        retrieval = await record(
            "retrieval",
            self.retrieval.retrieve(
                text,
                lang=intent.language_code if intent.language_code != "auto" else None,
                query_type=query_type,
                on_embed_queued=on_embed_queued,
            ),
        )
        grounding = await record("grounding", self.grounding.check(retrieval))
        if not grounding.allow:
            return self._refusal_result(request_id, transcript, intent, grounding, spans, t0, retrieval=retrieval)

        if mode == "extractive" or self.gen is None:
            answer = await record("generate.extractive", asyncio.to_thread(extractive_answer, text, retrieval)) or self._no_answer()
        else:
            answer = await record("generate", self.gen.generate(text, retrieval, lang_hint=intent.language_code))

        faithful = await record("faithfulness", self.faithfulness.check(answer, retrieval.candidates))
        if not faithful.allow:
            answer = self._unfaithful_refusal(answer)
        answer.latency_ms = int((time.perf_counter() - t0) * 1000)

        return PipelineResult(
            request_id=request_id,
            transcript=transcript,
            intent=intent,
            verdict=GuardVerdict(allow=True),
            retrieval=retrieval,
            answer=answer,
            spans=spans,
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    async def run_audio(self, audio_bytes: bytes) -> PipelineResult:
        transcript = await self.stt.transcribe(audio_bytes)
        return await self.run_transcript(transcript.text, lang=transcript.language_code)

    # ------------------------------------------------------------------ deep
    async def _run_deep(self, request_id: str, transcript: Transcript, intent: QueryIntent, spans: list[StageSpan], t0: float) -> PipelineResult:
        with_spans = spans[:]
        start = time.perf_counter()
        answer_text = await self.deep.query(intent.text, mode="mix")
        with_spans.append(StageSpan(name="deep.path", duration_ms=round((time.perf_counter() - start) * 1000, 2), ok=True))
        answer = Answer(text=answer_text, mode="llm", grounding_score=0.8, confidence=0.8)
        answer.latency_ms = int((time.perf_counter() - t0) * 1000)
        return PipelineResult(
            request_id=request_id,
            transcript=transcript,
            intent=intent,
            verdict=GuardVerdict(allow=True),
            retrieval=None,
            answer=answer,
            spans=with_spans,
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ------------------------------------------------------------------ misc
    def _no_answer(self) -> Answer:
        return Answer(text="No grounded passage found.", mode="refusal", confidence=0.0, refusal_reason="no passages")

    def _unfaithful_refusal(self, answer: Answer) -> Answer:
        return Answer(
            text="The draft answer wasn't fully supported by the sources.",
            mode="refusal",
            confidence=0.0,
            refusal_reason="unfaithful",
            citations=answer.citations,
        )

    def _refusal_result(
        self, request_id: str, transcript: Transcript, intent: QueryIntent, verdict: GuardVerdict, spans: list[StageSpan], t0: float, retrieval=None
    ) -> PipelineResult:
        text = REFUSAL_TEXTS.get(verdict.category) or "I can't answer that."
        answer = Answer(
            text=text,
            mode="refusal",
            grounding_score=retrieval.grounding_score if retrieval else 0.0,
            confidence=0.0,
            refusal_reason=verdict.reason,
        )
        answer.latency_ms = int((time.perf_counter() - t0) * 1000)
        return PipelineResult(
            request_id=request_id,
            transcript=transcript,
            intent=intent,
            verdict=verdict,
            retrieval=retrieval,
            answer=answer,
            spans=spans,
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )