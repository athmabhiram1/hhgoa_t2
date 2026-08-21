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
from ..retrieval.local_index import LocalFastIndex
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
        self.fast: LocalFastIndex | None = None
        self.gen = FastPathLLM(cfg, client) if client is not None else None
        self.safety = SafetyGuard(cfg, client)
        self.offtopic = OffTopicGuard(cfg, client)
        self.grounding = GroundingGuard(cfg)
        self.faithfulness = FaithfulnessGuard(cfg, client)
        self.deep = LightRAGDeepEngine(cfg, llm_func=getattr(client, "lightrag_llm_func", None))

    def bind_retrieval(self, store: Neo4jStore) -> None:
        self.store = store
        self.retrieval = RetrievalService(self.cfg, store)

    def bind_fast_path(self, index: LocalFastIndex) -> None:
        """Attach the local in-memory fast path (latency-first Tier 1).

        Reads cfg.fast_path_enabled at request time; a loaded index with
        fast_path_enabled=True routes mode=extractive queries entirely through
        the local path and streams a `quick` answer in auto mode while the full
        Vertex+Neo4j pipeline runs as progressive enhancement.
        """
        self.fast = index

    # ------------------------------------------------------------------ run
    async def run_transcript(
        self,
        text: str,
        *,
        lang: str | None = None,
        mode: str = "auto",
        query_type: str | None = None,
        on_embed_queued=None,
        on_fast=None,
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

        # ---- latency-first fast path (Tier 1) -------------------------------
        fast_enabled = self.cfg.fast_path_enabled and self.fast is not None and self.fast.ready

        if mode == "extractive" and fast_enabled:
            return await self._run_fast(request_id, transcript, intent, spans, t0, text)

        if self.retrieval is None or self.store is None:
            raise RuntimeError("retrieval not bound — call bind_retrieval() first")

        # Auto mode: run the local fast path in the background and stream its
        # extractive answer immediately (200ms-compliant) while the full
        # Vertex+Neo4j+RRF+grounding+LLM pipeline proceeds as progressive
        # enhancement. Guardrails already ran, so a quick answer here is safe.
        fast_task = None
        if fast_enabled and on_fast is not None:
            fast_task = asyncio.create_task(self._run_fast(request_id, transcript, intent, spans, t0, text, report=on_fast))

            def _fast_done(t: asyncio.Task) -> None:  # best-effort: never crash the request
                if not t.cancelled():
                    exc = t.exception()
                    if exc is not None:
                        logger.warning("fast path failed (best-effort, ignored): %s", exc)

            fast_task.add_done_callback(_fast_done)

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
            if fast_task is not None:
                fast_task.cancel()
            return self._refusal_result(request_id, transcript, intent, grounding, spans, t0, retrieval=retrieval)

        if mode == "extractive" or self.gen is None:
            answer = await record("generate.extractive", asyncio.to_thread(extractive_answer, text, retrieval)) or self._no_answer()
        else:
            answer = await record("generate", self.gen.generate(text, retrieval, lang_hint=intent.language_code))

        if fast_task is not None:
            fast_task.cancel()

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

    async def _run_fast(
        self,
        request_id: str,
        transcript: Transcript,
        intent: QueryIntent,
        spans: list[StageSpan],
        t0: float,
        text: str,
        report=None,
    ) -> PipelineResult:
        """Local-only fast path: bge-m3 embed + in-memory cosine + extractive.

        Returns a PipelineResult whose answer is the 200ms-compliant extractive
        span. When `report` is provided (auto mode) the answer is streamed via
        the callback as a `quick` SSE event and this result is discarded —
        callers must still await/cancel the task.
        """
        start = time.perf_counter()
        try:
            retrieval = await asyncio.to_thread(self.fast.search, text)
            spans.append(StageSpan(name="fast.retrieve", duration_ms=round((time.perf_counter() - start) * 1000, 2), ok=True))
        except Exception as exc:  # noqa: BLE001
            spans.append(StageSpan(name="fast.retrieve", duration_ms=round((time.perf_counter() - start) * 1000, 2), ok=False, detail=str(exc)))
            raise

        g_start = time.perf_counter()
        answer = extractive_answer(text, retrieval) or self._no_answer()
        spans.append(StageSpan(name="fast.answer", duration_ms=round((time.perf_counter() - g_start) * 1000, 2), ok=True))

        # Local cosine scale differs from the calibrated gemini scale, so use
        # the fast path's own floor to decide when a quick answer is too weak
        # to stream (the full pipeline's 0.78 gate still governs the final one).
        if retrieval.grounding_score < self.cfg.fast_path_grounding_floor:
            answer = Answer(
                text=REFUSAL_TEXTS["low_grounding"],
                mode="refusal",
                grounding_score=retrieval.grounding_score,
                confidence=0.0,
                refusal_reason="low_grounding",
            )

        answer.latency_ms = int((time.perf_counter() - t0) * 1000)
        result = PipelineResult(
            request_id=request_id,
            transcript=transcript,
            intent=intent,
            verdict=GuardVerdict(allow=True),
            retrieval=retrieval,
            answer=answer,
            spans=spans,
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        if report is not None:
            await report(answer, retrieval)
        return result

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