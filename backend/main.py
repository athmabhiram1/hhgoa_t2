"""VakRAG FastAPI app — voice + text RAG endpoint with streaming SSE events.

Endpoints:
  POST /v1/ask         SSE stream of stage events -> final answer
  POST /v1/ask/text    JSON convenience for programmatic clients
  GET  /v1/health      Neo4j + embedding readiness
  GET  /v1/telemetry   live P50/P70/P100 latency snapshot
  POST /v1/benchmark   run the latency harness on-demand
  GET  /v1/graph       knowledge-graph snapshot for the visual
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import Settings, get_settings
from .core.models import PipelineResult
from .core.providers import LLMClient
from .core.tracing import set_trace_sink
from .harness.benchmark import BenchmarkRunner
from .harness.pipeline import VakRagPipeline
from .harness.telemetry import tracker
from .retrieval.embeddings import EmbeddingService, set_embedding_service
from .retrieval.local_index import LocalFastIndex
from .retrieval.neo4j_store import Neo4jStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

cfg = get_settings()

# --------------------------------------------------------------------------
# App state
# --------------------------------------------------------------------------
class AppState:
    pipeline: VakRagPipeline | None = None
    store: Neo4jStore | None = None
    client: LLMClient | None = None
    embeddings_ready: bool = False
    fast_ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = LLMClient(cfg)
    pipeline = VakRagPipeline(cfg, client=client)
    store = Neo4jStore(cfg)
    if await store.verify_connectivity():
        pipeline.bind_retrieval(store)
        # Warm the embedding model at startup (downloads on first use).
        await asyncio.to_thread(_init_embeddings)
        asyncio.create_task(pipeline.deep.initialize())
        # Fast path (Tier 1): load the prebuilt local index when present.
        fast = LocalFastIndex(cfg)
        if cfg.fast_path_enabled and fast.load():
            # Warm the bge-m3 model + CUDA kernels on startup so the first
            # user query does not pay the cold-start cost (CUDA kernel compile
            # / cuDNN autotune ≈ 800ms on first encode, then ~25ms warm).
            # Fire one throwaway query through index.search() before marking
            # ready — discard result, log warm-up duration separately from
            # steady-state latency.
            import time as _time

            await asyncio.to_thread(fast._load_model)  # type: ignore[attr-defined]
            _t0 = _time.perf_counter()
            try:
                await asyncio.to_thread(fast.search, "warmup")  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.warning("fast path warmup query failed: %s", exc)
            else:
                _warm_ms = (_time.perf_counter() - _t0) * 1000
                logger.info("fast-path: warm-up query took %.1fms (discarded)", _warm_ms)
            pipeline.bind_fast_path(fast)
            state.fast_ready = True
            logger.info("VakRAG fast path ready: %d local chunks (200ms-compliant)", fast.size)
        state.pipeline = pipeline
        state.store = store
        state.client = client
        logger.info("VakRAG ready: Neo4j connected, retrieval bound, embeddings warm")
    else:
        logger.error("Neo4j unreachable at %s — /v1/ask will return 503 until it connects", cfg.neo4j_uri)
        state.pipeline = pipeline
        state.store = store
        state.client = client
    set_trace_sink(lambda t: None)
    yield
    if store is not None:
        await store.close()


def _init_embeddings() -> None:
    svc = EmbeddingService(cfg)
    try:
        svc.warm()
    except Exception as exc:  # noqa: BLE001 — optional backend; boot must degrade, not crash
        logger.error("Embedding backend unavailable (%s) — retrieval will be limited", exc)
        state.embeddings_ready = False
        return
    set_embedding_service(svc)
    state.embeddings_ready = True


app = FastAPI(title="VakRAG", version="0.1.0", lifespan=lifespan)
_cors_origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
if _cors_origins:
    # Locked Aug 2026: same-origin by default (nginx proxies /v1/ to uvicorn).
    # CORS is only enabled with an explicit allowlist; "*" is rejected here so
    # a misconfigured env var can't open the app to any origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    text: str | None = None
    audio_b64: str | None = None          # WAV/WebM bytes, base64
    lang: str | None = None
    mode: str = "auto"                    # auto | extractive
    query_type: str | None = None         # explicit opt-in retrieval filter (optional)


class AskTextResponse(BaseModel):
    result: PipelineResult


class BenchmarkRequest(BaseModel):
    n: int = 60


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _require_ready() -> VakRagPipeline:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return state.pipeline


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_and_track(pipe: VakRagPipeline, request: AskRequest, *, on_embed_queued=None, on_fast=None) -> PipelineResult:
    if request.audio_b64:
        try:
            audio = base64.b64decode(request.audio_b64)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid base64 audio")
        result = await pipe.run_audio(audio)
    elif request.text:
        result = await pipe.run_transcript(request.text, lang=request.lang, mode=request.mode, query_type=request.query_type, on_embed_queued=on_embed_queued, on_fast=on_fast)
    else:
        raise HTTPException(status_code=400, detail="provide text or audio_b64")
    tracker.record(result.total_ms, result.spans, refused=result.answer.mode == "refusal")
    return result


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.post("/v1/ask", response_model=None)
async def ask_stream(request: AskRequest) -> StreamingResponse:
    pipe = _require_ready()
    if not pipe.store or not await pipe.store.verify_connectivity():
        raise HTTPException(status_code=503, detail="Neo4j not connected")

    async def gen() -> AsyncIterator[str]:
        # Run the pipeline in a task so live stage events (embed "queued") can
        # be streamed before the final result — the frontend needs feedback
        # while the embed semaphore is contended, not a dead-looking wait.
        live: asyncio.Queue[str] = asyncio.Queue()

        async def on_embed_queued() -> None:
            await live.put(_sse("queued", {"stage": "embed", "message": "request queued behind other queries"}))

        async def on_fast(answer, retrieval) -> None:
            await live.put(
                _sse(
                    "quick",
                    {
                        "text": answer.text,
                        "mode": answer.mode,
                        "grounding_score": retrieval.grounding_score,
                        "latency_ms": retrieval.latency_ms,
                        "refusal_reason": answer.refusal_reason,
                    },
                )
            )

        task = asyncio.create_task(_run_and_track(pipe, request, on_embed_queued=on_embed_queued, on_fast=on_fast))

        while not task.done() or not live.empty():
            try:
                item = await asyncio.wait_for(live.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            yield item

        try:
            result = task.result()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline failure")
            yield _sse("error", {"message": str(exc)})
            return

        yield _sse("transcript", {"text": result.transcript.text, "language": result.transcript.language_code, "provider": result.transcript.provider})
        yield _sse("intent", {"language_code": result.intent.language_code, "query_type": result.intent.query_type, "needs_graph": result.intent.needs_graph})
        if not result.verdict.allow:
            yield _sse("guard", {"allow": False, "category": result.verdict.category, "reason": result.verdict.reason})
        if result.retrieval is not None:
            yield _sse(
                "retrieval",
                {
                    "grounding_score": result.retrieval.grounding_score,
                    "n_candidates": len(result.retrieval.candidates),
                    "latency_ms": result.retrieval.latency_ms,
                    "candidates": [
                        {"id": c.id, "text": c.text[:300], "score": c.score, "source": c.source, "lang": c.language_code, "query_type": c.query_type}
                        for c in result.retrieval.candidates[:8]
                    ],
                },
            )
        yield _sse(
            "answer",
            {
                "text": result.answer.text,
                "mode": result.answer.mode,
                "grounding_score": result.answer.grounding_score,
                "confidence": result.answer.confidence,
                "refusal_reason": result.answer.refusal_reason,
                "citations": [c.model_dump() for c in result.answer.citations],
            },
        )
        yield _sse("done", {"request_id": result.request_id, "total_ms": result.total_ms, "spans": [s.model_dump() for s in result.spans]})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/ask/text")
async def ask_text(request: AskRequest) -> dict:
    pipe = _require_ready()
    result = await _run_and_track(pipe, request)
    return {"request_id": result.request_id, "result": result.model_dump(mode="json")}


@app.get("/v1/health")
async def health() -> dict:
    neo4j_ok = bool(state.store) and await state.store.verify_connectivity() if state.store else False
    return {"status": "ok" if neo4j_ok else "degraded", "neo4j": neo4j_ok, "embeddings": state.embeddings_ready, "fast_path": state.fast_ready, "provider_chain": cfg.primary_llm_provider}


@app.get("/v1/telemetry")
async def telemetry() -> dict:
    return tracker.snapshot()


@app.post("/v1/benchmark")
async def benchmark(body: BenchmarkRequest) -> dict:
    # Locked Aug 2026: /v1/benchmark is a credit-burning, load-generating
    # endpoint. On a public deployment it must be off; it is only reachable
    # when BENCHMARK_ENABLED=true (local dev). The CLI harness
    # (python -m backend.harness.benchmark) is the intended path for real runs.
    if not cfg.benchmark_enabled:
        raise HTTPException(status_code=403, detail="benchmark endpoint disabled — enable with BENCHMARK_ENABLED=true (local dev only)")
    pipe = _require_ready()
    from .ingestion.dataset import load_sample
    from pathlib import Path

    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    sample = load_sample(sample_dir, cfg.langs)
    if not sample:
        raise HTTPException(status_code=404, detail="no sample indexed — run ingestion first")
    queries = [q.query for q in sample[: body.n]]
    report = await BenchmarkRunner(cfg, pipe).run(queries)
    return report


@app.get("/v1/graph")
async def graph() -> dict:
    if state.store is None:
        raise HTTPException(status_code=503, detail="store not ready")
    return await state.store.graph_snapshot()