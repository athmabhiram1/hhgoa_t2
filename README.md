# VakRAG — Voice-Enabled Multilingual RAG

**HH Goa 2026 · Task 2.** A user *speaks* a question in any of **14 Indian
languages**; VakRAG transcribes it, retrieves grounded passages from the
**MSMARCO-XI (IndicRAGSuite)** corpus, and answers — with an honest,
measured latency budget (P50/P70/P100) and guardrails that know when **not**
to answer.

`Vak` (वाक्) = "speech" in Sanskrit.

---

## Pipeline

```
Voice ─▶ STT ─▶ Guards ─▶ Query Router ─▶ Two-Tier Retrieval ─▶ Generation ─▶ Post-guards ─▶ Answer
        Sarvam     safety/off-topic     fast vs deep    Neo4j HNSW + BM25    extractive / LLM     grounding +
        → whisper  PII gate             path            → RRF fusion          (Gemini→Ollama→…)    faithfulness
```

- **STT** — Sarvam `saaras:v3` (22 Indic languages, code-mixing) with a free
  local `faster-whisper` fallback on RTX 5050.
- **Store** — Neo4j as a single unified store: HNSW **vector** index, Lucene
  **BM25** fulltext index, and a **metadata/co-occurrence graph**.
- **Embeddings** — local `Qwen3-Embedding:0.6b` via Ollama `/api/embed`
  (1024-dim, dense-only; the sparse/lexical arm is Neo4j Lucene BM25).
- **Two-tier retrieval** — Tier 1 *Fast Path* (hybrid vector+BM25 + RRF, meets
  the <200ms latency contract) and Tier 2 *Deep Path* (**LightRAG** graph
  retrieval over a knowledge graph built with local Ollama on the RTX 5050).
- **Generation** — extractive fast answer (no LLM, sub-100ms) or streamed LLM
  answer (Gemini 3.5 Flash-Lite → Ollama → Groq → OpenAI fallback chain).
- **Guardrails** — safety, off-topic, grounding gate, faithfulness check.
- **Harness** — typed Pydantic stages, OpenTelemetry spans, retries + circuit
  breakers, provider failover, and a benchmark runner for P50/P70/P100.

## Chunking

Five parallel index namespaces: natural passage units (metadata-tagged),
fixed-size with overlap, recursive-character, query-anchored pseudo-documents,
and semantic sentence-level topic-coherence merging — with coarse→fine
multi-scale search and metadata-aware filtered retrieval.

## Quick start (local, GPU)

```bash
# 1. Neo4j
docker compose up -d neo4j

# 2. Python
pip install -r backend/requirements.txt      # or: uv sync
cp .env.example .env                          # add keys

# 3. Stream a balanced sample (~150-200k passages, all 14 languages)
python -m backend.ingestion.cli --sample

# 4. Index into Neo4j (qwen3-embedding → vector + fulltext + graph)
python -m backend.ingestion.cli --index
#    fast iteration: --index --limit-passages 3 --namespaces passage_natural query_anchored
#    deploy subset (fits AuraDB Free ~200K nodes):
#      --index --queries-per-lang 204 --namespaces passage_natural query_anchored passage_en

# 5. Run the latency benchmark (P50/P70/P100 → benchmarks/*.json + .md)
python -m backend.harness.benchmark

# 6. Start the API
uvicorn backend.main:app --reload --port 8000

# 7. Frontend
cd frontend && npm install && npm run dev     # http://localhost:5173
```

## Tests

```bash
python -m pytest backend/tests                # 48 tests — chunking, fusion,
                                              # router, guardrails, fast path, telemetry, eval
```

## Evaluation caveats (honesty)

- The offline MRR/Recall eval runs against the **MSMARCO-XI** corpus, whose
  Indic passages are **machine-translated**. The IndicRAGSuite paper's Table 2
  baseline uses a **hand-verified 1,000-query IndicMSMarco** benchmark — so our
  scores are the same metric on a harder/noisier pool, and a direct
  head-to-head overstates reproducibility. Treat the leaderboard as
  directionally comparable, not identical-benchmark.
- **Sanskrit has no published baseline row** in the paper (13 languages only);
  we report our Sanskrit MRR without a table-2 comparison.
- Our retrieval eval uses an untouched holdout (never indexed); a query's gold
  passage must be *present in the index* to score, so the deploy-sized subset
  is used for latency/demo and the full local sample for the eval.

## Repository layout

```
backend/
  main.py            FastAPI app + SSE router (/v1/ask, telemetry, benchmark, graph)
  config.py          pydantic-settings from .env
  core/              models (typed stages), tracing, retry, multi-provider LLM
  stt/               Sarvam + faster-whisper
  ingestion/         MSMARCO-XI sampler, multi-strategy chunking, indexer, CLI
  retrieval/         embeddings, neo4j store, fusion (RRF), reranker, graph
  rag/               query router, fast path, LightRAG deep path, prompts
  guardrails/        safety, off-topic, grounding, faithfulness
  harness/           pipeline orchestrator, benchmark runner, telemetry
frontend/            React + Vite SPA (mic capture, citations, latency panel, graph)
```

## Docs
- **`CONTEXT.md`** — locked architectural decisions and latency contract.
- **`AGENTS.md`** — build conventions.
- `backend/ingestion/`, `backend/harness/` — inline docs for indexing & benchmarks.
