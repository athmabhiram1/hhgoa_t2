# VakRAG — Voice-Enabled Multilingual RAG

**HH Goa 2026 · Task 2.** A user *speaks* a question in any of **14 Indian
languages**; VakRAG transcribes it, retrieves grounded passages from the
**MSMARCO-XI (IndicRAGSuite)** corpus, and answers — with an honest, measured
latency budget (P50/P70/P100), a per-language MRR leaderboard tied to the
IndicRAGSuite paper baseline, and guardrails that know when **not** to answer.

`Vak` (वाक्) = "speech" in Sanskrit.

## Live demo

- **Live link:** https://vakrag.onrender.com (add after deploy)
- **Demo script:** Hindi voice → code-mixed ("Meri crop ke liye konsa fertilizer
  use karoon?") → cross-lingual (Hindi answer from an English passage) →
  off-topic refusal with a reason.
- **Videos + social:** 2 demo videos with `#RAGInGoa` on IG/X/LinkedIn.

---

## Pipeline

```
Voice ─▶ STT ─▶ Guards ─▶ Router ─┬─▶ Fast Path (local bge-m3 ANN, 45ms P50) ─▶ extractive span ─┬─▶ Answer (quick, 200ms)
        Sarvam  safety/off-topic      │  31k passages, SSE `quick`                 │              └─▶ Answer (full, LLM)
        →whisper PII gate             └─▶ Full Path (Vertex+Neo4j+RRF+grounding) ─▶ LLM (Gemini) ─┘
```

## What's under the hood

- **STT** — Sarvam `saaras:v3` (22 Indic languages, code-mixing; final
  transcripts only) with a free local `faster-whisper` fallback on the RTX 5050.
- **Store** — Neo4j as a single unified store: HNSW **vector** index, Lucene
  **BM25** fulltext index, and a **metadata/co-occurrence graph**.
- **Embeddings** — **local dev:** `Qwen3-Embedding:0.6b` via Ollama (1024-dim,
  dense-only; sparse arm = BM25). **Deploy:** `gemini-embedding-001` via the
  **Gemini API** (AI Studio `GEMINI_API_KEY` — free-tier quota, NOT the GCP
  credit and not Vertex). Byte-identical vectors to Vertex `RETRIEVAL_QUERY`
  (measured `max_abs_diff = 0.0`), so the deploy pool is self-consistent.
  Swapped on the evidence, not vibes: on MIRACL (nDCG@10) gemini-embedding-001
  scores 70.4 avg vs qwen3-embed-4B 69.5 — Hindi 65.1 vs 60.2, Telugu 81.3 vs
  68.9, Bengali 78.8.
- **Two-tier retrieval** — **Tier 1 Fast Path** (NEW Aug 21): local `bge-m3` (fp16,
  CUDA) + in-memory brute-force cosine over 31,259 passages (`passage_natural` +
  `passage_en`), **45ms P50** (70 queries, 14 langs) — the 200ms-compliant
  extractive output, streamed first as an SSE `quick` event. **Tier 2 Full Path**
  (hybrid Vertex+Neo4j vector+BM25 + RRF fusion, grounding 0.78) streams the LLM
  answer after as progressive enhancement. **Deep Path:** LightRAG graph retrieval.
- **Generation** — extractive fast answer (no LLM, sub-100ms, faithful by
  construction) or streamed LLM answer (Gemini 3.5 Flash-Lite).
- **Guardrails** — safety, off-topic, grounding gate (0.78), faithfulness
  check. All **deterministic** by default (LLM judge off); refusals are ordinary
  responses with a `refusal_reason`, never crashes.
- **Harness** — typed Pydantic stages, OpenTelemetry spans, retries + circuit
  breakers, provider failover, P50/P70/P100 benchmark runner.
- **API safety** — rate limits (slowapi), request-size caps, CORS
  same-origin, benchmark gating, nginx body limit. See "API safety".

## Chunking — six namespaces

`passage_natural` (semantic units) · `passage_fixed` (256 tok/48 overlap) ·
`passage_recursive` (recursive character) · `query_anchored`
(query+passage pseudo-docs) · `semantic` (deterministic sentence-level
topic-coherence) · `passage_en` (English cross-lingual arm). Plus coarse→fine
multi-scale search and metadata-aware filtering (query_type filter is opt-in —
auto-filtering was removed after measured harm).

## Retrieval quality (measured, not vibes)

`in_index_mrr` — every indexed query whose gold is in the pool, MRR@10 over the
full `passage_natural` pool (N=12,922; run `20260815-133028`):

| lang | VakRAG MRR@10 | paper baseline* |
|---|---|---|
| Assamese | 0.252 | 0.46 |
| Bengali | 0.248 | 0.49 |
| Gujarati | 0.279 | — |
| Hindi | 0.362 | 0.52 |
| Kannada | 0.278 | — |
| Malayalam | 0.305 | — |
| Marathi | 0.312 | — |
| Nepali | 0.317 | — |
| Odia | 0.271 | 0.45 |
| Punjabi | 0.275 | — |
| Sanskrit | 0.283 | no published baseline |
| Tamil | 0.266 | 0.49 |
| Telugu | 0.344 | 0.50 |
| Urdu | 0.380 | — |
| **overall** | **0.298** · Recall@10 0.546 · nDCG@10 0.357 | — |

\* IndicRAGSuite Table 2 (arXiv:2506.01615, bge-m3). **Same metric, not the
same benchmark** — our pool is machine-translated MSMARCO-XI; the paper uses a
hand-verified 1,000-query IndicMSMarco. We trail the baseline (local 0.6B
embedder vs bge-m3) and report it as a measured weakness. Sanskrit has no
published baseline row.

Every number is traceable to `eval/mrr_<ts>.json/.md`. `query_anchored_mrr`
(0.045) is labeled LEAKY and never compared with `in_index_mrr`; the genuine
unseen-query `held_out_mrr` is N=0 by construction of the dataset — `[not
measured]`, never estimated.

## Latency (measured contract — Aug 21, 2026)

**Tier 1 Fast Path (local, 200ms-compliant):** corpus re-embedded once with
`BAAI/bge-m3` (fp16, CUDA) into an in-memory brute-force index (31,259 chunks:
`passage_natural` + `passage_en`). No Vertex, no Neo4j at query time.

| metric | fast path (local bge-m3, 70 queries · 5/lang, RTX 5050) |
|---|---|
| P50 | **45.53 ms** |
| P70 | **50.61 ms** |
| P100 | **80.52 ms** |
| mean | 47.33 ms |
| per-lang P50 range | 38.7–56.2 ms (ta 38.7 · ne 56.2) |

Full report: `eval/fastpath_latency_20260821-074450.json` + `.md`.
Build the index: `python -m backend.harness.build_fastpath` (read-only Neo4j
text export, then local embed; ~165s on RTX 5050 fp16, output `data/fastpath/`).
Warm start loads the model at boot so the first user query does not pay the
cold-load cost (~32s).

**Full pipeline (Vertex + Neo4j + RRF + grounding + LLM):** retrieval + extractive
core ~650ms single cold query; cache hit ~6ms; full voice→answer dominated by STT
(Sarvam <250ms claimed). The fast-path extractive answer streams first as the
200ms-compliant output; the LLM answer streams after as progressive enhancement
and is reported separately in the UI (`quick` vs `full` cards + SSE events).

Legacy concurrency note: Ollama `/api/embed` is serial — cold P50 inflated to
~3.5s at 8-way; warm replay 11ms. Full table: `CONTEXT.md` → Latency Contract.

## Deployment (Render free tier)

`deploy/render/render.yaml` — one container: **nginx** (serves the built
frontend, proxies `/v1/` to uvicorn, SSE `proxy_buffering off`) + **FastAPI**.
Neo4j = **AuraDB** (cloud), embeddings = **Vertex AI** `gemini-embedding-001`,
generation = **Gemini**, STT = **Sarvam**. Secrets are Render dashboard env vars
(`sync: false`, never committed).

Free-tier reality (honest): Render free services spin down after 15 min idle
(~60s cold spin-up) and cap at **750 instance-hours/month**. Plan: daytime-only
keep-alive (6am–midnight IST via cron-job.org → `/v1/health`) ≈ 560 hrs/mo +
`backend/harness/prewarm.py` to re-fill the LRU cache after wake. The ~60s
cold start is documented in the demo, never hidden.

## API safety

Public endpoints are rate-limited and capped so the live demo can't be abused
into draining the Gemini generation RPD (500/day) or Sarvam credits:

1. **CORS** — same-origin only (frontend is served behind the same nginx).
2. **Request caps** — `text ≤ ~300 chars`, `audio_b64 ≤ ~4MB`, enums for
   `lang`/`mode`.
3. **Rate limits (slowapi)** — per-IP 20/min · 300/hr · 500/day; global
   1000/day · 1500/hr; HTTP 429 + `Retry-After`.
4. **Cost gate** — `/v1/benchmark` is disabled on the public deploy; global
   daily cap sits below the Gemini RPD.
5. **nginx** — `client_max_body_size 8M`.

## Costs & credits (we actually did the math)

- **Gemini API embeddings (deploy path):** `gemini-embedding-001` over the
  Gemini API is a **separate bucket from the GCP $300 credit** — free-tier
  quota (RPM/TPM/RPD), not a dollar spend. The AuraDB subset re-embed
  (~59k chunks, 100/batch, serial) is **~591 API calls** — fits in a day even
  on a conservative 1,000 RPD free tier; 429s if quota exhausts (billing
  account → Tier 1 raises caps, no spend required). Live numbers:
  `aistudio.google.com/rate-limit` (signed-in).
- **GCP free trial:** ₹28,693.88 credit (expires 2026-11-15) — the OBSOLETE
  Vertex path only; no GCP-credit spend on the current gemini embed path.
- **Sarvam:** ₹100 free credits ≈ 200 min of STT (Starter 60 req/min) —
  plenty for the demo; the rate limits above protect it.
- **Render:** free (with the 750 hrs/mo caveat above). **AuraDB Free:** 59,296
  nodes / 31,578 rels built (200K / 400K caps).

## Quick start (local, GPU)

```bash
# 1. Neo4j
docker compose up -d neo4j

# 2. Python
pip install -r backend/requirements.txt      # or: uv sync
cp .env.example .env                          # add keys (never commit .env)

# 3. Stream a balanced sample (~150-200k passages, all 14 languages)
python -m backend.ingestion.cli --sample

# 4. Index into Neo4j (qwen3-embedding → vector + fulltext + graph)
python -m backend.ingestion.cli --index
#    deploy subset (fits AuraDB Free):
#      --index --queries-per-lang 204 --namespaces passage_natural query_anchored passage_en

# 5. Run the latency benchmark (P50/P70/P100 → benchmarks/*.json + .md)
python -m backend.harness.benchmark

# 6. Retrieval eval (MRR leaderboard above)
python -m backend.harness.eval_mrr

# 7. Start the API
uvicorn backend.main:app --reload --port 8000

# 8. Frontend
cd frontend && npm install && npm run dev     # http://localhost:5173
```

## Tests

```bash
python -m pytest backend/tests                # 79 tests — chunking, fusion,
                                              # router, guardrails, fast path,
                                              # resilience, telemetry, eval,
                                              # vertex embed, embed gate, STT
```

## Repository layout

```
backend/
  main.py            FastAPI app + SSE router (/v1/ask, telemetry, graph)
  config.py          pydantic-settings from .env
  core/              models (typed stages), tracing, retry, multi-provider LLM
  stt/               Sarvam + faster-whisper
  ingestion/         MSMARCO-XI sampler, multi-strategy chunking, indexer, CLI
  retrieval/         embeddings (ollama/gemini/vertex), neo4j store, fusion, graph
  rag/               query router, fast path, LightRAG deep path, prompts
  guardrails/        safety, off-topic, grounding, faithfulness
  harness/           pipeline orchestrator, benchmark runner, eval_mrr, prewarm
frontend/            React + Vite SPA (mic capture, citations, latency panel, graph)
deploy/render/       Dockerfile + nginx.conf + render.yaml (free tier)
```

## Docs
- **`CONTEXT.md`** — locked architectural decisions, measured latency contract,
  eval methodology, reviewer-feedback status.
- **`AGENTS.md`** — build conventions.
- **`PLAN.md`** — phased execution checklist.
