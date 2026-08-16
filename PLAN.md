# VakRAG — Build Plan (phases)

Grounded in `CONTEXT.md` (source of truth for locked decisions). This plan is the
execution checklist the agent follows; any change to a locked decision must be
called out and reflected in `CONTEXT.md`.

## Anti-hallucination contract (applies to every phase)
- **Source of truth = CONTEXT.md.** Any divergence (e.g., README says "Five
  namespaces", CONTEXT.md says six) is a bug fixed by docs sync.
- **Doc-check gate:** every external API/model integration step opens with a
  ctx7/MCP or web verify. Code matches the *verified* signature/model-id, never
  an assumption. Anything not yet verified is marked `[verify]`.
- **Measured-not-estimated:** latency/quality numbers come only from harness runs
  or the cited paper (arXiv:2506.01615 Table 2). Unmeasured things are labeled
  `[not measured]`, never invented.
- **Firewalls:** `compileall` + pytest are the internal-logic firewall; benchmark
  runs are the latency firewall; the paper's Table 2 is the retrieval-quality
  firewall.
- **Blocked deps degrade gracefully:** no Sarvam key → STT reports
  `[not measured]`; never a fabricated number.

## Decisions locked from user input (2026-08-14)
- **Embeddings:** local-first via **Ollama** (`ollama pull qwen3-embedding:0.6b`
  primary candidate; `bge-m3` fallback). Chosen by **MRR@10 on our own holdout**,
  not leaderboard. Ollama `/api/embed` exposes dense only → **sparse arm = Lucene
  BM25** (already in Neo4j) unless FlagEmbedding python path is enabled later.
  → CONTEXT.md must be updated (embedding model + sparse-arm reality).
- **STT/LLM:** local-first for dev (faster-whisper / Ollama). At deployment add
  Gemini/Sarvam keys with proper rate limits + security. Provider chains in
  CONTEXT.md stay for production.
- **Deploy:** AuraDB Free (Neo4j) + **Vercel frontend**; backend host = free tier
  to verify (Render/Railway/Fly) or tunnel.

---

## Phase 0 — Environment & doc spikes (gate)
- **Goal:** bootable, verifiable baseline; zero unverified external-API assumptions.
- Steps:
  1. `docker compose up -d neo4j` (verify healthy on 7687).
  2. `cp .env.example .env`; fill what we have; verify Ollama running
     (`ollama list`) and `ollama pull qwen3-embedding:0.6b`.
  3. Install deps (`pip install -r backend/requirements.txt`); keep `lightrag`
     optional.
  4. **ctx7/web spikes** (before coding any external call):
     - current Gemini Flash-Lite model id `[verify]`
     - Sarvam `saaras:v3` STT request shape (endpoint, fields, audio format) `[verify]`
     - Ollama `/api/embed` batch contract `[verify]`
     - AuraDB Free caps (vector + fulltext, 200k nodes) `[verify]`
     - Vercel + backend-host free tiers `[verify]`
  5. Baseline check: `python -m compileall backend` OK; `pytest` 43 green;
     app boots; `/v1/health` = degraded without Neo4j.
- **Verify gate:** all of step 5 passes; spike findings recorded (in PLAN or
  CONTEXT.md notes).

## Phase 1 — Sample + holdout
- **Goal:** reproducible sample that separates eval from index (no contamination).
- Steps:
  1. Extend `ingestion/dataset.py`: stratified sampler reserves a **holdout split
     (~500 queries, balanced across 14 langs × query_types)**, excluded from the
     index sample; persisted to `data/holdout/<lang>.jsonl`, resumable.
  2. Add `--holdout` to `ingestion/cli.py`; keep sample size configurable.
- **Verify gate:** counts per lang; test asserts indexed vs holdout `query_id`s
  are disjoint; `load_sample` round-trips JSONL.

## Phase 2 — Embedding spike + index (six namespaces)
- **Goal:** dense embeddings in Neo4j for all six namespaces (incl. `passage_en`).
- Steps:
  1. Embedding candidates: `qwen3-embedding:0.6b` (primary), `4b`, `bge-m3` —
     measure warm-up, embed latency, and **MRR@10 on holdout**; lock winner into
     CONTEXT.md (embedding swap is a locked-component change → called out).
  2. Update `embeddings.py` to an Ollama `/api/embed` backend (keep fastembed/
     sentence-transformers as optional fallbacks).
  3. Index all six namespaces incl. `passage_en`; Lucene BM25 lexical arm;
     resumable `skip_done`; unique `chunk_id` constraint.
  4. Two index builds (see CONTEXT.md locked-change callout): **curated deploy
     subset** (`--queries-per-lang 204` × 3 query-path namespaces ≈ 88K nodes,
     fits AuraDB Free 200K) and **full sample × `passage_natural` only** (≈214K
     chunks) as the `in_index_mrr` eval pool.
- **Verify gate:** per-namespace counts; known-answer query retrieves its
  `is_selected` passage; vector dims match embedder (qwen3-0.6b = 1024);
  **deploy-subset node+relationship count printed from Neo4j and confirmed
  under 200K / 400K before Phase 6**.

## Phase 3 — Retrieval upgrade
- **Goal:** Tier-1 hybrid = native + `passage_en` arms → RRF → dedupe → optional
  rerank; LRU cache + speculative retrieval; query normalization.
- Steps:
  1. `RetrievalService`: parallel vector + BM25 across native namespaces +
     `passage_en` arm → RRF → dedupe → (optional) sparse re-score only if
     FlagEmbedding path on → optional rerank.
  2. Query-normalization layer: script/lang detection; retrieval variants
     (native script + Roman/code-mix) without changing answer language.
  3. LRU query→result cache + speculative retrieval hook on partial transcript.
  4. New tests: normalization (code-mixed/transliterated), cache hit≈1ms,
     cross-lingual query returns English passage.
- **Verify gate:** pytest + compileall; cache-hit bypasses vector DB;
  cross-lingual E2E answered from `passage_en`.

## Phase 4 — Offline eval harness (headline differentiator)
- **Goal:** three locked metrics — `in_index_mrr` (primary), `query_anchored_mrr`
  (LEAKY, separate), `held_out_mrr` (small-N, genuinely disjoint) — per language
  vs paper Table 2; answer + refusal-precision eval.
- Steps:
  1. `backend/harness/eval_mrr.py`: `in_index_mrr` retrieves every indexed query
     over the **full `passage_natural` pool** (full sample indexed separately
     from the deploy subset — see CONTEXT.md locked-change callout);
     `query_anchored_mrr` same but over `query_anchored` with the leakage caveat
     in JSON + Markdown; `held_out_mrr` = holdout queries whose gold text
     matches a recurring passage (coverage 1.0 by construction, asserted in
     tests).
  2. Baselines file embeds paper's bge-m3 numbers **with source citation**
     (arXiv:2506.01615 Table 2); Sanskrit shows **"no published baseline"**
     (absent from Table 2); report shows ours, baseline, delta.
  3. Answer eval: faithfulness vs retrieved set; correctness vs ground-truth
     `Answer` (lexical + optional LLM judge); refusal precision on off-topic set.
  4. Outputs → `eval/mrr_<ts>.json/.md`; every number traceable to a run
     timestamp or the paper; unmeasured cells = `[not measured]`.
- **Verify gate:** metric unit tests on tiny fixtures; **coverage is a
  first-class check — log coverage % before trusting any MRR number** (the
  original disjoint-holdout design had coverage 0.0 and was retired); tests
  assert the `held_out` eval set is disjoint from the index AND every query in
  it has gold-in-pool = True; artifacts exist.

## Phase 5 — E2E latency + guardrail validation (honest numbers)
- **Goal:** measured P50/P70/P100 matching the CONTEXT.md budget — or update the
  budget.
- Steps:
  1. Full sample index; benchmark ~150 queries (mixed langs/types) → per-stage
     percentiles + **cache hit/miss and warm/cold splits**.
  2. Concurrent load test for P100; guardrail validation sets (off-topic →
     refusal rate, low-grounding → refusal, on-topic → answer).
  3. **Update CONTEXT.md latency table with measured values** (honest contract;
     spans already exist).
- **Verify gate:** benchmark JSON/MD; `/v1/telemetry` live; every budget row has
  a measured number, else `[not measured]`.

## Phase 6 — Deploy (live link)
- **Goal:** judge-usable URL.
- Steps:
  1. AuraDB Free Neo4j; **Vercel frontend**; backend host (verify free tier or
     tunnel); prod `.env`; nginx SSE `proxy_buffering off`; boot warm
     (embedding model + cache); CORS + health checks.
  2. **Rate limiting + security:** per-IP request caps, request-size limits, CORS
     allowlist, no secrets in client, `.env` gitignored.
  3. STT/LLM at deploy: whisper→Sarvam; Ollama→Gemini (with keys).
- **Verify gate:** `/v1/health` OK on live URL; demo script runs end-to-end from
  a clean browser; cold vs warm reported.

## Phase 7 — Demo, videos, social, docs sync
- **Goal:** pitch + judging deliverables.
- Steps:
  1. 3-beat demo (Hindi voice → code-mix/cross-lingual → off-topic refusal) with
     live latency panel; 2 videos with `#RAGInGoa`.
  2. README: architecture diagram + eval leaderboard vs paper + latency
     distributions + honest limitations.
  3. **Docs sync:** fix README "five namespaces" → six; embedder change;
     provider defaults; CONTEXT.md/README consistency.
- **Verify gate:** README/CONTEXT.md consistent; `npm run build` OK; final
  `compileall` + full pytest; live link + artifacts listed.
