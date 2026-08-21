    # VakRAG — Voice-Enabled Multilingual RAG (HH Goa 2026 · Task 2)

    ## What We Are Building
    A voice-enabled Retrieval-Augmented Generation system over the **MSMARCO-XI**
    IndicRAGSuite dataset (ai4bharat). A user speaks a question in any of 14 Indian
    languages, the pipeline transcribes it, retrieves relevant passages from the
    indexed corpus, and returns a grounded answer — end to end — with an honest
    latency budget reported as P50 / P70 / P100. DATASET:https://huggingface.co/datasets/ai4bharat/MSMARCO-XI.

    `Vak` (वाक्) = "speech" in Sanskrit — the dataset includes Sanskrit, so the name
    is on-theme.

    ## The Core Problem
    Indian-language QA is underserved: hosted STT is weak on code-mixing, retrieval
    quality in Indic languages is poor with English-centric embeddings, and most
    "voice RAG" demos are single raw prompt-in/text-out calls. We ship a **harnessed,
    guarded, latency-measured** pipeline that knows when NOT to answer. We optimize
    for *real* user input — code-mixed ("Meri crop ke liye konsa fertilizer use
    karoon?"), transliterated, and **cross-lingual** (a Hindi voice question answered
    from an English passage) — and we prove retrieval quality by reproducing the
    IndicRAGSuite MRR benchmark per language (arXiv:2506.01615), not by vibes.

    ## Tech Stack (Locked — do not silently swap)
    - **Pipeline orchestration**: FastAPI (async) + Pydantic typed stages + OpenTelemetry spans
    - **Vector + fulltext + graph**: Neo4j (single unified store — HNSW vector index,
      Lucene BM25 fulltext index, metadata/co-occurrence graph)
    - **Embeddings — local dev**: **Qwen3-Embedding via Ollama `/api/embed`** —
      `qwen3-embedding:0.6b` (1024-dim, multilingual MTEB 64.33, 32k ctx; 100% GPU on
      RTX 5050; ~220ms/query steady single-shot, amortized by batching — verified Aug
      2026). Emits **dense vectors only** → the sparse/lexical arm is served by
      **Neo4j Lucene BM25** (no learned-sparse re-score unless a FlagEmbedding python
      path is later enabled). Fallbacks: `bge-m3` via fastembed/sentence-transformers.
      > **Locked-change callout (Aug 2026) — deploy embedder is the Gemini API
      > backend (`gemini-embedding-001`); local dev stays Ollama Qwen3.** Two
      > embedders, one schema (both 1024-dim cosine — no Neo4j schema change). The
      > deploy (`deploy/render/render.yaml`) runs `EMBED_BACKEND=gemini` (uses the
      > AI Studio `GEMINI_API_KEY`, already set and verified working); local
      > `.env` keeps `EMBED_BACKEND=ollama`. **This SUPERSEDES the earlier
      > "deploy embedder = Vertex AI" callout.** The Vertex path is not
      > deployable: Render is a generic Docker container with NO GCP metadata
      > server and no interactive gcloud, so Vertex needs an inline
      > `VERTEX_CREDENTIALS` SA JSON — and the org policy blocks SA-key creation
      > (`orgpolicy.googleapis.com` not enabled; `roles/aiplatform.user` console
      > only). Justified by evidence: (1) MIRACLRetrievalHardNegatives nDCG@10 —
      > gemini-embedding-001 avg 70.4 vs qwen3-embed-4B 69.5 (Hindi 65.1 vs 60.2,
      > Telugu 81.3 vs 68.9, Bengali 78.8); (2) **measured Aug 2026: Gemini API
      > `batchEmbedContents` (no task_type) is BYTE-IDENTICAL to Vertex
      > `RETRIEVAL_QUERY` (`max_abs_diff = 0.0`)** — so the Gemini API backend
      > lands in the same vector space as Vertex query vectors, and a single
      > backend is consistent for both indexing and querying. Vertex
      > `RETRIEVAL_DOCUMENT` differs (max_abs_diff ≈ 0.17) — the model's
      > document/query tuning, only relevant if the Vertex backend were ever used
      > for indexing. Cost — **the Gemini API is a SEPARATE billing bucket from
      > the GCP $300 credit**: AI-Studio-issued `GEMINI_API_KEY` usage is NOT
      > billed to the GCP project, it is either **free inside the Gemini API's
      > own free-tier quota (RPM/TPM/RPD)** or fails with HTTP 429 — there is no
      > dollar spend to track. The constraint is RATE, not money (see the
      > rate-limit note below). Status: AuraDB pool was indexed with Qwen3
      > vectors (Ollama, `embed_backend=NULL`, pre-property build) + 1,120
      > Vertex-document pilot chunks — **mixed vector space, must be fully
      > re-embedded with the `gemini` backend before eval/deploy** (queued,
      > pending user go-ahead).
      > **Gemini API rate limits (verified Aug 2026):** the model
      > `gemini-embedding-001` over the Gemini API is rate-limited per project
      > (usage tier dependent; live numbers only on the signed-in
      > aistudio.google.com/rate-limit page — Google no longer publishes a fixed
      > per-model table). The gemini backend batches **100 texts per
      > `batchEmbedContents` call, fully serial (no parallelism)** — so the
      > ~59k-chunk AuraDB re-embed is **~591 API requests** (well under even a
      > conservative 1,000 RPD), unlike the Vertex backend which is 1 text/call
      > with an 8-wide thread pool. If the free tier's RPD turns out too tight,
      > linking a billing account (no spend needed) moves the project to Tier 1
      > with meaningfully higher caps.
      > **gemini-embedding-001 dimension tradeoff (verified Aug 2026):** the model
      > recommends **3072 / 1536 / 768** output dims for best quality; 1024 is
      > supported but suboptimal. We are **LOCKED at 1024** because both embedders
      > must share one 1024-dim Neo4j vector schema — do NOT switch to 768/1536
      > (schema change + full re-embed) and do NOT switch to gemini-embedding-2
      > (different vector space — mixing would silently corrupt retrieval).
      > Full-pool re-embed cost estimate: `deployment-plan.md` §1.
      > Prior locked-change (bge-m3 ONNX → Qwen3 via Ollama) stands for the local
      > pool; dense-only reality means BM25 remains the sparse arm on both paths.
    - **STT — primary**: Sarvam AI `saaras:v3` (`POST /speech-to-text`), 22 Indic
      languages, code-mixing, claimed <250ms median. **Fallback**: local
      `faster-whisper` (CTranslate2) on RTX 5050 — free, no rate limits.
      > **Locked-change callout (Aug 2026):** verified the Sarvam REST
      > `/speech-to-text` emits **final transcripts only — no partial/interim
      > results**, and the decision is **LOCKED: keep REST final-transcript-only
      > for production STT; do NOT adopt WS `/speech-to-text/ws`** (its chunks are
      > not true word-by-word partials, so a partial-transcript prefetch was never
      > achievable). Speculative retrieval runs on the **last finalized utterance /
      > VAD end-of-speech**, not on mid-utterance partials. The embed cost is
      > therefore measured in the clocked core (see Latency Contract).
      > **Sarvam credit reality (verified):** signup gives **₹100 free credits ≈
      > 200 minutes of STT** (Starter tier, 60 req/min). Generous for a demo but
      > NOT unlimited — this is exactly why the API-safety rate limits + audio
      > caps exist (see API Safety). On Render, faster-whisper fallback is NOT
      > installed, so drained credits = broken STT.
    - **Generation LLM**: Gemini Flash-Lite (current-gen, TTFT ~0.2s) primary →
      Ollama local (RTX 5050) fallback → Groq / OpenAI pluggable. Multi-provider.
      > **Model id updated (Aug 2026):** `gemini_model` → `gemini-3.5-flash-lite`
      > (GA July 21, 2026; verified in Google AI docs) in config.py/.env/.env.example.
      > The old `gemini-2.5-flash-lite` string is stale for new keys.
    - **Indexing/extraction LLM**: Ollama local on RTX 5050 (free, no limits).
    - **Deep engine**: LightRAG (`lightrag-hku`) with `Neo4JStorage` graph +
      `NanoVectorDBStorage` vectors, extraction via Ollama. PolicySattva pattern.
    - **Frontend**: React + Vite (plain CSS, custom SVG knowledge-graph view, SSE
      stage stream, live P50/P70/P100 latency panel, Web Audio mic capture → base64).

    > **Locked-change callout (Aug 2026) — hosting: Render free tier. HF Spaces
    > REJECTED, Cloud Run and the spare laptop are retired fallbacks.** The earlier
    > HF Spaces Docker work was disproven (Docker Spaces require a paid plan,
    > verified) and is REJECTED. The locked deploy is **Render free tier**:
    > `deploy/render/render.yaml` — ONE web service running nginx (serves the built
    > frontend, proxies `/v1/` to uvicorn, SSE `proxy_buffering off`) + FastAPI in a
    > single container; Neo4j = AuraDB (cloud), embeddings = Vertex
    > `gemini-embedding-001`, generation = Gemini, STT = Sarvam. Secrets are Render
    > dashboard env vars (`sync: false`), never committed. **Free-tier constraints
    > (verified):** 15-min idle **spin-down** with ~60s cold **spin-up**, and a
    > **750 instance-hours/month cap** (exceeding it = suspension = dead demo link).
    > Mitigations locked: (1) **daytime-only keep-alive** — cron-job.org pings
    > `/v1/health` 6am–midnight IST ≈ **~560 hrs/mo**, inside the cap; (2)
    > `backend/harness/prewarm.py` re-fills the LRU cache after a wake; (3) the
    > ~60s cold start is stated in the demo/README, never hidden. The spare
    > Ubuntu laptop + Cloud Run remain documented ad-hoc fallbacks only.
    >
    > **AuraDB Free lifecycle (verified Aug 2026) — the keep-alive MUST touch
    > Neo4j, not just Render.** Free-tier AuraDB instances **auto-pause after 72h
    > of inactivity** and are **deleted after ~90 days paused** (Neo4j blog) /
    > "30 days without activity" (official FAQ page). The locked keep-alive
    > (`/v1/health` → `verify_connectivity()` against bolt://) satisfies both:
    > every ping resets the 72h clock and keeps the DB alive past the deletion
    > windows. Overnight idle is ~6h — far under 72h. **Removing the keep-alive =
    > the demo link spins down AND the graph DB eventually gets deleted.**
    > Deployment runbook: `deployment-plan.md` (§6).

    ## Architecture — Two-Tier Retrieval Engine
    ```
    Voice ─▶ STT (Sarvam → faster-whisper fallback) ─▶ Guards ─▶ Query Router
                                                                  │
      TIER 1  FAST PATH (default, latency-first, <200ms)          │
      Embed (Ollama local / Vertex gemini-embedding-001)          │
      → Neo4j HNSW vector + BM25 fulltext                         │
      → RRF fusion → extractive span answer OR streamed LLM       │
      LRU query→result cache (hit ≈1ms, bypasses vector DB) +     │
      speculative retrieval on VAD/end-of-speech (Sarvam REST     │
      = final-transcript-only; no mid-utterance partials)         │
                                                                  │
      TIER 2  DEEP PATH (LightRAG, quality play)                  │
      LightRAG naive/hybrid/mix query modes over Ollama-built     │
      knowledge graph in Neo4j — powers graph visual +            │
      complex/relational questions                                │
                                                                  ▼
      Post-guards (grounding gate + faithfulness check) → Response + citations + telemetry
    ```
    - Default query mode is **Tier 1** because LightRAG's graph query modes fire
      extra LLM calls at query time (quality, not latency). The router escalates to
      Tier 2 for relational/multi-hop questions and when the graph visual is needed.
      Tier 2 is **optional and never on the demo's critical path** — if it is
      missing, slow, or fails, Tier 1 has already answered and the demo never stalls.
    - Both tiers share the same Neo4j instance; namespaces isolate index tiers.

    ## Chunking Strategy (six parallel namespaces in Neo4j)
    1. `passage_natural` — each dataset passage is a semantic unit, tagged with
      `lang`, `query_id`, `query_type`, `position`, `is_selected`.
    2. `passage_fixed` — fixed-size (256 tokens, 48 overlap) for long passages.
    3. `passage_recursive` — recursive character splitting for structured text.
    4. `query_anchored` — query+passage pseudo-documents (question-aware semantics).
    5. `semantic` — deterministic sentence-level bigram topic-coherence merging
      (model-free, index-stable; no embedding calls during chunking).
    6. `passage_en` — English passages (cross-lingual arm): enables Indic-script
      voice queries to retrieve and answer from `English_passages` (bge-m3 is
      100+ language capable).
    Plus **coarse→fine multi-scale retrieval** (document-level → chunk-level) and
    metadata-aware filtered search (language filter, query_type filter).
    Ground-truth `is_selected` labels enable supervised recall@k evaluation.

    > **LOCKED-CHANGE CALLOUT (Phase 5, Aug 2026) — query_type filter is now opt-in
    > only.** Automatic query_type filtering was tried on the live retrieval path
    > and removed: the heuristic classifier's guess is only ~42% accurate
    > (17/40 measured on sample queries; worse than a coin flip across some
    > classes), and a wrong guess zeroed candidates for ~1/3 of queries → false
    > `low_grounding`/`no passages` refusals unrelated to actual grounding quality.
    > Decision: live retrieval searches across ALL query_types by default, exactly
    > matching what `in_index_mrr` / `query_anchored_mrr` already measured (neither
    > eval metric applied a query_type filter — CONTEXT.md eval section). The
    > filtering capability is retained but only activates when a caller passes
    > `query_type` explicitly in the request body (`POST /v1/ask`, optional field).
    > The heuristic classifier stays for telemetry/query-type tagging, but is NOT
    > wired into filtering decisions. This prevents classifier-driven false
    > refusals from polluting Phase 5 guardrail refusal-precision numbers.

    ## Dataset
    - `ai4bharat/MSMARCO-XI` — 11.5M rows / 55.6 GB / 14 Indic languages. Each
      language is its OWN parquet file: `train/*.parquet` (~3.5GB each, single row
      group) and `validation/*.parquet` (~450MB each, single row group).
    - **We index a curated, balanced sample: ~150-200k passages across all 14
      languages.** Sampling source = `dataset_split="validation"` (config) because
      the train split is 13×3.7GB ≈ 45GB (disk-infeasible on the demo box) and
      LACKS a Telugu file, while validation is 14×450MB ≈ 6.3GB and includes all 14
      languages. Files are fetched ONE at a time via `hf_hub_download` (resumable,
      token-aware, cached) and processed in small pyarrow batches — this replaced
      `datasets` streaming, whose `default` config concatenates ALL train files
      (OOM: `realloc of size 3221225472 failed`) and never caches to disk.
    - **Locked-change callout (Aug 2026) — two index sizes, deploy vs eval.** AuraDB
      Free caps at **200K nodes / 400K relationships** (official FAQ, verified Aug
      2026). The full ~214K-passage sample × six namespaces = **~1.49M chunk nodes —
      ~7.5x over the cap**; even the three query-path namespaces alone ≈ 642K nodes.
      So deployment uses a **curated subset**: `--queries-per-lang 204` (≈ 28.5K
      passages, all 14 languages, both cross-lingual arms) indexed into the three
      query-path namespaces (`passage_natural` + `query_anchored` + `passage_en`) ≈
      **~88K nodes — fits AuraDB Free with headroom**. The full sample stays in
      local Docker Neo4j for the offline MRR eval (no node cap locally). CLI:
      `python -m backend.ingestion.cli --index --queries-per-lang 204 --namespaces passage_natural query_anchored passage_en`.
    - **Known gap — no chunk-embedding cache (Aug 2026).** Embeddings are
      deterministic per `chunk_id` (content-stable chunk ids), but `indexer.py`
      recomputes every vector from raw passage text via Ollama on each `--index`
      run against a new DB target — there is no persistent cache keyed by
      `chunk_id` (the only cache, `embeddings._QUERY_CACHE`, is a bounded
      in-memory request-path cache). So re-pointing the index to AuraDB (or any
      future target) pays the full embed cost again. Follow-up idea (not
      implemented now): persist `{chunk_id: vec}` (e.g. a Parquet/JSONL sidecar)
      during ingestion so a re-deploy to a different DB target or a schema-only
      rebuild can load vectors without re-embedding. **This gap is the reason the
      gemini re-embed (queued) re-burns the full free-tier quota (~591 API calls
      for the subset), not a handful.**
    - **Locked — per-batch resume skip in the ingestion pipeline (Aug 16 2026,
      permanent, not a one-off patch).** A partial run that dies mid-namespace
      previously re-embedded every already-stored chunk from zero on restart
      (the namespace-level `skip_done` check only skips FULLY-populated
      namespaces). `Neo4jStore.existing_chunk_ids()` now returns which
      `chunk_id`s in a batch already exist (ONE batched `IN`-list query per
      batch — never one query per chunk), and `indexer.py` skips embedding +
      MERGE for those chunks at the per-batch level. Restarts resume from the
      actual stored position in seconds instead of re-embedding hours of work.
      This is locked ingestion behavior going forward, valuable for any future
      re-deploy. Verified on the Aug 16 AuraDB build (26 batches of 512 skipped
      in ~0.1s each, then fresh embedding resumed at batch 27).
    - **Verified AuraDB deploy-subset state (Aug 16 2026).** Built and
      confirmed directly against `neo4j+s://c551c599.databases.neo4j.io`
      (not log-reported numbers):
      `ChunkNatural 28,518` · `ChunkAnchored 28,518` · `ChunkEnglish 2,037`
      (passage_en dedups by design across languages — 26,481 dedup collisions
      expected) · `Query 204` · `Language 14` · `QueryType 5`.
      **Total nodes 59,296 / 200K cap · total relationships 31,578 / 400K
      cap** — fits AuraDB Free with headroom. All `*_vector` (cosine 1024-dim)
      + `*_fulltext` + RANGE indexes present (not just the 2 LOOKUP indexes
      AuraDB starts with). Completion timestamp 12:11:50 (run `20260816`,
      watchdog `aura_index3`). Retrieval smoke test against AuraDB (8 real
      Assamese queries): top-1 carries the correct `query_id` **8/8**, grounding
      **8/8 ≥ 0.78** (0.837–0.887); the exact ground-truth selected passage
      ranked top for 2/8 — the other 6 retrieved a different passage of the
      same query (normal ranking behavior, not missing data).
    - Each row: `query`, `Answer`, `query_id`, `query_type`, `passages{is_selected,
      English_passages, Translated_passages}`, `Eng_Query`.
    - We index `Translated_passages` (native-script) as the primary corpus AND
      `English_passages` as a first-class `passage_en` namespace — cross-lingual
      retrieval (Indic query → English passage) is a demoed feature, not a footnote.
    - `is_selected` = ground-truth relevance → used for recall@k benchmarks.

    ## Latency Contract (measured, not aspirational)
    - **200ms target applies to the retrieval+answer core (post-transcription)** on
      Tier 1. **Measured budget table (Aug 2026 benchmark, 140 mixed-language
      queries, concurrency 8, cold+warm two-pass)** — every row measured, none
      aspirational:
      | stage | cold single-query (ms) | span name |
      |---|---|---|
      | embed query (Qwen3 via Ollama, GPU) | ~155–260 (cache hit ≈0; +2.3s one-time model reload per process) | `retrieve.embed` |
      | Neo4j hybrid search (2×vector + 2×BM25, parallel) | ~265–630 (P50 ~400) | `retrieve.search` |
      | RRF fusion + dedupe | ~13–26 | `retrieve.fuse` |
      | extractive span answer | ~5–25 (P50 ~24) | `generate.extractive` |
      | **retrieval+extractive core (P50, single cold query)** | **~650ms** | — |
      | full pipeline total (P50, single cold query) | **~650ms** | — |
      | query→result cache hit (P50) | **~6ms** | `cache` |
    - **Concurrency column caveat:** all rows above are **single cold query (conc 1)**;
      the concurrent-load rows below are measured at their stated concurrency and are
      NOT comparable to the single-query rows (serial-embed server). Report with the
      concurrency attached. *Since the Aug 2026 benchmark: the request path now bounds
      in-flight embeds with `asyncio.Semaphore(embed_concurrency=2)` (lock, Phase 6)
      and emits a live SSE `queued` stage event when the gate is contended — verified
      6/8 concurrent distinct requests showed `queued`, all streams completed, none
      stalled. Cache hits bypass the gate entirely.*
    - **Concurrent-load finding (Aug 2026):** under 8-way concurrency with 140
      *distinct* cold queries, retrieval P50 inflates to ~3.5s because Ollama's
      `/api/embed` endpoint serves distinct embed requests **serially** — queue
      wait dominates wall-clock (140 × ~200ms serialized). This is a physical
      property of a single embed server under N distinct cold queries, not a code
      defect. Repeats collapse to the cache (P50 ~6ms, hit rate 1.0 on the warm
      pass). Same queries replayed: total P50 5.5ms, P100 25ms.
    - **Load test (Aug 2026, 140 queries, concurrency 32):** cold P50 12.45s /
      P100 29.76s / 2.16 qps — the serial-embed ceiling at 32-way saturation with
      all-distinct queries. Warm (cache-hit) P50 11.0ms / P100 16.8ms / 2019 qps.
      **The defensible P100 story is the warm path (≤25ms) plus the honest cold
      ceiling with the serial-embed caveat attached — never a gamed number.**
    - **The 200ms-compliant output is the EXTRACTIVE span answer on a CACHE HIT
      (~6ms) or on a single cold query whose 650ms is dominated by the embed
      (~200ms) + Neo4j search (~400ms).** The LLM answer streams afterward and is
      reported separately. The naive "<200ms core" budget is NOT met cold; the
      honest framing is "~650ms cold single-shot, ~6ms warm/cached" — and repeat
      queries (the demo flow: user repeats/refines) hit the cache.
    - Full voice→answer latency is reported separately (STT dominates; Sarvam claims
      <250ms median) and surfaced in the demo UI.
    - Percentiles are measured by the benchmark harness over ~100-200 queries (mixed
      languages × query_types), never a single best-case run. Live traffic is also
      sampled continuously by `GET /v1/telemetry` (in-memory P50/P70/P100).
    - **Phase-2 throughput finding (Aug 2026):** indexing is NOT bound by Ollama once
      batched, but batch size interacts badly with Ollama's Windows tokenize runner.
      Measured on a freshly-restarted server with `num_batch=1024` + `keep_alive=5m`:
      - `batch=128`: **112-132 chunks/sec, zero tokenize flakes** over 60 consecutive
        calls (≈ 3x faster than the original 37/sec) — this is the lock.
      - `batch=256`: nominally ~112-140/sec but triggers Ollama's intermittent
        `tokenize` 400 (connection refused on the runner's random port) on ~1 in 3
        calls; sustained throughput collapses to ~49/sec.
      - `batch=384+`: 400s become fatal even with retries.
      So `embed_batch_size=128` is fixed; resilience (8 retries, backoff to 8s,
      warm-up embed in the indexer) is in `_OllamaBackend`. Index time is dominated
      by embed latency + Neo4j writes, not dataset streaming.
    - Industry context (demo narrative): production voice agents run P50 1.4-1.7s /
      P95 4-5s (Hamming 2026 voice-eval guide); an honest full-path <700ms is
      exceptional. We also run a concurrent load test so P100 stays defensible.

    ## Guardrails (know when NOT to answer)
    - **Safety gate** — fast keyword blocklist (EN + Indic scripts) for
      hate/harassment/self-harm/illegal/explicit; optional LLM judge refines
      borderline cases → block + reason.
    - **Off-topic gate** — heuristic gate (too-short / chit-chat / opinion /
      non-interrogative) refuses; optional LLM judge for borderline cases.
    - **Grounding gate** — if top retrieval score < threshold (0.78) → refuse
      ("मुझे इसका उत्तर देने के लिए पर्याप्त संदर्भ नहीं मिला।").
    - **Locked-change callout (Aug 2026) — grounding threshold recalibrated
      0.35 → 0.78.** The 0.35 threshold was calibrated for a wider cosine range and
      NEVER fires with qwen3-embedding (decoder-backbone anisotropy: cosine
      compresses into a 0.70–0.93 band — confirmed by arXiv:2209.00218,
      arXiv:2606.29571: rank-based metrics beat cosine on such encoders, but RRF
      top-score (max ~0.033) and BM25 (unbounded, INVERTED for out-of-domain
      queries: fabricated-fact queries scored HIGHER than genuine — AUC 0.11) do
      not separate). Measured on 140 genuine + 25 fabricated queries: top-1 cosine
      separates genuine from fabricated at AUC 0.998/1.000 (fabricated max 0.783,
      genuine min 0.777); threshold 0.78 = 5.0% false-refusal on genuine, 92%
      fabricated-caught; 0.80 = 10.0% false-refusal, 100% caught. Chose 0.78
      (floor false-refusal; residual 2/25 fabricated slip to the faithfulness net).
      > **Note (Aug 2026): the 0.78 threshold is calibrated for qwen3-embedding's
      > cosine band.** After the Vertex re-embed, cosine distributions shift —
      > recalibrate 0.78 against gemini-embedding-001 on the same 140+25 set and
      > re-pin `backend/tests/test_guardrails.py` BEFORE deploying the live link.
    - **Faithfulness check** — answer must cite passages inside the retrieved set;
      token-overlap heuristic + optional LLM entailment; ungrounded answers are
      converted to refusals. Extractive answers (verbatim sentences extracted from
      a retrieved passage) are faithful by construction and skip the overlap
      heuristic — a short extracted sentence trivially shares <2 tokens with its
      full passage and was being false-refused.
    - Refusals are ordinary `PipelineResult` outputs with `mode="refusal"` and a
      `refusal_reason` — never exceptions.
    - **All default guards are deterministic** (LLM judge off by default) — the
      judges are optional refinements, not the decision layer.

    ## Deep Path Indexing Scope
    The LightRAG deep path is index-time-only cost (Ollama on RTX 5050) and indexes
    a curated slice (~300 queries) as pseudo-documents; the full 150-200k-passage
    corpus stays on the Tier-1 Neo4j index. LightRAG is an optional dependency —
    if missing or if Neo4j is unreachable, every deep-path method degrades
    gracefully and the pipeline uses the fast path only (the app never crashes on a
    missing optional dependency).

    Tier 2 is **not the hero path**: every demo answer routes through Tier 1 first;
    Tier 2 is additive (relational/multi-hop questions, graph visual). If Tier 2 is
    slow or fails during the demo, Tier 1 has already answered — the demo never
    stalls (JetBrains judging: demo reliability beats architecture).

    ## Harness
    - Pipeline = composable, typed stages (`Pydantic`); structured I/O end to end.
    - Retries with exponential backoff + jitter per external call; circuit breaker;
      provider fallback chains (STT: Sarvam→whisper; LLM: Gemini→Ollama→Groq→OpenAI).
    - OpenTelemetry spans per stage + request-ID tracing; `Langfuse`-compatible
      logging (optional).
    - Benchmark runner exports per-stage + end-to-end latency percentiles to JSON.
    - > **Locked-change callout (Aug 2026) — `call_resilient` coroutine bug fixed.**
      > `call_resilient` built `inner = retry_with_backoff(...)` (an async def → a
      > coroutine) then handed it to `CircuitBreaker.call`, which does `await fn()`
      > on it → `TypeError: 'coroutine' object is not callable`. Since `providers.py`
      > always passes a breaker, **every generation call silently failed** through
      > the failover path; the 0.921 benchmark answer-rate never saw it because the
      > harness runs `mode="extractive"` (no LLM). Fixed in `backend/core/retry.py`
      > by wrapping the retry in an `async def inner()` and calling `await inner()`.
      > Regression tests in `backend/tests/test_resilience.py`. Also fixed
      > `_should_retry`: it only checked `exc.status_code`, but `httpx.HTTPStatusError`
      > keeps the status on `exc.response.status_code`, so HTTP 429/5xx were never
      > backoff-retried. Verified end-to-end: forced Gemini failure (bad key) → fell
      > through to Ollama (`llama3.2:3b`) and answered.
    - > **Rate-limit note (Aug 2026):** Gemini free tier ≈ 15 RPM / 1,000–1,500 RPD
      > for Flash-Lite (per project, not per key). The existing backoff + breaker +
      > Gemini→Ollama→Groq→OpenAI chain is the mitigation — a 429 is backoff-retried
      > (now actually detected after the `_should_retry` fix) then falls to the next
      > provider. Demo volume is well under the caps; no new rate-limiting code in
      > the generation path (request-path limiting is the API-safety layer below).

    ## API Safety & Cost Control (locked design — implementation in progress)
    The live API is a public URL; without limits, abuse drains the Gemini
    generation RPD, Sarvam STT credits, and Vertex embed budget, and inflates the
    P100 latency story. Locked layered defense (fastapi-limiter/slowapi pattern,
    verified docs):
    - **L1 Request caps (Pydantic)** — `text ≤ ~300 chars`, `audio_b64 ≤ ~4MB`,
      `lang`/`mode` as enums, reject control chars. STT budget math: 4MB ≈ ~4 min
      of audio at 16kHz/16-bit; Sarvam's ₹100 ≈ 200 min total → limits keep the
      demo within credits.
    - **L2 Rate limits (slowapi)** — per-IP `20/min · 300/hr · 500/day`; global
      `1000/day · 1500/hr` circuit-breaker; HTTP 429 + `Retry-After`. `memory://`
      storage is correct because Render is a single instance.
    - **L3 Cost gate** — **`POST /v1/benchmark` is disabled on the public
      deploy** (it would let anyone run heavy retrievals on our credits); global
      daily cap sits below the 500/day Gemini RPD; `generation_max_tokens=512`
      caps per-answer cost.
      > **IMPLEMENTED (Aug 2026):** `POST /v1/benchmark` now returns **HTTP 403**
      > unless `BENCHMARK_ENABLED=true` (default **false**, `backend/config.py`,
      > `backend/main.py`). `render.yaml` pins `BENCHMARK_ENABLED=false`. The CLI
      > harness (`python -m backend.harness.benchmark`) is the intended path for
      > real runs and is unaffected.
    - **L4 CORS** — same-origin only (frontend is served by the same nginx, so no
      external origin is needed); no `allow_origins=["*"]` on the live deploy.
      > **IMPLEMENTED (Aug 2026):** the wildcard middleware is gone. CORS is now
      > opt-in via an explicit `CORS_ORIGINS` comma list (default empty = no CORS);
      > `*` is rejected even if set (`backend/config.py`, `backend/main.py`).
    - **L5 nginx** — `client_max_body_size 8M` in `deploy/render/nginx.conf.template`
      (the 4MB audio cap has headroom; oversized uploads are rejected at the edge).
      > **IMPLEMENTED (Aug 2026):** `client_max_body_size 8M;` is in the `server {}`
      > block of `deploy/render/nginx.conf.template` (verified present).
    - Tracked as Phase A; implementation order: Pydantic caps → slowapi limits →
      benchmark gating → CORS tighten → nginx body size. **L3/L4/L5 done (Aug
      2026); L1 (Pydantic caps) and L2 (slowapi) still pending** (slowapi not yet
      in `deploy/render/requirements.txt`).

    ## AuraDB connection resilience (locked Aug 2026)
    AuraDB Free drops Bolt connections under concurrent/sustained load
    (`SessionExpired`, reproduced in the Aug-18 benchmark at concurrency 8 and in
    full-pool `eval_mrr`; sequential probes pass). Mitigation **IMPLEMENTED**:
    `Neo4jStore._run` / `_run_write` go through `_with_retry` — a bounded (4
    attempts, exponential backoff + jitter) retry that re-runs the query on a
    FRESH session/connection for `SessionExpired`, `ServiceUnavailable`, and
    `TransientError`; non-transient errors raise immediately. Regression tests in
    `backend/tests/test_neo4j_retry.py`. This is defense-in-depth; the eval
    harness still avoids hammering the free tier with unbounded concurrency.

    ## API Contract (Locked)
    - `POST /v1/ask` — JSON body `{text?, audio_b64?, lang?, mode?}` → streams SSE
      events: `transcript → intent → guard → retrieval → answer → done`
    - `POST /v1/ask/text` — same body, JSON envelope (for programmatic clients)
    - `GET /v1/health` · `GET /v1/telemetry` · `POST /v1/benchmark` (off on public
      deploy) · `GET /v1/graph`
    - Response envelope always includes `request_id`, `stages[]` latencies,
      `citations[]`, `grounding_score`, `guardrail` decisions.

    ## Evaluation & Credibility (research-backed)
    - **Locked-change callout (Aug 2026) — the disjoint-holdout MRR design was
      INVALID and is retired.** Measured evidence: the sampled corpus has
      214,018 passage records / 212,949 unique normalized texts / only **342
      recurring passages (~0.16%)**; and **0/224 holdout gold passages were present
      in the full sample index**. A held-out query's `is_selected` passage is
      essentially never in the pool, so "MRR against a disjoint holdout" scores ~0
      for every query — not a retrieval failure, a broken design. It is never
      reported; nothing approximates it.
    - **Retrieval eval — three locked metrics** (`backend/harness/eval_mrr.py`):
      1. **`in_index_mrr` (PRIMARY)**: every indexed query whose `is_selected`
        passage is in the `passage_natural` pool; embed the query and retrieve
        over the **full `passage_natural` pool** (the full sample, indexed
        separately from the deploy subset). Legitimate because `passage_natural`
        nodes carry no query text — the pool does not reveal the gold, so it is
        structurally analogous to the paper's Table 2 (known gold retrieved from a
        pool). Report MRR@10 / Recall@10 / nDCG@10 per language and overall.
      2. **`query_anchored_mrr` (REPORTED SEPARATELY, LABELED LEAKY)**: same
        computation over `query_anchored`. That namespace embeds query+passage
        together, so scores are inflated by construction; the leakage caveat is
        attached in both JSON and Markdown. Never presented beside `in_index_mrr`
        without it.
      3. **`held_out_mrr` (SECONDARY, small-N, genuinely disjoint)**: holdout
        queries whose gold text matches one of the recurring passages (present in
        ≥2 indexed queries). Gold-in-pool = True by construction (coverage 1.0
        asserted in tests). This is the ONLY true unseen-query number in the
        report; expected N is tiny. If N=0 it is reported as N=0 and
        `[not measured]`, never estimated.
    - **Measured eval (run `20260815-133028`, full `passage_natural` pool 213,928
      chunks; N=12,922 indexed queries with gold; traceable to
      `eval/mrr_20260815-133028.json/.md`)**:
      - **`in_index_mrr` (PRIMARY)** — overall: **MRR@10 0.298 · Recall@10 0.546 ·
        nDCG@10 0.357** (Recall@10 == gold found in top-10: 7,051/12,922). Per
        language (N=923 each): hi 0.362, ur 0.380, te 0.344, ne 0.317, mr 0.312,
        ml 0.305, gu 0.279, kn 0.278, pa 0.275, or 0.271, ta 0.266, sa 0.283,
        as 0.252, bn 0.248. *(No separate "coverage" field: top-10 presence IS
        Recall@10. Pool membership is ~1.0 by construction — every evaluated gold
        was itself indexed. Only held_out_mrr carries a coverage field, meaning
        gold-in-pool by recurrence construction.)*
      - **`query_anchored_mrr` (LEAKY)** — overall: MRR@10 0.045 · Recall@10 0.100 ·
        nDCG@10 0.058. **POOL-SIZE MISMATCH** flagged in the JSON/MD: that
        namespace holds only the curated subset (28,518 chunks), so most full-sample
        golds are absent from the pool. Never compared with `in_index_mrr`.
      - **`held_out_mrr`** — **N=0 — `[not measured]`**. Recurrence recomputed
        against the FULL pool (21,420 records / 1,530 distinct query_ids): **342
        recurring passage texts**, and **0** of 350 holdout records (25 distinct
        query_ids) have a gold whose text recurs → the genuine unseen-query eval is
        empty by construction of the data, not by harness failure. No score is
        implied.
    - **Delta vs paper Table 2 (bge-m3, arXiv:2506.01615)** — same MRR@10 metric,
      NOT the same benchmark (see pool caveat below). Ours (in_index_mrr) vs paper:
      Hindi 0.362 vs 0.52, Bengali 0.248 vs 0.49, Tamil 0.266 vs 0.49, Telugu 0.344
      vs 0.50, Odia 0.271 vs 0.45, Assamese 0.252 vs 0.46. Sanskrit: **no published
      baseline** (absent from Table 2). Every language trails the paper baseline —
      expected: qwen3-embedding:0.6b is a 0.6B local embedder vs bge-m3 (~570M but
      multilingual-pretrained at scale), and our pool is machine-translated
      MSMARCO-XI (no human verification). This is reported as a measured weakness,
      not hidden. **The Vertex re-embed is expected to narrow this gap (MIRACL
      shows gemini-embedding-001 > qwen3-embed on Indic); the delta is re-measured
      after re-embed, never assumed.**
    - Every leaderboard number is traceable to a harness run (eval/mrr_*.json +
      .md, run timestamp cited) or to the paper's Table 2 (arXiv:2506.01615).
      Anything not yet measured is labeled **`[not measured]`**.
    - **Pool caveat (same metric, NOT the same benchmark)**: our eval pool is
      **machine-translated MSMARCO-XI with no human verification**; the paper's
      Table 2 uses a **hand-verified 1,000-query IndicMSMarco** benchmark. We
      report the same MRR@10 metric, but a head-to-head against the paper
      overstates reproducibility — the README says so explicitly. Measured delta
      above (Section Evaluation) is honest context, not a competitive claim.
    - **Sanskrit baseline**: the paper's Table 2 has **no Sanskrit row at all**
      (it appears only in the paper's training-data table). The leaderboard shows
      **"no published baseline"** for Sanskrit — never a blank or an interpolated
      number. Measured Sanskrit in_index_mrr: **MRR@10 0.283** (N=923).
    - Paper bge-m3 baselines where published: Hindi 0.52, Bengali 0.49, Tamil 0.49,
      Telugu 0.50, Odia 0.45, Assamese 0.46. Full-paragraph passage structure (the
      paper's own finding) validates our `passage_natural`/`query_anchored`
      chunking.
    - **Answer eval**: faithfulness, correctness vs the ground-truth `Answer`
      (lexical + optional LLM judge), and **refusal precision** (guards fire on
      off-topic/low-grounding, stay silent on on-topic).
    - **Honesty as a feature**: publish where we're weak (low-resource languages
      track the paper baseline) and report cache hit/miss + warm/cold latency
      distributions. Measured claims beat gamed numbers with judges.

    ## Reviewer Feedback (Aug 2026) — status
    - [x] Gemini Flash-Lite model id stale → updated to `gemini-3.5-flash-lite`
          (GA 2026-07-21, verified).
    - [x] Neo4j vector index dim explicit → `vector.dimensions` = `embed_dim` (1024)
          set explicitly in `neo4j_store.py` CREATE VECTOR INDEX.
    - [x] Sarvam REST `/speech-to-text` = final transcripts only, no interim/partials
          (verified). **DECISION (Aug 2026): keep Sarvam REST final-transcript-only
          for production STT; DO NOT adopt WS `/speech-to-text/ws` for mid-utterance
          partials.** Sarvam WS returns chunks, not true word-by-word partials —
          the "speculative retrieval on partial transcript" framing is reframed to
          **"speculative retrieval triggered on VAD/end-of-speech"** (a partial-
          transcript prefetch was never achievable with this provider). Architecture
          + phase-2 latency-contract paragraphs use the reframed framing.
    - [x] README benchmark caveat → done for MSMARCO-XI machine-translated vs the
          paper's hand-verified 1,000-query IndicMSMarco; Sanskrit has no baseline
          row (13 languages only) — leaderboard shows "no published baseline".
          (README.md "Retrieval quality" + eval/mrr_*.md footer.)
    - [x] Position extractive span answer (0-10ms) as THE 200ms-compliant output,
          LLM = progressive enhancement → framing added in CONTEXT.md Latency
          Contract + demo UI copy (frontend LatencyPanel budget note +
          "extractive = 200ms-compliant output" badge in the answer card).
    - [ ] qwen3-embedding is instruction-aware → A/B query-side retrieval
          instruction (+1-5% reported); add eval row. (Re-run after Vertex re-embed.)
    - [ ] Reranker eval → add rerank on/off MRR rows to the leaderboard. MIRACL
          evidence (hybrid 0.579 > rerank-only 0.542 for non-English) suggests we
          keep RRF-only on the live path; the eval row documents it.
    - [x] Eval methodology locked (Aug 2026) → three metrics in
          `backend/harness/eval_mrr.py`: `in_index_mrr` (full `passage_natural`
          pool), `query_anchored_mrr` (labeled LEAKY), `held_out_mrr`
          (recurrence-filtered, coverage 1.0 asserted). Disjoint-holdout MRR
          retired with measured evidence (214,018 records / 212,949 unique /
          342 recurring / 0-of-224 gold coverage). See Evaluation section.
    - [x] Eval methodology locked WITH REAL NUMBERS (Aug 15 2026) → run
          `20260815-133028` recorded: `in_index_mrr` overall MRR@10 0.298
          (N=12,922, Recall@10 0.546), per-language table + paper delta in
          CONTEXT.md Evaluation; `query_anchored_mrr` MRR@10 0.045 (LEAKY +
          pool-size mismatch flagged); `held_out_mrr` N=0 — [not measured]
          (full-pool recurrence = 342 texts, 0 holdout gold recurs).
          Artifacts: `eval/mrr_20260815-133028.json/.md`.
    - [ ] Social checklist → every member on IG/X/LinkedIn, ≥1 public Instagram,
          every post `#RAGInGoa`, videos of the live demo.
    - [x] Rotate shared HF token — DONE Aug 14 2026. Old token revoked on HF; new token
          lives only in `.env` (gitignored, never committed — task_2 is untracked, and
          `git log --all -p` shows zero history matches for the old token).
    - [ ] **Hosting locked to Render free tier** (Aug 2026) → render.yaml written;
          AuraDB subset built & verified (59,296 nodes); **deploy itself is BLOCKED
          on the Vertex service-account credentials** (re-embed + eval + threshold
          recalibration first). Cloud Run + spare laptop retired as fallbacks.
    - [ ] **API-safety Phase A** (Pydantic caps → slowapi limits → benchmark gating
          → CORS tighten → nginx body size) — design locked in CONTEXT.md, code not
          yet implemented. Adds `slowapi` to requirements (not yet present).
    - [ ] **Vertex re-embed + eval rerun + grounding threshold recalibration** —
          blocked on service-account JSON; do NOT deploy live until done.
    - [ ] README/CONTEXT.md sync (Aug 2026) — six namespaces, Render hosting,
          Vertex embedder, API safety, eval leaderboard, cost/credit math, 79 tests.

    ## What "Done" Looks Like
    A judge opens the live link, presses the mic, asks "विमान किसने बनाया?" (Hindi,
    "who invented the airplane?"), sees a live transcript, a grounded answer with
    highlighted source passage, a grounding score, guardrail status, and a live
    P50/P70/P100 latency panel — then asks something off-topic and the system
    politely refuses with a reason. Then they fire a cross-lingual beat ("Meri crop
    ke liye konsa fertilizer use karoon?") and get a Hindi answer from an English
    passage — and the README shows an MRR leaderboard tied to the IndicRAGSuite
    paper baseline per language. The live link is rate-limited (no abuse), stays up
    all demo hours (keep-alive within the Render free-tier cap), and reports the
    ~60s cold start honestly.
