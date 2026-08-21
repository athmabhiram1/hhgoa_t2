# VakRAG — Deployment Plan (Render free + AuraDB Free)

Free-tier deployment runbook for the live demo link, derived from the deep
research (Aug 2026). Every cost is an estimate; every constraint is cited.
Follow the order: **blockers → build → deploy → verify → operate**.

---

## 0. Target state

```
Judge opens https://<app>.onrender.com → presses the mic →
Sarvam STT → guards → Tier-1 Vertex embeddings → AuraDB retrieval →
RRF → extractive span + streamed Gemini answer → P50/P70/P100 panel.
Off-topic/unsafe/low-grounding questions are refused with a reason.
Rate-limited, kept alive during demo hours, honest ~60s cold start.
```

| component | choice (locked) | free-tier reality |
|---|---|---|
| App | ONE Render web service (nginx + uvicorn) | 750 hrs/mo, 15-min idle spin-down, ~60s spin-up, local FS ephemeral |
| Graph DB | AuraDB Free `c551c599` (neo4j+s://c551c599.databases.neo4j.io) | 200K nodes / 400K rels cap (verified); **72h-inactivity auto-pause; deleted ~90 days paused (blog) / "30 days no activity" (FAQ)** |
| Embeddings | `gemini-embedding-001` via **Gemini API** (`EMBED_BACKEND=gemini`, AI Studio key), 1024-dim cosine | **free-tier quota (RPM/TPM/RPD), NOT the GCP credit** — see §2; rate-limited, 429s on exhaustion |
| Generation | Gemini `gemini-3.5-flash-lite` (AI Studio key) | ~15 RPM / 1,000–1,500 RPD free tier |
| STT | Sarvam `saaras:v3` | ₹100 ≈ 200 min credits (Starter, 60 req/min) |

---

## 1. Cost estimate — indexing ALL runs (CORRECTED Aug 2026: free tier, not GCP credit)

**Correction:** the deploy embedder is now the **Gemini API backend**, which is a
**separate billing bucket from the GCP $300 credit** — AI-Studio-key usage is
either free inside its own quota or 429s; it never bills the GCP project. The
dollar figures below are therefore only what the OBSOLETE Vertex path *would*
have cost on the credit; the actual runs cost **$0** and are constrained by
**rate**, not money. (Vertex `:predict` = $0.15/1M input tokens, 1 text/call,
`VERTEX_EMBED_CONCURRENCY` parallelism. The gemini path = 100 texts per
`batchEmbedContents`, fully serial → ~591 calls for the 59,073-chunk subset.)

| run | chunks | API calls (gemini @100/batch) | GCP-credit cost (Vertex path, obsolete) |
|---|---|---|---|
| Progressive pilot gate (ONE run: 8 q/lang × 14 langs → `passage_natural`) | ~1,120 | ~12 | ~$0.01 |
| AuraDB deploy subset re-embed (passage_natural 28,518 + query_anchored 28,518 + passage_en 2,037 = **59,073**) | 59,073 | ~591 | ~$0.49 (CONTEXT: ~$0.25; range $0.25–$1.50) |
| Full local eval-pool re-embed (`passage_natural`, **213,928** chunks, for the MRR rerun) | 213,928 | ~2,140 | ~$1.76 (CONTEXT: ~$2; range $2–$5) |

- **The real budget is RATE, not money.** ~591 calls / ~3.3M tokens for the
  subset fits inside a day even at a conservative 1,000 RPD / 30K TPM free tier;
  the full local pool (~2,140 calls) may exceed a tight RPD → either run it
  across days, batch harder, or link a billing account for Tier 1 (no spend).
  Live per-project numbers: `aistudio.google.com/rate-limit` (signed-in).
- **Time estimate (rough):** subset ≈ 25–40 min, full pool ≈ 1.5–2.5 h — but
  paced by the free tier's serial batching and any 429 backoff, not by an 8-wide
  thread pool (gemini path has none).
- **The one real lever — a chunk-embedding cache.** CONTEXT.md "Known gap"
  (Aug 2026): `{chunk_id: vec}` is deterministic but never persisted, so every
  re-point to a new DB target re-embeds from zero. Adding a Parquet/JSONL
  sidecar during ingestion would make EVERY re-deploy instant instead of
  re-burning free-tier quota. Recommended as the first post-demo improvement,
  not a pre-demo requirement.
- Not in this table (Ollama on RTX 5050, free): LightRAG deep-path graph
  extraction. Not in this table (demo-time): Sarvam STT (₹100) + Gemini
  generation (free RPD) + gemini query embeddings (free-tier RPD — negligible).

---

## 2. Embedding credentials — Gemini API key (the deploy embedder, Aug 2026)

**Decision (locked): the deploy embedder is the Gemini API backend
(`EMBED_BACKEND=gemini`, `gemini-embedding-001` via `batchEmbedContents`),
authenticated by `GEMINI_API_KEY` — NOT Vertex AI.** Rationale, verified Aug
2026:

- **Render cannot run Vertex.** Render runs a generic Docker container with NO
  GCP metadata server and no interactive `gcloud`, so Vertex requires an inline
  `VERTEX_CREDENTIALS` service-account JSON.
- **The org policy blocks SA-key creation.** `orgpolicy.googleapis.com` is not
  enabled on the free-trial project (`project-6d0a199b-f3a1-4b7d-9b0`); the
  `gcloud org-policies describe` path fails and the console only surfaces the
  policy (role grant + key creation are not reachable via API).
- **Measured vector-space parity:** Gemini API `batchEmbedContents` (no
  `task_type`) returns **byte-identical** vectors to Vertex `RETRIEVAL_QUERY`
  (`max_abs_diff = 0.0`). So the Gemini API backend is self-consistent for both
  indexing and querying AND lands in the same space as Vertex query vectors.
  (Vertex `RETRIEVAL_DOCUMENT` differs by ≈0.17 — the model's document/query
  tuning; only relevant if the Vertex backend were used for indexing.)
- **Cost is NOT the constraint (correction, Aug 2026):** the Gemini API /
  `GEMINI_API_KEY` is a **separate billing bucket from the GCP $300 credit** —
  Vertex spend is the only thing that touches the GCP credit. Gemini API usage
  is **free inside its own free-tier quota** (per-project RPM/TPM/RPD; live
  numbers only on the signed-in `aistudio.google.com/rate-limit` page) or
  fails with HTTP 429. There is no dollar spend to track; the real constraint
  is RATE. The gemini embed backend batches 100 texts per call, fully serial
  (~591 calls for the full 59k-chunk re-embed), so it is safe under a
  conservative 1,000 RPD. Fallback if the free tier is too tight: link a
  billing account (no spend required) → Tier 1, meaningfully higher caps.

What to set (all in the Render dashboard as env vars; secrets `sync: false`):

| variable | value |
|---|---|
| `EMBED_BACKEND` | `gemini` (already in `render.yaml`) |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey (already works; verified) |
| `GEMINI_EMBED_DIM` | `1024` (locked schema, do NOT change) |
| `GEMINI_EMBED_BATCH_SIZE` | `100` |

Local sanity check (any backend, from repo root):
```powershell
python -c "from backend.config import Settings; from backend.retrieval.embeddings import EmbeddingService; s=EmbeddingService(Settings(embed_backend='gemini')); v=s.embed_one('namaste'); print('OK dim=',len(v),'first5=',[round(x,4) for x in v[:5]])"
```
Expected: `OK dim=1024 first5=[...]`.

> **Vertex AI remains usable locally** via ADC (`GOOGLE_APPLICATION_CREDENTIALS`
> → `google.auth.default()`), and its query vectors are identical to the Gemini
> API backend — but it is NOT the deploy path and must not be used to index the
> AuraDB pool (a `RETRIEVAL_DOCUMENT` index + `gemini` query vectors would mix
> two spaces). **Index the deploy pool with the `gemini` backend.**

Other secrets (all set as Render dashboard env vars, `sync: false`):
`NEO4J_PASSWORD` (AuraDB console → instance `c551c599`), `SARVAM_API_KEY`
(Sarvam dashboard), `HF_TOKEN` (https://huggingface.co/settings/tokens —
rotated Aug 14 2026).

---

## 3. Pre-deploy checklist (blockers first)

- [x] **Embedding credentials — Gemini API key (DONE Aug 2026).** Deploy
      embedder is `EMBED_BACKEND=gemini` (GEMINI_API_KEY already set + verified).
      The former Vertex SA-key blocker is retired: org policy blocks SA-key
      creation and Render has no metadata server — see §2. Index the deploy pool
      with the **gemini** backend, never Vertex (document/query space mismatch).
- [ ] **Re-embed (gemini) + MRR rerun** — re-embed the AuraDB subset (~59,073
      chunks, §1) with the `gemini` backend and rerun `in_index_mrr` on the full
      pool. Expected: MRR lifts vs the 0.298 qwen3 baseline (MIRACL:
      gemini-embedding-001 > qwen3-embed on Indic). **Never assume the lift —
      measure it.**
- [ ] **Grounding threshold recalibrated** — 0.78 is calibrated for qwen3's
      0.70–0.93 cosine band; the gemini cosine band shifts. Re-pin on the same
      140+25 set, update `guard_grounding_threshold` and
      `backend/tests/test_guardrails.py`.
- [ ] **Embedding-drift gate** — after the re-embed, MRR@10 must not regress
      >10% vs the qwen3 baseline (watch −5% / investigate −10% / reindex −15%).
      Promote only if `MRR@10 ≥ 0.298 × 0.90 ≈ 0.268`.
- [ ] **Progressive index gate** — the subset build runs
      `--progressive --gate-batch 8 --gate-threshold 0.40`; Recall@10 ≥ 0.40 on
      the pilot, or STOP and improve (never force).
- [x] **nginx body size (DONE Aug 2026).** `client_max_body_size 8M;` is in the
      `server {}` block of `deploy/render/nginx.conf.template`.
- [ ] **API-safety Phase A (partial — L3/L4/L5 done, L1/L2 pending):**
      `POST /v1/benchmark` now returns 403 unless `BENCHMARK_ENABLED=true`
      (default off; render.yaml pins `false`), CORS is same-origin by default
      (`CORS_ORIGINS` allowlist opt-in, no `*`), nginx 8M is in place. STILL
      PENDING: Pydantic request caps (L1) and slowapi rate limits (L2) —
      `slowapi` is NOT yet in `deploy/render/requirements.txt`.
- [ ] `python -m compileall backend` clean; targeted pytest green.

---

## 4. Build & index (local, Windows)

From repo root (PowerShell). Two builds exist: the **AuraDB deploy subset** and
the **full local eval pool** (local Docker Neo4j, no cap).

```powershell
# 1) Sample + holdout (validation split, all 14 langs)
python -m backend.ingestion.cli --sample --holdout

# 2) Full local eval pool (no node cap locally) — keep for the MRR rerun
python -m backend.ingestion.cli --index

# 3) AuraDB deploy subset — 204 queries/lang into the 3 query-path namespaces,
#    gated by the progressive quality gate
python -m backend.ingestion.cli --index --progressive --gate-batch 8 --gate-threshold 0.40 `
  --queries-per-lang 204 --namespaces passage_natural query_anchored passage_en

# 4) MRR rerun after the gemini re-embed (drift gate)
python -m backend.harness.eval_mrr
```

Gate result → `eval/gate_<ts>.json`; MRR → `eval/mrr_<ts>.json/.md`. On gate
failure (`SystemExit(2)`) the CLI stops — improve the embed/chunking/threshold,
never `--force` a bad index.

---

## 5. Deploy to Render

1. Push to `github.com/athmabhiram1/hhgoa_t2` (render.yaml → `deploy/render/`,
   `dockerContext: .`).
2. Render dashboard → **New → Blueprint** → select the repo → `render.yaml` →
   Create. Region **oregon**, plan **free**, `autoDeploy: true`,
   `healthCheckPath: /v1/health`.
3. Set secrets in the service → Environment (all `sync: false`):
   `NEO4J_PASSWORD`, `GEMINI_API_KEY`, `SARVAM_API_KEY`. `EMBED_BACKEND=gemini`,
   `STT_PROVIDER=sarvam`, `PRIMARY_LLM_PROVIDER=gemini`,
   `GEMINI_MODEL=gemini-3.5-flash-lite` are set by the blueprint.
4. Watch the first build log (≈ 5–8 min: python deps + `npm ci` + vite build).
5. **Smoke test the public URL** (see §7). Fix the nginx body-size gap BEFORE
   this — a `413` on audio is the first thing a judge would hit.

---

## 6. Runtime operations (free-tier survival)

- **Keep-alive (CRITICAL — protects BOTH Render AND AuraDB).** cron-job.org
  pings `/v1/health` 6am–midnight IST (≈560 hrs/mo, under the 750h cap). The
  health handler runs `verify_connectivity()` against bolt://, so **every ping
  also resets AuraDB's 72h inactivity clock** and the Render idle timer. Overnight
  idle ≈ 6h — far under both thresholds. **If this job is removed, the demo link
  spins down AND the graph DB eventually gets deleted.**
- **AuraDB Free lifecycle (verified):** auto-pauses after **72h inactivity**;
  paused instances are deleted after **~90 days paused** (Neo4j blog) / "30 days
  without activity" (FAQ page). A keep-alive that touches Neo4j is a documented
  operational requirement, not a nicety.
- **Prewarm:** `backend/harness/prewarm.py` refills the in-memory LRU after a
  cold wake so the first judge query is a cache hit (~6ms). Local FS (incl. the
  LRU) is lost on every Render spin-down/redeploy.
- **Don't exceed 750 instance-hrs/mo** — community reports Render **permanently
  suspends** free apps that repeatedly blow the cap. Daytime-only keep-alive is
  the guardrail; disable it outside demo hours.
- **Sarvam credits are finite** (₹100 ≈ 200 min). L1 caps (4MB audio) + L2
  rate limits protect them; drained credits = broken STT on Render (no
  faster-whisper in the image).

---

## 7. Post-deploy verification (the demo checklist)

Run against the LIVE URL:
- [ ] Mic → Hindi "विमान किसने बनाया?" → transcript + grounded answer + citations
      + live P50/P70/P100.
- [ ] Off-topic question → polite refusal with `refusal_reason`.
- [ ] Cross-lingual "Meri crop ke liye konsa fertilizer use karoon?" → Hindi
      answer from an English passage (passage_en arm).
- [ ] Repeat the same question → cache hit, ~6ms.
- [ ] Relational question → Tier 2 escalation (graph visual) or graceful Tier-1
      fallback.
- [ ] Upload a ~4MB audio clip → NOT a 413 (nginx body-size fix present).
- [ ] `/v1/health` responds from a cron-job.org run; telemetry on `/v1/telemetry`
      shows live percentiles.
- [ ] 10 rapid requests → 429 after the per-IP limit (Phase A).

---

## 8. Rollback & fallbacks

- **Rollback:** Render redeploy of the previous successful commit (one click /
  `autoDeploy` history). Index rollback = `--force` re-embed against the same DB
  or re-point to the previous namespace — Neo4j holds one namespace per tier.
- **Documented ad-hoc fallbacks (NOT the primary path):** Cloud Run + the spare
  Ubuntu laptop. Koyeb free tier (one 512MB/0.1vCPU web service, scale-to-zero
  after 1h idle, no instance-hour cap) is a verified secondary if Render free is
  suspended — costs the same (needs the same env vars).

---

## 9. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Render free suspension (>750h or abuse) | low–med | dead demo link | daytime-only keep-alive; Koyeb fallback ready |
| AuraDB deleted (no activity 30–90d) | low | full re-index | keep-alive touches Neo4j via /v1/health |
| Grounding threshold stale after Vertex re-embed | high if skipped | false refusals / unsafe accepts | recalibrate + re-pin tests before deploy (§3) |
| Embedding drift (Vertex MRR < qwen3) | low | quality regression | drift gate MRR≥0.268, block promote |
| nginx 413 on audio (8M missing) | certain if unpatched | demo feature broken | §3 checklist item, fix before deploy |
| Sarvam credits drained | med (demo volume) | STT broken on Render | L1/L2 caps; 4MB audio cap |
| GCP credit expiry 2026-11-15 | fixed date | embedding stops billing free | demo happens before; then paid or Ollama |
| Vertex quota / region issues | low | :predict failures | backoff+retry; verify test_vertex_embed first |