# AGENTS.md — Build Conventions for VakRAG

## Golden rules
- **Never silently swap a locked component.** Locked decisions live in
  `CONTEXT.md`. If a change requires breaking one, update CONTEXT.md and call it
  out explicitly.
- **Latency is a measured contract, not a hope.** Every stage that touches the
  request path MUST be instrumented with a span and its latency recorded. No
  stage gets merged without a span.
- **Structured I/O everywhere.** All pipeline stages accept/emit Pydantic models.
  No ad-hoc dicts across stage boundaries. LLM outputs are parsed against
  schemas; unparseable output triggers a retry, never a crash.
- **Guardrails are not optional filters.** The pipeline must KNOW WHEN NOT TO
  ANSWER. Off-topic, unsafe, low-grounding, and unfaithful answers are real
  outputs with a `refusal` reason, not exceptions.
- **Multi-provider discipline.** STT: Sarvam → faster-whisper. Generation:
  Gemini → Ollama → Groq → OpenAI. Each external call goes through
  `retry.with_backoff` + circuit breaker in `providers.py`.

## Environment
- Windows (win32), PowerShell. Python 3.12, Node 24, Docker Desktop, RTX 5050 (8GB).
- Backend deps installed with `uv` or `venv+pip` — see README.
- Neo4j runs in Docker via `docker-compose.yml` (bolt://localhost:7687).

## Commands
- Backend dev: `uvicorn backend.main:app --reload --port 8000` (from repo root)
- Index the dataset: `python -m backend.ingestion.cli --sample --index`
- Index with the **progressive quality gate** (LOCKED workflow — never index the
  whole corpus in one shot): pilot → golden-set Recall@10 eval → continue only
  on pass. On gate failure the CLI stops with `SystemExit(2)` and an
  `eval/gate_<ts>.json` report — improve (chunking / embed backend / threshold),
  never `--force` a failing index.
  ```
  python -m backend.ingestion.cli --index --progressive \
    --gate-batch 8 --gate-threshold 0.40 \
    --queries-per-lang 204 --namespaces passage_natural query_anchored passage_en
  ```
- Run latency benchmark: `python -m backend.harness.benchmark`
- Frontend dev: `cd frontend && npm run dev` (Vite, port 5173)

## Code layout
```
backend/
  main.py            # FastAPI app + routers
  config.py          # Settings from .env (pydantic-settings)
  core/              # models.py (typed stages), tracing.py, retry.py, providers.py
  stt/               # base.py, sarvam.py, whisper_local.py
  ingestion/         # dataset.py (sampler), chunking.py, indexer.py, cli.py
  retrieval/         # embeddings.py, neo4j_store.py, fusion.py, reranker.py, graph.py
  rag/               # router.py, fast_path.py, lightrag_engine.py, prompts.py
  guardrails/        # safety.py, off_topic.py, grounding.py, faithfulness.py
  harness/           # pipeline.py, benchmark.py
```

## Verification before finishing a task
- `python -m compileall backend` — syntax gate.
- Targeted stage tests exist under `backend/tests/`; run with pytest when present.
- If you changed the request path, update the latency budget table in CONTEXT.md.
- **Never commit without being asked.**
- **Deploy must-haves (locked Aug 2026):** the progressive gate is the ONLY way
  to run a full `--index`; `client_max_body_size 8M` MUST exist in
  `deploy/render/nginx.conf.template` (currently missing — 4MB audio → 413); the
  daytime keep-alive MUST keep pinging `/v1/health` (it verifies Neo4j
  connectivity, which keeps AuraDB Free alive past its 72h auto-pause and
  30–90-day deletion windows).
