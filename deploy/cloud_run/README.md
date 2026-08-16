# VakRAG on Cloud Run (free tier)

Runs the VakRAG FastAPI backend with Ollama (`qwen3-embedding:0.6b`) in a single
container, the same immutable embedding decision as the HF Spaces image (see
CONTEXT.md). AuraDB Free is the retrieval store.

## Why Cloud Run (vs HF Spaces)

- HF Spaces **Docker SDK is paid-only on the free tier** (definitive; PRO
  required). This image is the direct port of that container to Cloud Run.
- Cloud Run free tier: 2M requests/mo, 180K vCPU-seconds, 360K GiB-seconds,
  1 GB egress NA/mo. **Free-tier credits only apply in `us-central1`,
  `us-east1`, or `us-west1`** — deploy there or you get billed.
- Scale-to-zero by default (min instances 0) + request-based billing = $0 when
  idle. Do NOT set `--min-instances` (bills an idle rate) and do NOT use
  `--no-cpu-throttling` (instance-based billing).

## Files

- `Dockerfile` — FastAPI + Ollama + baked `qwen3-embedding:0.6b`, listens on
  `$PORT` (Cloud Run contract; default 8080).
- `entrypoint.sh` — starts Ollama, polls readiness (max 60s), execs uvicorn on
  `$PORT`. App pre-warms embeddings at startup so the first request is fast.
- `requirements.txt` — request-path deps only.

## Build

```powershell
# Build context MUST be the repo root (backend/ lives there). The Dockerfile
# already uses repo-root-relative COPY paths.
docker build -t vakrag-cloudrun -f deploy/cloud_run/Dockerfile .
```

The image reuses the same locked decisions as `vakrag-space:local` (Ollama +
baked `qwen3-embedding:0.6b`), differing only in the port contract ($PORT vs
7860) and the pip install strategy (online vs offline wheels).

## Local verification (before any Cloud Run deploy)

```powershell
# Uses .env.docker (comment-free, --env-file safe) — NOT .env. Docker's
# --env-file parses literally, so .env's inline comments (e.g. the ones on
# EMBED_BATCH_SIZE, OLLAMA_EMBED_NUM_BATCH) would crash pydantic config with
# "Input should be a valid integer". .env.docker is auto-derived (gitignored).
docker build -t vakrag-cloudrun -f deploy/cloud_run/Dockerfile .
docker run --rm -p 8080:8080 --memory 4g --env-file .env.docker -e PORT=8080 vakrag-cloudrun:local
curl.exe http://localhost:8080/v1/health   # expect {"status":"ok","neo4j":true,...}
```

Regenerate `.env.docker` from `.env` if `.env` changes:
```powershell
# strips inline '# comment' text after every value; keeps full-line comments/blanks
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#') { $_ }
  elseif ($_ -match '^\s*$') { '' }
  elseif ($_ -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
    "$($Matches[1])=$((($Matches[2] -replace '^\s+','') -replace '\s*#.*$','').TrimEnd())"
  }
  else { $_ }
} | Set-Content -Encoding ASCII .env.docker
```

## Deploy (requires: gcloud CLI + billing account, see parent notes)

```powershell
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set run/region us-central1

# Tag and push to Artifact Registry (create repo once):
gcloud artifacts repositories create vakrag --repository-format=docker --location=us-central1
docker tag vakrag-cloudrun us-central1-docker.pkg.dev/<PROJECT_ID>/vakrag/vakrag-cloudrun
docker push us-central1-docker.pkg.dev/<PROJECT_ID>/vakrag/vakrag-cloudrun

# Deploy the service.
# MANDATORY flags: us-central1 (free tier), --memory 4Gi (Ollama+model needs it),
# --allow-unauthenticated (public so Vercel can call it). min-instances stays 0.
# CPU: 1 vCPU is default; Ollama is CPU-only here.
gcloud run deploy vakrag `
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/vakrag/vakrag-cloudrun `
  --region us-central1 `
  --memory 4Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 2 `
  --allow-unauthenticated `
  --timeout 300 `
  --set-env-vars NEO4J_URI=neo4j+s://<id>.databases.neo4j.io,NEO4J_USERNAME=neo4j,PRIMARY_LLM_PROVIDER=gemini,GEMINI_MODEL=<model>,OLLAMA_FALLBACK_MODEL=llama3.2:3b `
  --set-secrets NEO4J_PASSWORD=...   # or --set-env-vars for a throwaway demo
```

Secrets (`GEMINI_API_KEY`, `SARVAM_API_KEY`, `NEO4J_PASSWORD`) should be set via
Secret Manager or env vars. For a demo-day deployment, `--set-env-vars` with the
values from `.env` is acceptable; do not commit them.

Cold-start note: scale-to-zero means the first request after idle spins a fresh
instance (image ~6.7GB, Ollama + model load). This is the measured demo-day
risk — keep `--min-instances 0` for cost but expect the first hit to be slow.
`--timeout 300` covers model-warmup + first answer.

## Verify

```powershell
# Health (Neo4j + embeddings readiness)
curl.exe https://<SERVICE_URL>/v1/health

# One SSE ask
curl.exe -N -X POST https://<SERVICE_URL>/v1/ask `
  -H "Content-Type: application/json" `
  -d '{"text":"<assamese question>","lang":"as"}'
```

## Endpoints

- `POST /v1/ask` — SSE stream of stage events -> final answer
- `POST /v1/ask/text` — JSON convenience
- `GET /v1/health` — Neo4j + embedding readiness
- `GET /v1/telemetry` — live latency snapshot
- `GET /v1/graph` — knowledge-graph snapshot

## Frontend

Point Vercel (or `frontend/` Vite dev) at `VITE_API_BASE=https://<SERVICE_URL>`.
CORS is already wide-open in `backend/main.py` (`allow_origins=["*"]`).