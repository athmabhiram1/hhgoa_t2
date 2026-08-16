# VakRAG demo backend for Google Cloud Run (free tier).
# Runs FastAPI (listening on Cloud Run's injected $PORT, default 8080) + Ollama
# (localhost:11434) serving qwen3-embedding:0.6b in the same container.
#
# Locked decisions this file must not silently change (see CONTEXT.md):
#   - Embeddings stay on qwen3-embedding:0.6b via Ollama. Do NOT switch the
#     embedding backend to Gemini/an API — it would break every grounding and
#     eval number in CONTEXT.md and require a reindex.
#   - Generation LLM: Gemini API primary, Ollama fallback (same chain as local).
#
# Cloud Run contract notes (see docs.google/container-contract, 2026-07-31):
#   - The ingress container MUST listen on 0.0.0.0:$PORT (injected, default 8080).
#   - The filesystem is writable but in-memory: writes count against the
#     instance memory limit and do not survive instance stop. The baked model
#     below stays readable from the image layer, so no runtime re-pull occurs.
#   - The instance must be listening within 4 minutes of start. This entrypoint
#     polls Ollama (max 60s) before handing off to uvicorn, and the app warms
#     the embedding model during startup, so the first request is fast.
#   - Deploy-time flags that MUST be set (see README.md): --region us-central1
#     (free-tier region), --memory 4Gi (default 512Mi is far too small for
#     Ollama + model), --allow-unauthenticated (public, so Vercel can call it).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Install Ollama (root) -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# --- Bake the embedding model into an image layer (survives scale-to-zero).
# OLLAMA_MODELS is pinned to a fixed path used at BOTH build and runtime, so the
# model baked here is visible to the UID 1000 runtime user with no re-pull.
# Note: at runtime the dir is on Cloud Run's writable-but-ephemeral filesystem;
# reads still resolve to the image layer, and Ollama's small state writes go to
# the in-memory overlay without issue.
ENV OLLAMA_MODELS=/opt/ollama-models
RUN mkdir -p /opt/ollama-models \
    && (ollama serve >/dev/null 2>&1 &) \
    && for i in $(seq 1 30); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done \
    && ollama pull qwen3-embedding:0.6b

# --- Python deps (request path only; heavier dev deps stay out of the image) -
# Build context is the REPO ROOT (backend/ lives there), so COPY paths are
# repo-root-relative. Wheels are pre-fetched on the build host (pip concurrency
# stalls inside the Docker network — same reason the HF image used wheels) and
# installed offline for a deterministic build.
WORKDIR /app
COPY deploy/cloud_run/requirements.txt ./requirements.txt
COPY deploy/cloud_run/wheels ./wheels
RUN pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt \
    && rm -rf wheels

# --- Source + entrypoint ---------------------------------------------------
COPY backend ./backend
COPY deploy/cloud_run/entrypoint.sh ./entrypoint.sh

# Cloud Run runs the container as the Dockerfile USER. UID 1000 owns the
# Ollama model dir and the app dir so Ollama does not try to re-pull into its
# own $HOME and the baked model stays visible.
RUN useradd -m -u 1000 user \
    && chown -R 1000:1000 /opt/ollama-models /app \
    && chmod +x /app/entrypoint.sh

USER 1000
ENV HOME=/home/user \
    OLLAMA_MODELS=/opt/ollama-models \
    HOST=0.0.0.0

# Cloud Run injects $PORT (default 8080); do not hardcode a fixed port here.
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]