# VakRAG demo backend for Hugging Face Spaces (Docker SDK, free CPU tier).
# Runs FastAPI (port 7860, exposed by Spaces) + Ollama (localhost:11434)
# serving qwen3-embedding:0.6b in the same container.
#
# Locked decisions this file must not silently change (see CONTEXT.md):
#   - Embeddings stay on qwen3-embedding:0.6b via Ollama. Do NOT switch the
#     embedding backend to Gemini/an API — it would break every grounding and
#     eval number in CONTEXT.md and require a reindex.
#   - Generation LLM: Gemini API primary, Ollama fallback (same chain as local).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Install Ollama (root) -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# --- Bake the embedding model into an image layer (survives Space restarts).
# OLLAMA_MODELS is pinned to a fixed path used at BOTH build and runtime, so the
# model baked here is visible to the UID 1000 runtime user with no re-pull.
ENV OLLAMA_MODELS=/opt/ollama-models
RUN mkdir -p /opt/ollama-models \
    && (ollama serve >/dev/null 2>&1 &) \
    && for i in $(seq 1 30); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done \
    && ollama pull qwen3-embedding:0.6b

# --- Python deps (request path only; heavier dev deps stay out of the image) -
# Wheels are pre-fetched on the build host (pip concurrency stalls inside the
# Docker network) and installed offline for a deterministic build.
WORKDIR /app
COPY requirements.txt .
COPY wheels ./wheels
RUN pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt \
    && rm -rf wheels

# --- Source + entrypoint ---------------------------------------------------
COPY backend ./backend
COPY entrypoint.sh .

# HF Spaces runs containers as UID 1000. The Ollama model dir and app dir must
# be writable/readable by that user, or Ollama would try to re-pull into its
# own $HOME and the baked model would be invisible.
RUN useradd -m -u 1000 user \
    && chown -R 1000:1000 /opt/ollama-models /app \
    && chmod +x /app/entrypoint.sh

USER 1000
ENV HOME=/home/user \
    OLLAMA_MODELS=/opt/ollama-models \
    HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]