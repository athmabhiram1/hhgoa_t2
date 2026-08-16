#!/bin/sh
# VakRAG HF Space entrypoint.
# Starts Ollama in the background, waits until it is ready (poll, not sleep),
# then hands over to uvicorn on the Spaces-exposed port (7860).

set -e

echo "[entrypoint] starting ollama serve"
ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] waiting for ollama at localhost:11434"
ready=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" != "1" ]; then
  echo "[entrypoint] ERROR: ollama did not become ready in 60s" >&2
  exit 1
fi

echo "[entrypoint] ollama ready; verifying baked model"
if ! ollama list | grep -q "qwen3-embedding:0.6b"; then
  echo "[entrypoint] WARNING: qwen3-embedding:0.6b not in OLLAMA_MODELS; first embed will pull it"
fi

exec uvicorn backend.main:app --host 0.0.0.0 --port 7860