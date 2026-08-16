---
title: VakRAG
emoji: 🇮🇳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# VakRAG — Indic multilingual voice + text RAG demo

Hugging Face Space (free CPU tier) running the VakRAG FastAPI backend with
Ollama (`qwen3-embedding:0.6b`) in the same container.

## Architecture

- **Retrieval store:** Neo4j AuraDB Free (set `NEO4J_URI`, `NEO4J_USERNAME`,
  `NEO4J_PASSWORD` as Space secrets). Index built from the `vakrag_v1`
  namespace — do not change `NEO4J_INDEX_NAMESPACE` or embeddings.
- **Embeddings:** `qwen3-embedding:0.6b` served by the in-container Ollama on
  `localhost:11434` (immutable decision — see CONTEXT.md).
- **Generation:** Gemini API primary (`GEMINI_API_KEY`, `GEMINI_MODEL`),
  Ollama fallback (`OLLAMA_FALLBACK_MODEL=llama3.2:3b`).
- **STT:** Sarvam API (`SARVAM_API_KEY`), `STT_PROVIDER=sarvam`.

## Space secrets (Settings → Variables and secrets)

| Key                  | Value                                          |
| -------------------- | ---------------------------------------------- |
| `GEMINI_API_KEY`     | Google AI Studio key                           |
| `SARVAM_API_KEY`     | Sarvam key                                     |
| `NEO4J_URI`          | `neo4j+s://<id>.databases.neo4j.io`            |
| `NEO4J_USERNAME`     | `neo4j`                                        |
| `NEO4J_PASSWORD`     | AuraDB password                                |
| `PRIMARY_LLM_PROVIDER` | `gemini`                                     |

The model is baked into the image at build time (`RUN ollama pull ...`), so a
Space sleep/restart does not re-download it.

## Endpoints

- `POST /v1/ask` — SSE stream of stage events -> final answer
- `POST /v1/ask/text` — JSON convenience
- `GET /v1/health` — Neo4j + embedding readiness
- `GET /v1/telemetry` — live latency snapshot
- `GET /v1/graph` — knowledge-graph snapshot