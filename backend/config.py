"""Typed configuration loaded from `.env` via pydantic-settings.

Locked decisions from CONTEXT.md map to settings here. Any change that alters
the embedding model, Neo4j namespace, or provider order must be called out in
CONTEXT.md as well.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 14 Indic languages in MSMARCO-XI (HF subset names)
ALL_INDIC_LANGS = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur",
]

# query_type values observed in MSMARCO (used for metadata-aware retrieval)
QUERY_TYPES = ["DESCRIPTION", "ENTITY", "NUMERIC", "PERSON", "LOCATION", "MISC"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    otel_exporter: str = "console"  # console | langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # --- Neo4j ------------------------------------------------------------
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "change-me"
    neo4j_database: str = "neo4j"
    neo4j_index_namespace: str = "vakrag_v1"

    # --- Embeddings -------------------------------------------------------
    embed_model: str = "BAAI/bge-m3"
    embed_backend: str = "ollama"  # ollama | gemini | vertex | fastembed | sentence-transformers
    embed_device: str = "cpu"  # cpu | cuda (used only by torch backends)
    embed_onnx_dir: Path = Path("./models/bge-m3-onnx")
    embed_dim: int = 1024
    embed_batch_size: int = 128
    embed_max_chars: int = 8192
    # Ollama embedding backend (local-first; see CONTEXT.md Phase-2 decision)
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_embed_dim: int = 1024
    # Gemini embedding API backend (embed_backend="gemini")
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dim: int = 1024          # output_dimensionality (128-3072; 1024 = no Neo4j schema change)
    gemini_embed_batch_size: int = 100    # batchEmbedContents caps at 100 requests/call
    gemini_embed_max_chars: int = 4000    # 2048-token input cap ~= 4000 chars (Indic ~0.4 tok/char)
    # Vertex AI embedding backend (embed_backend="vertex") — same
    # gemini-embedding-001 model served from aiplatform.googleapis.com and
    # BILLED TO THE GCP PROJECT (covered by the Free Trial $300 credit), unlike
    # the AI Studio Gemini API. Reuses gemini_embed_dim/batch/max_chars above.
    # Vertex exposes no batchEmbedContents for this model: each text is one
    # :predict call, parallelized with a thread pool (see embeddings.py).
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_embed_model: str = "gemini-embedding-001"
    vertex_credentials: str = ""          # path to service-account JSON OR inline JSON string
    vertex_embed_concurrency: int = 16    # concurrent :predict calls (1 text/call)
    # Throughput tuning: num_batch 512->1024 = +50-80% embed throughput on GPU
    # (measured), and keep_alive prevents per-batch model reload.
    ollama_embed_num_batch: int = 1024
    ollama_embed_keep_alive: str = "5m"
    # Request-path embed concurrency bound. Ollama's /api/embed is serial at the
    # server; limiting client-side in-flight embeds keeps the pipeline from
    # piling up requests and lets the SSE stream surface a "queued" stage event
    # instead of a dead-looking wait. (Directive Phase 6: bound 2-3.)
    embed_concurrency: int = 2

    # --- STT ---------------------------------------------------------------
    sarvam_api_key: str = ""
    stt_provider: str = "sarvam"  # sarvam | whisper
    whisper_model_size: str = "small"  # small | base | medium
    whisper_device: str = "cuda"  # cuda | cpu
    whisper_compute_type: str = "float16"
    whisper_max_audio_seconds: int = 30

    # --- Generation LLM ---------------------------------------------------
    primary_llm_provider: str = "gemini"  # gemini | ollama | groq | openai
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_fallback_model: str = "llama3.2:3b"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    generation_max_tokens: int = 512
    generation_temperature: float = 0.2

    # --- Retrieval --------------------------------------------------------
    retrieval_vector_k: int = 40
    retrieval_bm25_k: int = 40
    retrieval_fusion_topk: int = 12
    retrieval_rrf_k: int = 60
    # RRF arm weights (locked Aug 2026, verified B2): vector arms keep weight
    # 1.0, BM25 arms are down-weighted to 0.5. Applied per-list in rrf_fuse.
    retrieval_vector_weight: float = 1.0
    retrieval_bm25_weight: float = 0.5
    retrieval_rerank: str = "none"  # none | local
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    # Controlled A/B experimental mode (Phase: reranker A/B). Default OFF.
    # When enabled, `python -m backend.harness.live_reranker_ab` runs the
    # B2+GPU-rerank path against the default B2 path on a fixed 20-query smoke
    # set. /v1/ask is untouched and never reads this flag.
    reranker_ab_enabled: bool = False

    # --- Fast path (latency-first Tier 1, Locked Aug 2026) ----------------
    # A fully-local retrieval+extractive path: the corpus is re-embedded ONCE
    # with a local model (bge-m3, NOT the Vertex/gemini-embedding-001 space
    # that Neo4j stores) into an in-memory numpy ANN index built from a
    # read-only Neo4j text export. Query-time: local embed -> brute-force
    # cosine (~31k x 1024 is <15ms) -> extractive span answer. This is the
    # 200ms-compliant output; the Vertex+Neo4j+RRF+grounding+LLM pipeline
    # streams the full answer as progressive enhancement. Default OFF — the
    # index must be built first (`python -m backend.harness.build_fastpath`);
    # set FAST_PATH_ENABLED=true to serve it.
    fast_path_enabled: bool = False
    fast_path_model: str = "BAAI/bge-m3"       # MUST match the index build model
    fast_path_device: str = "cuda"             # RTX 5050; CPU works, slower
    fast_path_index_dir: Path = Path("./data/fastpath")
    fast_path_topk: int = 12
    fast_path_batch: int = 32                 # corpus embed batch size at build (8GB VRAM)
    fast_path_namespaces: list[str] = ["passage_natural", "passage_en"]
    fast_path_grounding_floor: float = 0.30    # local-cosine floor for extractive mode

    # --- Guardrails -------------------------------------------------------
    guard_safety: bool = True
    guard_offtopic: bool = True
    guard_grounding: bool = True
    guard_grounding_threshold: float = 0.78
    guard_faithfulness: bool = True
    guard_llm_judge: bool = False

    # --- Dataset / indexing ----------------------------------------------
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    hf_token: str = ""  # HF token (HF_TOKEN) for dataset downloads; empty = public
    # Demo sampling source. "train" = ~3.7GB × 13 files (no Telugu); "validation"
    # = ~450MB × 14 files (incl. Telugu) and fits the demo disk budget. Either is
    # downloaded one file at a time and deleted after sampling.
    dataset_split: str = "validation"
    dataset_langs: str = "all"  # "all" or comma list e.g. "hi,ta,te"
    dataset_max_passages: int = 180_000
    dataset_max_passages_per_query: int = 10
    dataset_seed: int = 42
    dataset_holdout_per_lang: int = 35  # eval queries reserved per lang (never indexed)
    index_batch_size: int = 512
    # Phase 6C (Aug 2026): AuraDB Free intermittently stalls on sustained vector
    # writes — a single write exceeds the proxy's ~30s read window under Lucene
    # index-maintenance load (isolated diagnostics reproduced 95-97s stalls; the
    # driver retry always recovers). A small inter-batch pause reduces the burst
    # pile-up. 0 disables pacing.
    index_pace_s: float = 0.5
    # Progressive-index quality gate (locked Aug 2026): never index the whole
    # corpus in one shot. Index a small pilot (index_gate_batch queries per
    # language), run a golden-set Recall@10 eval over the just-indexed pilot
    # pool, and only continue the full build if Recall@10 >= index_gate_threshold.
    # On failure the CLI stops and reports what to improve (chunking strategy,
    # embed backend, threshold) instead of burning hours of embed time on a bad
    # index. CLI: --progressive --gate-batch N --gate-threshold T.
    index_gate_batch: int = 8
    index_gate_threshold: float = 0.40  # pilot Recall@10 floor; continue only above

    # --- Benchmark --------------------------------------------------------
    benchmark_n_queries: int = 150
    benchmark_outdir: Path = Path("./benchmarks")
    benchmark_concurrency: int = 8
    benchmark_cache_size: int = 512  # LRU query→result cache entries (query path)
    benchmark_warm: bool = True      # embed warm-up + warm/cold split measurement
    loadtest_concurrency: int = 32   # concurrent load test (P100 under saturation)
    # Live /v1/benchmark is DISABLED by default (Locked Aug 2026): it burns
    # Vertex/Gemini/Sarvam credits and load on a public app. Enable explicitly
    # for local dev only via BENCHMARK_ENABLED=true. The CLI harness
    # (python -m backend.harness.benchmark) is unaffected.
    benchmark_enabled: bool = False

    # --- Security / deploy hardening (Locked Aug 2026) --------------------
    # CORS is OFF by default (frontend is same-origin behind nginx). When a
    # separate dev origin needs access, set CORS_ORIGINS to a comma-separated
    # allowlist; the wildcard "*" is never accepted.
    cors_origins: str = ""

    @property
    def langs(self) -> list[str]:
        if self.dataset_langs.strip().lower() == "all":
            return list(ALL_INDIC_LANGS)
        return [x.strip() for x in self.dataset_langs.split(",") if x.strip()]

    @property
    def grounding_threshold(self) -> float:
        return self.guard_grounding_threshold


@lru_cache
def get_settings() -> Settings:
    return Settings()