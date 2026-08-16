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
    embed_backend: str = "ollama"  # ollama | fastembed | sentence-transformers
    embed_device: str = "cpu"  # cpu | cuda (used only by torch backends)
    embed_onnx_dir: Path = Path("./models/bge-m3-onnx")
    embed_dim: int = 1024
    embed_batch_size: int = 128
    embed_max_chars: int = 8192
    # Ollama embedding backend (local-first; see CONTEXT.md Phase-2 decision)
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_embed_dim: int = 1024
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
    retrieval_rerank: str = "none"  # none | local
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"

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

    # --- Benchmark --------------------------------------------------------
    benchmark_n_queries: int = 150
    benchmark_outdir: Path = Path("./benchmarks")
    benchmark_concurrency: int = 8
    benchmark_cache_size: int = 512  # LRU query→result cache entries (query path)
    benchmark_warm: bool = True      # embed warm-up + warm/cold split measurement
    loadtest_concurrency: int = 32   # concurrent load test (P100 under saturation)

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