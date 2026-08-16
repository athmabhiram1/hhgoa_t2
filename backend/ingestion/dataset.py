"""MSMARCO-XI (IndicRAGSuite) sampling.

Streams the HuggingFace dataset per-language (no 55GB download), applies a
stratified sample balanced across language and `query_type`, and emits typed
records ready for chunking + indexing. The sample is persisted to JSONL so
re-indexing never re-streams the source.

Only the sampled slice is indexed in the demo; the pipeline is streaming and
resumable, so it scales to the full corpus without code changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import ALL_INDIC_LANGS, QUERY_TYPES, Settings

logger = logging.getLogger(__name__)

LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}


@dataclass
class PassageRecord:
    position: int
    text: str              # translated (native script)
    english_text: str      # original English passage
    is_selected: int       # ground-truth relevance (1 = answer passage)
    lang: str
    query_id: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryRecord:
    query_id: int
    query: str
    answer: str
    query_type: str
    lang: str
    passages: list[PassageRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "query_type": self.query_type,
            "lang": self.lang,
            "passages": [p.to_dict() for p in self.passages],
        }


# Map short codes → dataset target_lang values (MSMARCO-XI Flores-style tags)
_LANG_TAG = {
    "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "ne": "nep_Deva",
    "or": "ori_Orya", "pa": "pan_Guru", "sa": "san_Deva", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab",
}

# Map short codes → per-language parquet filename prefix (repo layout).
# NB: the TRAIN split has NO `teltrain.parquet` (Telugu is train+val under a
# different layout); the VALIDATION split has all 14 incl. `telval.parquet`.
# Languages missing a file are skipped by the sampler with a warning.
_FILE_TAG = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}

_DATASET_REPO = "ai4bharat/MSMARCO-XI"
_ROW_BUFFER = 10_000


def _iter_rows(lang: str, split: str = "validation", streaming: bool = True):
    """Lazy iterator over rows for one language.

    MSMARCO-XI stores each language as its own parquet file (e.g.
    `train/hintrain.parquet`) under a single `default` HF config, and each file
    is a SINGLE parquet row group (~3.5GB). So there is no partial read — the
    whole file must be fetched. We download it ONCE to the HF disk cache
    (`hf_hub_download`, resumable and token-aware), stream it locally in small
    batches (bounded RAM), then delete the local copy to keep disk usage at ~one
    file (~3.5GB) instead of ~45GB for the full train split.
    """
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ..config import get_settings

    file_prefix = _FILE_TAG.get(lang)
    if file_prefix is None:
        logger.warning("No parquet file for language %s — skipping", lang)
        return
    filename = f"{split}/{file_prefix}{'train' if split == 'train' else 'val'}.parquet"
    cfg = get_settings()
    local_path = hf_hub_download(
        _DATASET_REPO,
        filename,
        repo_type="dataset",
        token=cfg.hf_token or None,
    )
    try:
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=_ROW_BUFFER, columns=["target_lang", "query", "Answer", "query_id", "query_type", "passages", "Eng_Query"]):
            for row in batch.to_pylist():
                yield row
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def _extract_query(row: dict, lang: str) -> QueryRecord | None:
    qid = row.get("query_id")
    if qid is None:
        return None
    passages = row.get("passages") or {}
    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    selected = passages.get("is_selected") or []
    if not translated:
        return None
    records = [
        PassageRecord(
            position=i,
            text=str(t).strip(),
            english_text=str(english[i]).strip() if i < len(english) else "",
            is_selected=int(selected[i]) if i < len(selected) else 0,
            lang=lang,
            query_id=int(qid),
        )
        for i, t in enumerate(translated)
        if t and str(t).strip()
    ]
    if not records:
        return None
    return QueryRecord(
        query_id=int(qid),
        query=str(row.get("query", "")).strip(),
        answer=str(row.get("Answer", "")).strip(),
        query_type=str(row.get("query_type", "UNKNOWN")).upper(),
        lang=lang,
        passages=records,
    )


class StratifiedSampler:
    """Balances queries across language and query_type while streaming."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.queries: list[QueryRecord] = []

    def _budget_per_lang(self) -> int:
        max_p = self.cfg.dataset_max_passages
        avg_ppq = 7
        per_lang = max(50, (max_p // max(1, len(self.cfg.langs)) // avg_ppq))
        return per_lang

    def _budgets(self, target: int, min_per_type: int = 10) -> dict[str, int]:
        types = [t for t in QUERY_TYPES if t != "UNKNOWN"]
        per_type = max(min_per_type, target // len(types))
        return {t: per_type for t in types}

    def sample_lang(self, lang: str, holdout_per_lang: int = 0) -> tuple[list[QueryRecord], list[QueryRecord]]:
        cfg = self.cfg
        index_budgets = self._budgets(self._budget_per_lang())
        holdout_budgets = self._budgets(holdout_per_lang, min_per_type=1) if holdout_per_lang > 0 else {}
        taken: list[QueryRecord] = []
        holdout: list[QueryRecord] = []
        seen_ids: set[int] = set()

        for row in _iter_rows(lang, split=self.cfg.dataset_split):
            q = _extract_query(row, lang)
            if q is None or q.query_id in seen_ids:
                continue
            qtype = q.query_type if q.query_type in index_budgets or q.query_type in holdout_budgets else "MISC"
            if index_budgets.get(qtype, 0) > 0:
                index_budgets[qtype] -= 1
            elif holdout_budgets.get(qtype, 0) > 0:
                holdout_budgets[qtype] -= 1
                q.passages = q.passages[: cfg.dataset_max_passages_per_query]
                holdout.append(q)
                seen_ids.add(q.query_id)
                continue
            else:
                continue
            seen_ids.add(q.query_id)
            # Cap passages per query to keep the index lean.
            if len(q.passages) > cfg.dataset_max_passages_per_query:
                q.passages = q.passages[: cfg.dataset_max_passages_per_query]
            taken.append(q)
            if sum(index_budgets.values()) <= 0 and sum(holdout_budgets.values()) <= 0:
                break
        logger.info("Sampled %d index + %d holdout from %s", len(taken), len(holdout), lang)
        return taken, holdout

    def run(self, holdout_per_lang: int = 0) -> tuple[list[QueryRecord], list[QueryRecord]]:
        index_queries: list[QueryRecord] = []
        holdout_queries: list[QueryRecord] = []
        for lang in self.cfg.langs:
            try:
                idx, ho = self.sample_lang(lang, holdout_per_lang)
                index_queries.extend(idx)
                holdout_queries.extend(ho)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed sampling %s: %s", lang, exc)
        logger.info(
            "Total index queries: %d, holdout queries: %d, passages: %d",
            len(index_queries),
            len(holdout_queries),
            sum(len(q.passages) for q in index_queries),
        )
        return index_queries, holdout_queries


def save_sample(queries: list[QueryRecord], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    by_lang: dict[str, list[QueryRecord]] = {}
    for q in queries:
        by_lang.setdefault(q.lang, []).append(q)
    for lang, qs in by_lang.items():
        with (outdir / f"{lang}.jsonl").open("w", encoding="utf-8") as fh:
            for q in qs:
                fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved sample to %s (%d languages)", outdir, len(by_lang))


def load_sample(indir: Path, langs: list[str] | None = None) -> list[QueryRecord]:
    if not indir.exists():
        return []
    queries: list[QueryRecord] = []
    for f in sorted(indir.glob("*.jsonl")):
        lang = f.stem
        if langs and lang not in langs:
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                passages = [
                    PassageRecord(position=p["position"], text=p["text"], english_text=p["english_text"],
                                  is_selected=p["is_selected"], lang=p["lang"], query_id=p["query_id"])
                    for p in d.get("passages", [])
                ]
                queries.append(QueryRecord(query_id=d["query_id"], query=d["query"], answer=d["answer"],
                                           query_type=d["query_type"], lang=d["lang"], passages=passages))
    return queries