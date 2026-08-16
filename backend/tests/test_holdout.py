import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.ingestion.dataset as dataset
from backend.config import get_settings
from backend.ingestion.dataset import StratifiedSampler, load_sample, save_sample


def _row(query_id: int, query_type: str, lang: str = "hi", n_passages: int = 3) -> dict:
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "Answer": f"answer {query_id}",
        "query_type": query_type,
        "passages": {
            "Translated_passages": [f"translated passage {query_id}-{i}" for i in range(n_passages)],
            "English_passages": [f"english passage {query_id}-{i}" for i in range(n_passages)],
            "is_selected": [1 if i == 0 else 0 for i in range(n_passages)],
        },
    }


def _fake_rows(per_type: int):
    """Deterministic fake stream: each query_type gets per_type queries (ids are globally unique)."""
    types = [t for t in dataset.QUERY_TYPES if t != "UNKNOWN"]
    qid = 1
    for t in types:
        for _ in range(per_type):
            yield _row(qid, t)
            qid += 1


def _cfg():
    cfg = get_settings()
    cfg.dataset_langs = "hi"
    cfg.dataset_max_passages = 1000  # small index budget so holdout gets filled
    cfg.dataset_max_passages_per_query = 3
    return cfg


def test_sample_lang_disjoint_index_vs_holdout(monkeypatch):
    monkeypatch.setattr(dataset, "_iter_rows", lambda lang, split="train", streaming=True: _fake_rows(per_type=100))

    cfg = _cfg()
    sampler = StratifiedSampler(cfg)
    index, holdout = sampler.sample_lang("hi", holdout_per_lang=12)

    index_ids = {q.query_id for q in index}
    holdout_ids = {q.query_id for q in holdout}
    # Core guarantee: the holdout is never part of the indexed sample.
    assert index_ids.isdisjoint(holdout_ids)
    assert index_ids
    assert len(holdout) == 12
    assert all(len(q.passages) <= cfg.dataset_max_passages_per_query for q in index)


def test_run_split_and_roundtrip_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "_iter_rows", lambda lang, split="train", streaming=True: _fake_rows(per_type=100))

    cfg = _cfg()
    sampler = StratifiedSampler(cfg)
    index, holdout = sampler.run(holdout_per_lang=12)

    sample_dir = tmp_path / "sample"
    holdout_dir = tmp_path / "holdout"
    save_sample(index, sample_dir)
    save_sample(holdout, holdout_dir)

    loaded_index = load_sample(sample_dir, ["hi"])
    loaded_holdout = load_sample(holdout_dir, ["hi"])
    assert len(loaded_index) == len(index)
    assert len(loaded_holdout) == len(holdout)

    # Disjointness must survive persistence (index vs holdout JSONL).
    index_ids = {q.query_id for q in loaded_index}
    holdout_ids = {q.query_id for q in loaded_holdout}
    assert index_ids.isdisjoint(holdout_ids)

    # Round-trip fidelity.
    first = loaded_holdout[0]
    assert first.query == f"query {first.query_id}"
    assert len(first.passages) == 3


def test_holdout_balanced_across_query_types(monkeypatch):
    monkeypatch.setattr(dataset, "_iter_rows", lambda lang, split="train", streaming=True: _fake_rows(per_type=100))

    cfg = _cfg()
    sampler = StratifiedSampler(cfg)
    _, holdout = sampler.sample_lang("hi", holdout_per_lang=12)

    types = {q.query_type for q in holdout}
    assert len(types) >= 2  # spread across query types, not one type hoarded