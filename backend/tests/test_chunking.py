import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.ingestion.chunking import (
    NAMESPACES,
    chunk_queries,
    fixed_chunks,
    natural_chunks,
    query_anchored_chunks,
    recursive_chunks,
    semantic_chunks,
)
from backend.ingestion.dataset import PassageRecord, QueryRecord


def _rec(text: str = "Some passage text about India and its rivers. " * 20) -> PassageRecord:
    return PassageRecord(position=0, text=text, english_text=text, is_selected=1, lang="hi", query_id=42)


def _query(passages: list[PassageRecord] | None = None) -> QueryRecord:
    return QueryRecord(
        query_id=42,
        query="गंगा किस नदी की सहायक है?",
        answer="यमुना",
        query_type="ENTITY",
        lang="hi",
        passages=passages or [_rec()],
    )


def test_namespaces_are_six_and_stable():
    assert NAMESPACES == ["passage_natural", "passage_fixed", "passage_recursive", "query_anchored", "semantic", "passage_en"]


def test_natural_chunks_one_per_passage():
    chunks = natural_chunks([_rec(), _rec("Second passage text here. " * 10)])
    assert len(chunks) == 2
    assert chunks[0].namespace == "passage_natural"


def test_fixed_chunks_split_long_text():
    chunks = fixed_chunks([_rec()], window_tokens=16, overlap_tokens=4)
    assert len(chunks) > 1
    assert all(c.namespace == "passage_fixed" for c in chunks)
    assert all(c.passage_pos is not None for c in chunks)


def test_recursive_chunks_short_stays_single():
    chunks = recursive_chunks([_rec("Short passage. ")])
    assert len(chunks) == 1
    assert chunks[0].text == "Short passage."


def test_query_anchored_prefixes_query():
    chunks = query_anchored_chunks(_query())
    assert chunks[0].text.startswith("गंगा किस नदी की सहायक है?")
    assert chunks[0].query_type == "ENTITY"


def test_semantic_chunks_merge_related_sentences():
    p = PassageRecord(position=0, text="India is a country in South Asia. India has many rivers and mountains. Hockey is a sport played with sticks.", english_text="", is_selected=1, lang="en", query_id=1)
    chunks = semantic_chunks([p])
    assert len(chunks) == 1  # group-vs-whole-chunk coherence merges all three
    assert "India" in chunks[0].text and "Hockey" in chunks[0].text


def test_chunk_ids_are_content_stable():
    c1 = natural_chunks([_rec("Same text here. ")])[0]
    c2 = natural_chunks([_rec("Same text here. ")])[0]
    assert c1.chunk_id == c2.chunk_id


def test_chunk_queries_dispatcher_tags_query_type():
    chunks = chunk_queries([_query()], strategies=["passage_natural", "query_anchored"])
    assert all(c.query_type == "ENTITY" for c in chunks)


def test_chunk_queries_respects_strategies():
    qs = chunk_queries([_query()], strategies=["query_anchored"])
    assert {c.namespace for c in qs} == {"query_anchored"}


def test_config_langs_all():
    cfg = get_settings()
    assert len(cfg.langs) == 14