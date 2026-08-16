import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.models import RetrievedPassage, RetrievalResult
from backend.rag.fast_path import extractive_answer
from backend.core.providers import _parse_json_object


def _retrieval() -> RetrievalResult:
    cands = [
        RetrievedPassage(
            id="p1",
            text="The capital of India is New Delhi. New Delhi is the seat of government. Rain is wet.",
            score=0.92,
            source="vector",
        )
    ]
    return RetrievalResult(query="what is the capital of India", candidates=cands, grounding_score=0.92)


def test_extractive_selects_matching_sentence():
    r = _retrieval()
    answer = extractive_answer("what is the capital of India?", r)
    assert answer is not None
    assert "capital" in answer.text
    assert answer.mode == "extractive"
    assert answer.citations[0].passage_id == "p1"


def test_extractive_empty_candidates():
    r = RetrievalResult(query="q", candidates=[], grounding_score=0.0)
    assert extractive_answer("anything", r) is None


def test_extractive_sets_grounding():
    r = _retrieval()
    answer = extractive_answer("capital", r)
    assert abs(answer.grounding_score - 0.92) < 1e-6


def test_parse_json_object_fences():
    raw = '```json\n{"answer": "New Delhi", "confidence": 0.9}\n```'
    data = _parse_json_object(raw)
    assert data["answer"] == "New Delhi"


def test_parse_json_object_trims_prose():
    raw = 'Here you go: {"answer": "42", "confidence": 1.0} hope that helps'
    data = _parse_json_object(raw)
    assert data["answer"] == "42"


def test_parse_json_object_raises_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        _parse_json_object("not json at all")