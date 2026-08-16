import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.harness.eval_mrr import gold_texts, held_out_eligible, recurring_passages
from backend.ingestion.dataset import PassageRecord, QueryRecord


def _passage(text: str, selected: bool = False) -> PassageRecord:
    return PassageRecord(position=0, text=text, english_text="", is_selected=int(selected), lang="hi", query_id=1)


def _query(qid: int, passages: list[str], selected: int = 0) -> QueryRecord:
    return QueryRecord(
        query_id=qid,
        query=f"query {qid}",
        answer=f"answer {qid}",
        query_type="ENTITY",
        lang="hi",
        passages=[_passage(t, i == selected) for i, t in enumerate(passages)],
    )


def test_held_out_eligible_is_disjoint_and_fully_covered():
    # One passage recurs across two indexed queries -> it is in the pool.
    recurring_text = "Rashtrapati Bhavan is the official residence of the President of India."
    sample = [
        _query(101, [recurring_text, "A unique passage about cricket."], selected=0),
        _query(102, [recurring_text, "Another unique passage about rivers."], selected=1),
        _query(103, ["A passage only seen here."], selected=0),
    ]
    recurring = recurring_passages(sample)
    assert recurring_text and _norm(recurring_text) in recurring  # core premise holds

    # Holdout A's gold IS the recurring text -> eligible and gold-in-pool.
    holdout_a = _query(501, [recurring_text, "decoy"], selected=0)
    # Holdout B's gold is unique -> NOT eligible.
    holdout_b = _query(502, ["A gold passage that appears nowhere in the index."], selected=0)

    eligible = held_out_eligible([holdout_a, holdout_b], recurring)

    assert [q.query_id for q in eligible] == [501]

    # Disjointness: no eligible holdout query_id is an indexed query_id.
    index_ids = {q.query_id for q in sample}
    assert all(q.query_id not in index_ids for q in eligible)

    # Coverage = 1.0 by construction: every eligible query's gold is in the pool.
    for q in eligible:
        assert any(_norm(g) in recurring for g in gold_texts(q))


def test_held_out_eligible_empty_when_no_gold_recurs():
    sample = [_query(201, ["Alpha passage."], selected=0)]
    holdout = [_query(601, ["Beta passage never shared."], selected=0)]
    recurring = recurring_passages(sample)
    assert recurring == {}
    assert held_out_eligible(holdout, recurring) == []


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())