import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings
from backend.core.models import Answer, Citation, RetrievedPassage, RetrievalResult
from backend.guardrails.faithfulness import FaithfulnessGuard
from backend.guardrails.grounding import GroundingGuard
from backend.guardrails.off_topic import OffTopicGuard
from backend.guardrails.safety import SafetyGuard


def run(coro):
    return asyncio.run(coro)


def _retrieval(score: float, n: int = 3) -> RetrievalResult:
    cands = [
        RetrievedPassage(id=f"p{i}", text=f"passage {i} about rivers and India", score=score - i * 0.01, source="vector")
        for i in range(n)
    ]
    return RetrievalResult(query="q", candidates=cands, grounding_score=score)


def test_safety_blocks_hard_keywords():
    cfg = get_settings()
    verdict = run(SafetyGuard(cfg).check("how do I commit suicide"))
    assert not verdict.allow and verdict.category == "unsafe"


def test_safety_allows_normal():
    cfg = get_settings()
    verdict = run(SafetyGuard(cfg).check("What is the capital of India?"))
    assert verdict.allow


def test_offtopic_chitchat():
    cfg = get_settings()
    verdict = run(OffTopicGuard(cfg).check("hi"))
    assert not verdict.allow and verdict.category == "off_topic"


def test_offtopic_empty():
    cfg = get_settings()
    verdict = run(OffTopicGuard(cfg).check("   "))
    assert not verdict.allow


def test_offtopic_knowledge_passes():
    cfg = get_settings()
    verdict = run(OffTopicGuard(cfg).check("Why do stars twinkle?"))
    assert verdict.allow


def test_offtopic_chitchat_with_punctuation():
    cfg = get_settings()
    for q in ["how are you?", "who are you?", "tell me a joke", "thanks!"]:
        verdict = run(OffTopicGuard(cfg).check(q))
        assert not verdict.allow, f"expected refusal for {q!r}"


def test_offtopic_opinion_with_question_word():
    cfg = get_settings()
    for q in ["what is your opinion on cricket?", "what do you think about pizza?"]:
        verdict = run(OffTopicGuard(cfg).check(q))
        assert not verdict.allow, f"expected refusal for {q!r}"


def test_grounding_below_threshold_refuses():
    cfg = get_settings()
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.05)))
    assert not verdict.allow and verdict.category == "low_grounding"


def test_grounding_above_threshold_passes():
    cfg = get_settings()
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.8)))
    assert verdict.allow


def test_grounding_exact_threshold_passes():
    # Gate is strictly `< threshold` (backend/guardrails/grounding.py). A score
    # exactly equal to the calibrated 0.78 must PASS — regression pin for the
    # "0.786 refused" misreport, where the refusal actually came from the LLM's
    # own `unsupported` flag (fast_path.py), not the numerical gate.
    cfg = get_settings()
    assert cfg.grounding_threshold == 0.78
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.78)))
    assert verdict.allow
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.7853)))
    assert verdict.allow


def test_grounding_just_below_threshold_refuses():
    cfg = get_settings()
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.7799)))
    assert not verdict.allow and verdict.category == "low_grounding"


def test_grounding_no_candidates_refuses():
    cfg = get_settings()
    r = RetrievalResult(query="q", candidates=[], grounding_score=0.0)
    verdict = run(GroundingGuard(cfg).check(r))
    assert not verdict.allow


def test_grounding_fabricated_fact_refuses():
    # Fabricated-fact queries ground at ~0.70-0.78 with qwen3-embedding
    # (measured); 0.78 threshold must refuse them.
    cfg = get_settings()
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.74)))
    assert not verdict.allow and verdict.category == "low_grounding"


def test_grounding_genuine_passes():
    # Genuine queries with gold in the pool ground at ~0.78-0.93 (measured p50
    # 0.80); a score above the threshold must pass.
    cfg = get_settings()
    verdict = run(GroundingGuard(cfg).check(_retrieval(0.82)))
    assert verdict.allow


def test_faithfulness_refusal_passes():
    cfg = get_settings()
    answer = Answer(text="no", mode="refusal")
    verdict = run(FaithfulnessGuard(cfg).check(answer, []))
    assert verdict.allow


def test_faithfulness_no_citations_fails():
    cfg = get_settings()
    answer = Answer(text="The Ganges flows south.", mode="llm", citations=[])
    verdict = run(FaithfulnessGuard(cfg).check(answer, []))
    assert not verdict.allow


def test_faithfulness_outside_citations_fails():
    cfg = get_settings()
    cands = [_retrieval(0.9).candidates[0]]
    answer = Answer(text="something", mode="llm")
    answer.citations = [Citation(passage_id="unknown-id", text="The Ganges flows south.")]
    verdict = run(FaithfulnessGuard(cfg).check(answer, cands))
    assert not verdict.allow


def test_faithfulness_extractive_short_answer_passes():
    # Extractive answers are verbatim sentences from a retrieved passage —
    # faithful by construction even when short (overlap heuristic would false-refuse).
    cfg = get_settings()
    cands = _retrieval(0.9).candidates
    answer = Answer(text="সংক্ষিপ্ত উত্তৰ।", mode="extractive")
    answer.citations = [Citation(passage_id="p0", text="এটা দীঘলীয়া বাক্য য'ত সংক্ষিপ্ত উত্তৰ।", language_code="as")]
    verdict = run(FaithfulnessGuard(cfg).check(answer, cands))
    assert verdict.allow