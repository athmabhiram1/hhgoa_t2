"""Multi-strategy chunking.

Six parallel namespaces are indexed (see CONTEXT.md). Each chunk carries its
full metadata (lang, query_type, query_id, position, is_selected) so Neo4j can
filter on any of them and the eval harness can score against ground truth.

Chunk ids are content-stable (sha1 over namespace+text+query+position) so
re-indexing is idempotent.

IMPORTANT — id-compatibility note (do not change without a CONTEXT.md
callout, per AGENTS.md's golden rule): passage_natural, passage_en, and
query_anchored are the three namespaces in the live AuraDB deploy subset.
Their id formula is untouched here on purpose, so ids stay byte-identical to
what's already indexed. passage_fixed, passage_recursive, and semantic now
fold a sub-chunk index into the id (see Chunk.build_id) to close a real
collision risk — those three namespaces are NOT part of the deploy subset,
so this only affects the local full-pool eval index, not anything live.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..config import QUERY_TYPES
from .dataset import PassageRecord, QueryRecord

# Namespaces mirror the six strategies: five chunking strategies + document-level
# coarse index, plus an English-only cross-lingual namespace (see CONTEXT.md).
NAMESPACES = ["passage_natural", "passage_fixed", "passage_recursive", "query_anchored", "semantic", "passage_en"]

# Matches sentence-ending punctuation across the languages in MSMARCO-XI:
# ASCII . ! ? , Devanagari-family danda ।/॥ (Hindi, Sanskrit, Marathi, ...),
# Urdu/Persian ؟. Used both for the semantic strategy and as a recursion
# level below — a plain ". " separator (the old approach) never matches most
# of these scripts, silently degrading "recursive" splitting to whitespace
# splitting for 13 of the 14 languages.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।।॥؟])[\s\n]+")


@dataclass
class Chunk:
    chunk_id: str
    namespace: str
    text: str
    lang: str
    query_id: int
    query_type: str
    position: int | None = None     # position within the query's passage list
    is_selected: int | None = None  # ground-truth relevance
    passage_pos: int | None = None  # index of source passage within query
    doc_key: str = ""               # coarse document key (query id based)
    token_estimate: int = 0         # approx. token count (see _tokens_approx)

    @staticmethod
    def build_id(namespace: str, text: str, query_id: int, position: int | None, sub_index: int | None = None) -> str:
        """Content-stable chunk id.

        `sub_index` disambiguates multiple sub-chunks carved out of the same
        source passage (fixed/recursive/semantic strategies) so two
        sub-chunks can never collide even if their trimmed text happens to
        be identical. Left as None (default) for one-chunk-per-passage
        namespaces to keep those ids byte-identical to what's already
        indexed in AuraDB — see the module docstring before changing this.
        """
        parts = [namespace, str(query_id), str(position)]
        if sub_index is not None:
            parts.append(str(sub_index))
        parts.append(text)
        raw = "|".join(parts).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    @staticmethod
    def from_passage(p: PassageRecord, namespace: str, text: str | None = None, sub_index: int | None = None) -> "Chunk":
        body = text or p.text
        return Chunk(
            chunk_id=Chunk.build_id(namespace, body, p.query_id, p.position, sub_index),
            namespace=namespace,
            text=body,
            lang=p.lang,
            query_id=p.query_id,
            query_type="UNKNOWN",
            position=p.position,
            is_selected=p.is_selected,
            passage_pos=p.position,
            doc_key=str(p.query_id),
            token_estimate=_tokens_approx(body),
        )


def _tokens_approx(text: str, chars_per_token: float | None = None) -> int:
    """Rough token estimate, script-aware.

    Indic scripts tokenize denser than Latin script (~2.5 chars/token per
    deployment-plan.md's own cost table, vs ~4 chars/token for English) — a
    flat 4-chars/token divisor underestimates token counts, and therefore
    underestimates embedding quota/cost, for 13 of this project's 14
    languages. This is a display/planning estimate only; it does not change
    chunk-boundary sizing (see _split_windows).
    """
    if not text:
        return 0
    if chars_per_token is None:
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        chars_per_token = 2.5 if non_ascii > len(text) * 0.3 else 4.0
    return max(1, int(len(text) / chars_per_token))


def _split_windows(text: str, window_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    # NOTE: flat 4-chars/token here on purpose — this sizes actual chunk
    # boundaries, not just a cost estimate, and changing it would shift
    # every passage_fixed chunk's content. Left as-is to avoid destabilizing
    # tuned gate thresholds; _tokens_approx above is the more accurate
    # estimator if this ever needs revisiting (flag it, don't silently swap it).
    window_chars = max(1, window_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)
    if len(text) <= window_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + window_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + window_chars // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Strategy 1 — natural passage units (the semantic base unit)
# ---------------------------------------------------------------------------
def natural_chunks(passages: list[PassageRecord]) -> list[Chunk]:
    return [Chunk.from_passage(p, "passage_natural") for p in passages if p.text]


# ---------------------------------------------------------------------------
# Strategy 2 — fixed-size windows with overlap (for long passages)
# ---------------------------------------------------------------------------
def fixed_chunks(passages: list[PassageRecord], window_tokens: int = 256, overlap_tokens: int = 48) -> list[Chunk]:
    out: list[Chunk] = []
    for p in passages:
        for i, part in enumerate(_split_windows(p.text, window_tokens, overlap_tokens)):
            chunk = Chunk.from_passage(p, "passage_fixed", text=part, sub_index=i)
            chunk.position = p.position
            chunk.passage_pos = p.position + i
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Strategy 3 — recursive character splitting (structure-aware)
# ---------------------------------------------------------------------------
# Progressively finer separator levels: paragraph -> line -> sentence
# (language-aware: reuses _SENTENCE_SPLIT, so Devanagari/Bengali/Urdu
# sentence punctuation is respected, not just ASCII '.') -> word. Anything
# still oversized after the word level falls through to a hard character
# window (_split_windows) as an unconditional last resort — no chunk is ever
# silently left over max_chars, unlike the previous single-level splitter.
_RECURSIVE_LEVELS = [
    lambda t: t.split("\n\n"),
    lambda t: t.split("\n"),
    lambda t: _SENTENCE_SPLIT.split(t),
    lambda t: t.split(" "),
]


def _split_recursive(text: str, max_chars: int, min_chars: int, level: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if level >= len(_RECURSIVE_LEVELS):
        # Word-level splitting still left something oversized (e.g. a script
        # with no spaces, or one pathologically long token). Hard character
        # window guarantees termination with every piece under max_chars.
        return _split_windows(text, max_chars // 4, 0)

    pieces = [p.strip() for p in _RECURSIVE_LEVELS[level](text) if p and p.strip()]
    if len(pieces) <= 1:
        # This separator didn't actually break the text apart; try a finer one.
        return _split_recursive(text, max_chars, min_chars, level + 1)

    out: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf} {piece}".strip() if buf else piece
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf and len(buf) >= min_chars:
            out.append(buf)
            buf = ""
        elif buf:
            # buf below min_chars: fold it onto the piece instead of emitting
            # a too-small fragment; the next recursion level will resplit if
            # the combined piece is still too large.
            piece = f"{buf} {piece}".strip()
            buf = ""
        if len(piece) > max_chars:
            out.extend(_split_recursive(piece, max_chars, min_chars, level + 1))
        else:
            buf = piece
    if buf:
        if out and len(buf) < min_chars:
            out[-1] = f"{out[-1]} {buf}".strip()
        else:
            out.append(buf)
    return out


def recursive_chunks(passages: list[PassageRecord], max_chars: int = 900, min_chars: int = 250) -> list[Chunk]:
    out: list[Chunk] = []
    for p in passages:
        parts = _split_recursive(p.text, max_chars, min_chars)
        for i, part in enumerate(parts):
            chunk = Chunk.from_passage(p, "passage_recursive", text=part, sub_index=i)
            chunk.position = p.position
            chunk.passage_pos = p.position + i
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Strategy 4 — query-anchored pseudo-documents (question-aware semantics)
# ---------------------------------------------------------------------------
# NOTE — LEAKY (see CONTEXT.md eval section): the query text is prepended
# into the passage body, so retrieval against this namespace trivially
# matches the query it was built from. It is a real, useful retrieval
# strategy but must never be compared against in_index_mrr / used as the
# primary reported metric.
def query_anchored_chunks(q: QueryRecord, max_chars: int = 1200) -> list[Chunk]:
    out: list[Chunk] = []
    prefix = q.query.strip()
    for i, p in enumerate(q.passages):
        if not p.text:
            continue
        body = f"{prefix} {p.text}".strip()
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0]
        chunk = Chunk(
            chunk_id=Chunk.build_id("query_anchored", body, q.query_id, p.position),
            namespace="query_anchored",
            text=body,
            lang=p.lang,
            query_id=q.query_id,
            query_type=q.query_type,
            position=p.position,
            is_selected=p.is_selected,
            passage_pos=i,
            doc_key=str(q.query_id),
            token_estimate=_tokens_approx(body),
        )
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Strategy 5 — semantic (sentence-level, topic-coherence merging)
# ---------------------------------------------------------------------------
def _jaccard_bigram(a: str, b: str) -> float:
    def ngrams(s: str, n: int = 2) -> set[str]:
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i : i + n] for i in range(len(s) - n + 1) if s[i] != " "}

    na, nb = ngrams(a), ngrams(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def semantic_chunks(passages: list[PassageRecord], merge_threshold: float = 0.12, max_chars: int = 1100) -> list[Chunk]:
    """Greedy sentence merge using lexical (bigram) topic coherence.

    Deterministic and model-free: a sentence joins the running chunk while
    its overlap with the *whole current group* (not just the immediately
    preceding sentence — a pairwise-chain comparison can drift topic across
    a long run of transitively-but-not-mutually similar sentences) exceeds
    the threshold; otherwise a new chunk starts. Variable-size, topic-aligned
    chunks.

    merge_threshold is an untuned constant carried over from the original
    implementation — flagging rather than silently re-guessing it: per this
    project's own "measured, not aspirational" standard it should be swept
    against the golden eval set before being trusted, especially across
    languages with different average sentence/word lengths.
    """
    out: list[Chunk] = []
    for p in passages:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(p.text) if s.strip()]
        if not sentences:
            out.append(Chunk.from_passage(p, "semantic"))
            continue
        groups: list[list[str]] = []
        for sent in sentences:
            if groups:
                group_text = " ".join(groups[-1])
                fits = sum(len(s) for s in groups[-1]) + len(sent) <= max_chars
                coherent = _jaccard_bigram(group_text, sent) >= merge_threshold
                if coherent and fits:
                    groups[-1].append(sent)
                    continue
            groups.append([sent])
        for i, group in enumerate(groups):
            body = " ".join(group)
            chunk = Chunk.from_passage(p, "semantic", text=body, sub_index=i)
            chunk.position = p.position
            chunk.passage_pos = p.position + i
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Strategy 6 — English passages (cross-lingual retrieval target)
# ---------------------------------------------------------------------------
def english_chunks(passages: list[PassageRecord]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in passages:
        if not p.english_text or not p.english_text.strip():
            continue
        chunk = Chunk.from_passage(p, "passage_en", text=p.english_text)
        chunk.position = p.position
        chunk.passage_pos = p.position
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def chunk_queries(queries: list[QueryRecord], *, strategies: list[str] | None = None) -> list[Chunk]:
    """Produce all chunks across the enabled strategies."""
    enabled = set(strategies or NAMESPACES)
    chunks: list[Chunk] = []
    if "query_anchored" in enabled:
        for q in queries:
            chunks.extend(query_anchored_chunks(q))
    passage_records = [p for q in queries for p in q.passages if p.text]
    if "passage_natural" in enabled:
        chunks.extend(natural_chunks(passage_records))
    if "passage_fixed" in enabled:
        chunks.extend(fixed_chunks(passage_records))
    if "passage_recursive" in enabled:
        chunks.extend(recursive_chunks(passage_records))
    if "semantic" in enabled:
        chunks.extend(semantic_chunks(passage_records))
    if "passage_en" in enabled:
        chunks.extend(english_chunks(passage_records))
    # Tag query_type onto passage-derived chunks (Chunk.from_passage leaves it UNKNOWN).
    qtype_by_id = {q.query_id: q.query_type for q in queries}
    for c in chunks:
        if c.query_type == "UNKNOWN":
            c.query_type = qtype_by_id.get(c.query_id, "UNKNOWN")
    return chunks