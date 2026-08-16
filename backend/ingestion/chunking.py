"""Multi-strategy chunking.

Six parallel namespaces are indexed (see CONTEXT.md). Each chunk carries its
full metadata (lang, query_type, query_id, position, is_selected) so Neo4j can
filter on any of them and the eval harness can score against ground truth.

Chunk ids are content-stable (sha1 over namespace+text+query+position) so
re-indexing is idempotent.
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

    @staticmethod
    def build_id(namespace: str, text: str, query_id: int, position: int | None) -> str:
        raw = f"{namespace}|{query_id}|{position}|{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    @staticmethod
    def from_passage(p: PassageRecord, namespace: str, text: str | None = None) -> "Chunk":
        body = text or p.text
        return Chunk(
            chunk_id=Chunk.build_id(namespace, body, p.query_id, p.position),
            namespace=namespace,
            text=body,
            lang=p.lang,
            query_id=p.query_id,
            query_type="UNKNOWN",
            position=p.position,
            is_selected=p.is_selected,
            passage_pos=p.position,
            doc_key=str(p.query_id),
        )


def _tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def _split_windows(text: str, window_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
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
            chunk = Chunk.from_passage(p, "passage_fixed", text=part)
            chunk.position = p.position
            chunk.passage_pos = p.position + i
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Strategy 3 — recursive character splitting (structure-aware)
# ---------------------------------------------------------------------------
def recursive_chunks(passages: list[PassageRecord], max_chars: int = 900, min_chars: int = 250) -> list[Chunk]:
    out: list[Chunk] = []
    for p in passages:
        if len(p.text) <= max_chars:
            out.append(Chunk.from_passage(p, "passage_recursive"))
            continue
        for sep in ["\n\n", "\n", ". ", " "]:
            if sep not in p.text:
                continue
            parts = _recursive_split(p.text, sep, max_chars, min_chars)
            for i, part in enumerate(parts):
                chunk = Chunk.from_passage(p, "passage_recursive", text=part)
                chunk.position = p.position
                chunk.passage_pos = p.position + i
                out.append(chunk)
            break
        else:
            out.append(Chunk.from_passage(p, "passage_recursive"))
    return out


def _recursive_split(text: str, sep: str, max_chars: int, min_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces = text.split(sep)
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = (buf + sep + piece).strip() if buf else piece
        if len(candidate) <= max_chars or not buf:
            buf = candidate
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
                buf = piece
            else:
                buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4 — query-anchored pseudo-documents (question-aware semantics)
# ---------------------------------------------------------------------------
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
    its overlap with the chunk exceeds the threshold; otherwise a new chunk
    starts. Variable-size, topic-aligned chunks.
    """
    out: list[Chunk] = []
    for p in passages:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(p.text) if s.strip()]
        if not sentences:
            out.append(Chunk.from_passage(p, "semantic"))
            continue
        groups: list[list[str]] = []
        for sent in sentences:
            if groups and _jaccard_bigram(groups[-1][-1], sent) >= merge_threshold and sum(len(s) for s in groups[-1]) + len(sent) <= max_chars:
                groups[-1].append(sent)
            else:
                groups.append([sent])
        for i, group in enumerate(groups):
            body = " ".join(group)
            chunk = Chunk.from_passage(p, "semantic", text=body)
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