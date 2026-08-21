"""Neo4j as the single unified store: HNSW vector index, Lucene BM25 fulltext
index, and a lightweight metadata/co-occurrence graph (Language / QueryType /
Query / Chunk). Per-namespace labels isolate the five chunking strategies so
each HNSW index is clean (no namespace post-filter scan).

All search paths are async and go through the Bolt driver with a persistent
pool (connection_pool_size), which keeps query-time latency low.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

from neo4j import AsyncGraphDatabase, TrustCustomCAs, TrustSystemCAs

from ..config import Settings
from ..core.models import RetrievedPassage
from ..ingestion.chunking import NAMESPACES
from .embeddings import fresh_backend_of

logger = logging.getLogger(__name__)

# Transient Bolt errors that are safe to retry with a fresh session/connection.
# SessionExpired is the concrete failure AuraDB Free surfaces when it drops the
# connection under load (observed Aug 2026); ServiceUnavailable covers pool /
# routing hiccups; TransientError covers server-side transient failures. All
# are resolved by re-opening a session, which is what the retry loop does.
_neo4j_transient = None


def _is_transient_neo4j(exc: Exception) -> bool:
    global _neo4j_transient
    if _neo4j_transient is None:
        from neo4j.exceptions import SessionExpired, ServiceUnavailable, TransientError

        _neo4j_transient = (SessionExpired, ServiceUnavailable, TransientError)
    return isinstance(exc, _neo4j_transient)


def _trust() -> TrustCustomCAs | TrustSystemCAs:
    # AuraDB serves a publicly-issued cert over neo4j+s://, but some Python
    # builds (e.g. the Windows Store interpreter) ship no usable system CA
    # bundle, so TLS verify fails with CERTIFICATE_VERIFY_FAILED. certifi's
    # bundle is a dependency (via httpx) and works everywhere; fall back to
    # the platform trust store when it's not importable.
    try:
        import certifi

        return TrustCustomCAs(certifi.where())
    except Exception:  # noqa: BLE001
        return TrustSystemCAs()


def _connect_config(cfg: Settings) -> dict[str, Any]:
    """Driver kwargs for the configured URI.

    neo4j+s:// forces the system CA store in this driver version (custom
    trusted_certificates are rejected for the +s/+ssc schemes), which breaks
    on Python builds without a CA bundle. Rewrite to the plain scheme and
    supply `encrypted=True` + `trusted_certificates` explicitly — real cert
    verification preserved via certifi's bundle. Plain local URIs (bolt/neo4j)
    keep their current no-encryption behavior.
    """
    uri = cfg.neo4j_uri
    if uri.startswith("neo4j+s://"):
        return {
            "uri": "neo4j://" + uri[len("neo4j+s://") :],
            "encrypted": True,
            "trusted_certificates": _trust(),
        }
    if uri.startswith("bolt+s://"):
        return {
            "uri": "bolt://" + uri[len("bolt+s://") :],
            "encrypted": True,
            "trusted_certificates": _trust(),
        }
    return {"uri": uri}

# namespace -> node label
LABEL_BY_NS = {
    "passage_natural": "ChunkNatural",
    "passage_fixed": "ChunkFixed",
    "passage_recursive": "ChunkRecursive",
    "query_anchored": "ChunkAnchored",
    "semantic": "ChunkSemantic",
    "passage_en": "ChunkEnglish",
}


def vector_index_name(ns: str) -> str:
    return f"{ns}_vector"


def fulltext_index_name(ns: str) -> str:
    return f"{ns}_fulltext"


class Neo4jStore:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            conn = _connect_config(self.cfg)
            self._driver = AsyncGraphDatabase.driver(
                conn["uri"],
                auth=(self.cfg.neo4j_username, self.cfg.neo4j_password),
                max_connection_pool_size=8,
                # Phase 6 (Aug 2026): AuraDB Free silently reaps idle Bolt
                # connections; with liveness_check_timeout=None (driver default)
                # the pool served them back up, causing intermittent 60-95s
                # "read timed out / defunct connection" stalls on vector writes.
                # liveness_check_timeout pings idle connections before reuse and
                # max_connection_lifetime bounds connection age so the pool
                # refreshes ahead of AuraDB's idle reaping. connection_acquisition_timeout
                # keeps pool-exhaustion waits bounded.
                liveness_check_timeout=10.0,
                max_connection_lifetime=600,
                connection_acquisition_timeout=30.0,
                **{k: v for k, v in conn.items() if k != "uri"},
            )
        return self._driver

    async def close(self) -> None:
        if self._driver is None:
            return
        # Shutdown-only hardening (Phase 7D, AuraDB Free): on Windows/proactor
        # the driver's close() can race an already-torn-down SSL socket and raise
        # (e.g. AttributeError on 'NoneType'.send / 'Event loop is closed').
        # The pool/session will be gone regardless, so swallow ONLY exceptions
        # raised during this shutdown teardown. Query/write, transaction and
        # retry behavior are untouched.
        try:
            await self._driver.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j driver close() raised during shutdown: %s", exc)
        self._driver = None

    async def verify_connectivity(self) -> bool:
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Neo4j connectivity failed: %s", exc)
            return False

    async def _run(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        return await self._with_retry(lambda: self._run_once(query, params))

    async def _run_once(self, query: str, params: dict[str, Any] | None) -> list[dict]:
        async with self.driver.session(database=self.cfg.neo4j_database) as session:
            result = await session.run(query, parameters=params or {})
            return [record.data() async for record in result]

    async def _run_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        await self._with_retry(lambda: self._run_write_once(query, params))

    async def _run_write_once(self, query: str, params: dict[str, Any] | None) -> None:
        async with self.driver.session(database=self.cfg.neo4j_database) as session:
            await session.execute_write(lambda tx: tx.run(query, parameters=params or {}))

    async def _with_retry(self, make_coro: Callable[[], Awaitable[Any]], *, attempts: int = 4, base_delay: float = 0.4) -> Any:
        """Run a single-driver-call factory with bounded retry on transient Bolt
        errors. AuraDB Free drops connections under concurrent load
        (SessionExpired, observed Aug 2026); each retry calls `make_coro()` for
        a FRESH coroutine that opens a fresh session on a fresh connection,
        which is the reconnect. Non-transient errors raise immediately."""
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await make_coro()
            except Exception as exc:  # noqa: BLE001
                if not _is_transient_neo4j(exc):
                    raise
                if attempt >= attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) * (1.0 + random.uniform(-0.2, 0.2))
                logger.warning(
                    "Neo4j transient error, retry %d/%d in %.2fs: %s",
                    attempt, attempts, delay, exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover — unreachable

    # ------------------------------------------------------------------ schema
    async def ensure_schema(self) -> None:
        for ns in NAMESPACES:
            label = LABEL_BY_NS[ns]
            await self._run(
                f"CREATE CONSTRAINT `{ns}_id` IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.chunk_id IS UNIQUE"
            )
            await self._run(
                f"""CREATE VECTOR INDEX `{vector_index_name(ns)}` IF NOT EXISTS
                    FOR (n:`{label}`) ON (n.embedding)
                    OPTIONS {{indexConfig: {{`vector.dimensions`: {self.cfg.embed_dim},
                                             `vector.similarity_function`: 'cosine'}}}}"""
            )
            await self._run(
                f"""CREATE FULLTEXT INDEX `{fulltext_index_name(ns)}` IF NOT EXISTS
                    FOR (n:`{label}`) ON EACH [n.text]"""
            )
        await self._run("CREATE CONSTRAINT `query_id` IF NOT EXISTS FOR (n:Query) REQUIRE n.query_id IS UNIQUE")
        await self._run("CREATE CONSTRAINT `lang_code` IF NOT EXISTS FOR (n:Language) REQUIRE n.code IS UNIQUE")
        await self._run("CREATE CONSTRAINT `qtype_name` IF NOT EXISTS FOR (n:QueryType) REQUIRE n.name IS UNIQUE")
        logger.info("Neo4j schema ensured for %d namespaces", len(NAMESPACES))

    # ------------------------------------------------------------------ ingest
    async def upsert_chunks(self, rows: list[dict], namespace: str, *, run_id: str | None = None) -> None:
        label = LABEL_BY_NS[namespace]
        # PHASE 7 GUARD (contamination postmortem, Aug 2026): a row may claim
        # embed_backend='vertex' ONLY if its vector was freshly produced by the
        # Vertex backend in THIS process (registered by EmbeddingService.embed_batch).
        # The Phase-5/6 bench scripts read the stored `embedding` property from
        # Neo4j and re-tagged those stale Qwen3-era vectors as Vertex without ever
        # calling the Vertex API — 300 chunks contaminated. This guard makes that
        # write fail instead. Reusing a stored vector for ANY backend is the same
        # class of bug, so the check is applied whenever the row tags a backend
        # that is not 'vertex' too — every vector must carry the backend that
        # actually produced it in this process.
        for row in rows:
            tagged = row.get("embed_backend")
            producer = fresh_backend_of(row.get("embedding"))
            if producer != tagged:
                raise RuntimeError(
                    "REFUSED embed write: row tags embed_backend="
                    f"{tagged!r} but its vector was produced by {producer!r} "
                    "(or read from a stale stored embedding). Only vectors freshly "
                    "generated by the embedding service in this process may be "
                    "written — re-tagging a stored embedding is forbidden (Phase 7)."
                )
        # Clean-recovery provenance (Phase 7B): when a run_id is supplied every
        # row MUST carry that exact reembed_run tag, so a chunk is only ever
        # marked complete by the recovery run that actually wrote its vector.
        if run_id:
            if not run_id.strip():
                raise RuntimeError("run_id must be non-empty")
            for row in rows:
                if row.get("reembed_run") != run_id:
                    raise RuntimeError(
                        "REFUSED embed write: run_id given but a row does not carry "
                        f"reembed_run={run_id!r} (got {row.get('reembed_run')!r})"
                    )
        query = self._upsert_query(namespace, label, run_id=run_id)
        await self._run_write(query, {"rows": rows})

    @staticmethod
    def _upsert_query(namespace: str, label: str, *, run_id: str | None = None) -> str:
        """UNWIND MERGE upsert query for `label`.

        passage_en nodes are language-agnostic BY DESIGN: the English passage
        text is identical across all 14 languages for a shared query_id, so
        their chunk_ids dedup across languages and any `lang` property would be
        last-writer-wins garbage ('ur'). Never write a lang property here — and
        REMOVE any stale one so re-indexing heals old nodes. See CONTEXT.md.
        A non-None `run_id` appends the recovery provenance term
        (n.reembed_run = row.reembed_run) so completion is attributable to THIS
        recovery run only.
        """
        terms = [
            "n.namespace = row.namespace",
            "n.text = row.text",
        ]
        if namespace != "passage_en":
            terms.append("n.lang = row.lang")
        terms += [
            "n.query_id = row.query_id",
            "n.query_type = row.query_type",
            "n.position = row.position",
            "n.is_selected = row.is_selected",
            "n.passage_pos = row.passage_pos",
            "n.doc_key = row.doc_key",
            "n.embed_backend = row.embed_backend",
        ]
        if run_id:
            terms.append("n.reembed_run = row.reembed_run")
        # The SET list is comma-joined with no trailing comma, so `REMOVE
        # n.lang` (passage_en) and the following `WITH` are separate clauses —
        # a dangling comma can never precede either.
        set_block = ",\n".join(f"            {t}" for t in terms)
        lang_cleanup = "REMOVE n.lang" if namespace == "passage_en" else ""
        return f"""
        UNWIND $rows AS row
        MERGE (n:`{label}` {{chunk_id: row.chunk_id}})
        SET {set_block}
            {lang_cleanup}
        WITH n, row
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        RETURN count(n) AS c
        """

    async def upsert_query_graph(self, queries: list[dict], namespace: str, chunk_ids: list[str]) -> None:
        """Query/Language/QueryType nodes + HAS_PASSAGE edges to the given chunks."""
        q_rows = [{"query_id": q["query_id"], "query": q["query"], "answer": q["answer"], "query_type": q["query_type"], "lang": q["lang"]} for q in queries]
        await self._run_write(
            """
            UNWIND $rows AS row
            MERGE (q:Query {query_id: row.query_id})
            SET q.query = row.query, q.answer = row.answer, q.query_type = row.query_type, q.lang = row.lang
            MERGE (l:Language {code: row.lang})
            MERGE (t:QueryType {name: row.query_type})
            MERGE (l)-[:HAS_QUERY]->(q)
            MERGE (q)-[:OF_TYPE]->(t)
            """,
            {"rows": q_rows},
        )
        label = LABEL_BY_NS[namespace]
        await self._run_write(
            f"""
            UNWIND $rows AS row
            MATCH (q:Query {{query_id: row.query_id}})
            MATCH (n:`{label}` {{chunk_id: row.chunk_id}})
            MERGE (q)-[:HAS_PASSAGE]->(n)
            """,
            {"rows": [{"query_id": row["query_id"], "chunk_id": row["chunk_id"]} for row in chunk_ids]},
        )

    async def count_chunks(self, namespace: str, *, embed_backend: str | None = None) -> int:
        label = LABEL_BY_NS[namespace]
        if embed_backend:
            rows = await self._run(
                f"MATCH (n:`{label}`) WHERE n.embed_backend = $backend RETURN count(n) AS c",
                {"backend": embed_backend},
            )
        else:
            rows = await self._run(f"MATCH (n:`{label}`) RETURN count(n) AS c")
        return int(rows[0]["c"]) if rows else 0

    async def existing_chunk_ids(self, namespace: str, chunk_ids: list[str], *, embed_backend: str | None = None) -> set[str]:
        """Return the subset of chunk_ids already stored in this namespace.

        One batched IN-list query, so resumable indexing can skip embedding +
        MERGE for chunks that survived a previous partial run instead of
        re-embedding them from zero. When `embed_backend` is given, only chunks
        whose vectors were produced by that backend count as "existing" — after
        a backend switch (e.g. Ollama → Gemini) the old vectors must be
        overwritten, so re-indexing re-embeds them.
        """
        if not chunk_ids:
            return set()
        label = LABEL_BY_NS[namespace]
        backend_clause = " AND n.embed_backend = $backend" if embed_backend else ""
        rows = await self._run(
            f"MATCH (n:`{label}`) WHERE n.chunk_id IN $ids{backend_clause} RETURN n.chunk_id AS id",
            {"ids": chunk_ids, "backend": embed_backend},
        )
        return {r["id"] for r in rows}

    async def existing_run_ids(self, namespace: str, chunk_ids: list[str], run_id: str) -> set[str]:
        """Subset of `chunk_ids` completed by the recovery run `run_id`.

        The ONLY completion authority for clean recovery: a chunk is complete
        when its `reembed_run` property equals THIS run id. Never derived from
        vertex counts, embed_backend, embedding presence, NULL, or ollama status.
        """
        if not chunk_ids:
            return set()
        label = LABEL_BY_NS[namespace]
        rows = await self._run(
            f"MATCH (n:`{label}`) WHERE n.chunk_id IN $ids AND n.reembed_run = $run "
            f"RETURN n.chunk_id AS id",
            {"ids": chunk_ids, "run": run_id},
        )
        return {r["id"] for r in rows}

    async def fetch_chunk_texts(self, namespace: str, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunk TEXT (and metadata) for the given ids — NEVER the stored
        `embedding` property.

        This is the harness's sanctioned input source for re-embedding: a
        benchmark can only read the text it must re-embed, so a "fresh" Vertex
        vector can only come from calling the Vertex EmbeddingService on that
        text — never from copying a stored vector back out (the Phase-5/6
        contamination mechanism). Deliberately omits `embedding` from the
        projection so stale vectors cannot even be selected by mistake.
        """
        if not chunk_ids:
            return []
        label = LABEL_BY_NS[namespace]
        rows = await self._run(
            f"MATCH (n:`{label}`) WHERE n.chunk_id IN $ids "
            "RETURN n.chunk_id AS chunk_id, n.namespace AS namespace, n.text AS text, "
            "n.lang AS lang, n.query_id AS query_id, n.query_type AS query_type, "
            "n.position AS position, n.is_selected AS is_selected, "
            "n.passage_pos AS passage_pos, n.doc_key AS doc_key",
            {"ids": chunk_ids},
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "namespace": r["namespace"],
                "text": r["text"],
                "lang": r["lang"],
                "query_id": r["query_id"],
                "query_type": r["query_type"],
                "position": r["position"],
                "is_selected": r["is_selected"],
                "passage_pos": r["passage_pos"],
                "doc_key": r["doc_key"],
            }
            for r in rows
        ]

    async def fetch_namespace_texts(self, namespace: str) -> list[dict]:
        """Read-only export of chunk TEXT + metadata for a whole namespace.

        Same contract as fetch_chunk_texts — NEVER the stored `embedding`
        property. Used by the fast-path index build
        (backend/retrieval/local_index.py) to re-embed the corpus locally in a
        self-consistent vector space (bge-m3), instead of reusing the stored
        gemini-embedding-001 vectors.
        """
        label = LABEL_BY_NS[namespace]
        rows = await self._run(
            f"MATCH (n:`{label}`) "
            "RETURN n.chunk_id AS chunk_id, n.namespace AS namespace, n.text AS text, "
            "n.lang AS lang, n.query_id AS query_id, n.query_type AS query_type, "
            "n.is_selected AS is_selected",
            {},
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "namespace": r["namespace"],
                "text": r["text"],
                "lang": r["lang"],
                "query_id": r["query_id"],
                "query_type": r["query_type"],
                "is_selected": r["is_selected"],
            }
            for r in rows
        ]

    async def count_queries(self) -> int:
        rows = await self._run("MATCH (n:Query) RETURN count(n) AS c")
        return int(rows[0]["c"]) if rows else 0

    async def count_has_passage(self, namespace: str) -> int:
        label = LABEL_BY_NS[namespace]
        rows = await self._run(
            f"MATCH (q:Query)-[:HAS_PASSAGE]->(n:`{label}`) RETURN count(*) AS c"
        )
        return int(rows[0]["c"]) if rows else 0

    # ------------------------------------------------------------------ search
    async def vector_search(self, namespace: str, vector: list[float], k: int, *, lang: str | None = None, query_type: str | None = None) -> list[RetrievedPassage]:
        if namespace == "passage_en" and lang:
            # passage_en nodes carry NO lang property (English is language-agnostic
            # by design — any Indic query language retrieves into it). A lang filter
            # here is structurally impossible: it would silently match nothing.
            raise ValueError("passage_en is language-agnostic — lang filter is not supported")
        idx = vector_index_name(namespace)
        label = LABEL_BY_NS[namespace]
        filter_clause = ""
        if lang:
            filter_clause += " AND n.lang = $lang"
        if query_type:
            filter_clause += " AND n.query_type = $query_type"
        query = (
            f"CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node AS n, score"
            f" WHERE 1=1{filter_clause}"
            f" RETURN n, score ORDER BY score DESC LIMIT $k"
        )
        rows = await self._run(query, {"idx": idx, "k": k, "vec": vector, "lang": lang, "query_type": query_type})
        return [self._node_to_passage(r["n"], r["score"], "vector", namespace) for r in rows]

    async def bm25_search(self, namespace: str, text: str, k: int, *, lang: str | None = None, query_type: str | None = None) -> list[RetrievedPassage]:
        if namespace == "passage_en" and lang:
            # Same contract as vector_search: passage_en is language-agnostic.
            raise ValueError("passage_en is language-agnostic — lang filter is not supported")
        idx = fulltext_index_name(namespace)
        label = LABEL_BY_NS[namespace]
        filter_clause = ""
        if lang:
            filter_clause += " AND n.lang = $lang"
        if query_type:
            filter_clause += " AND n.query_type = $query_type"
        query = (
            f"CALL db.index.fulltext.queryNodes($idx, $q, {{limit: $k}}) YIELD node AS n, score"
            f" WHERE 1=1{filter_clause}"
            f" RETURN n, score ORDER BY score DESC LIMIT $k"
        )
        rows = await self._run(query, {"idx": idx, "q": _lucene_query(text), "k": k, "lang": lang, "query_type": query_type})
        return [self._node_to_passage(r["n"], r["score"], "bm25", namespace) for r in rows]

    def _node_to_passage(self, node: dict, score: float, source: str, namespace: str) -> RetrievedPassage:
        return RetrievedPassage(
            id=str(node.get("chunk_id", "")),
            text=str(node.get("text", "")),
            language_code=str(node.get("lang", "")),
            score=float(score),
            raw_cosine=float(score) if source == "vector" else None,
            source=source,
            query_id=str(node.get("query_id") or ""),
            query_type=node.get("query_type"),
            position=node.get("position"),
            is_selected=node.get("is_selected"),
            namespace=namespace,
        )

    # ------------------------------------------------------------------ graph viz
    async def graph_snapshot(self, limit: int = 300) -> dict[str, Any]:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        rows = await self._run(
            f"MATCH (n) WHERE n:Query OR n:Language OR n:QueryType RETURN n LIMIT $limit",
            {"limit": limit},
        )
        for r in rows:
            n = r["n"]
            nid = f"{n['name'] if 'name' in n else n['code'] if 'code' in n else n['query_id']}"
            labels = list(n.labels)
            kind = "query" if "Query" in labels else ("lang" if "Language" in labels else "qtype")
            nodes[str(n.element_id)] = {"id": str(n.element_id), "label": str(nid), "type": kind}
        rel_rows = await self._run(
            f"MATCH (a)-[r:HAS_QUERY|OF_TYPE]->(b) RETURN a, b, type(r) AS t LIMIT $limit",
            {"limit": limit},
        )
        for r in rel_rows:
            edges.append({"source": str(r["a"].element_id), "target": str(r["b"].element_id), "label": r["t"]})
        return {"nodes": list(nodes.values()), "edges": edges}


def _lucene_query(text: str) -> str:
    """Build a BM25 query from plain text (token OR-join, escaped)."""
    import re

    tokens = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    tokens = [t for t in tokens if len(t) > 1][:32]
    escaped = [t.replace('"', '\\"') for t in tokens]
    return " OR ".join(f'"{t}"~1' if t else t for t in escaped) if escaped else '""'