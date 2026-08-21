
"""Neo4jStore transient-connection retry contract.

Regression for the Aug 2026 finding: AuraDB Free drops Bolt connections under
concurrent load (SessionExpired). _run/_run_write must retry transient errors
with a fresh coroutine (fresh session/connection) and re-raise non-transient
errors immediately.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j.exceptions import SessionExpired, ServiceUnavailable, TransientError

from backend.config import Settings
from backend.retrieval.neo4j_store import Neo4jStore, _is_transient_neo4j


def run(coro):
    return asyncio.run(coro)


def _store():
    s = Neo4jStore.__new__(Neo4jStore)
    s.cfg = Settings(neo4j_uri="bolt://localhost:7687")
    return s


def test_transient_classification():
    assert _is_transient_neo4j(SessionExpired("drop")) is True
    assert _is_transient_neo4j(ServiceUnavailable("pool")) is True
    assert _is_transient_neo4j(TransientError("tx")) is True
    assert _is_transient_neo4j(ValueError("bad")) is False


def test_with_retry_recovers_transient():
    s = _store()
    calls = {"n": 0}

    def make_coro():
        async def _one():
            calls["n"] += 1
            if calls["n"] < 3:
                raise SessionExpired("AuraDB dropped the connection")
            return [{"ok": True}]

        return _one()

    out = run(s._with_retry(make_coro, attempts=4, base_delay=0.01))
    assert out == [{"ok": True}]
    assert calls["n"] == 3


def test_with_retry_raises_after_exhaustion():
    s = _store()
    calls = {"n": 0}

    def make_coro():
        async def _one():
            calls["n"] += 1
            raise SessionExpired("persistent drop")

        return _one()

    with pytest_raises(SessionExpired):
        run(s._with_retry(make_coro, attempts=3, base_delay=0.01))
    assert calls["n"] == 3


def test_with_retry_raises_non_transient_immediately():
    s = _store()
    calls = {"n": 0}

    def make_coro():
        async def _one():
            calls["n"] += 1
            raise ValueError("not transient")

        return _one()

    with pytest_raises(ValueError):
        run(s._with_retry(make_coro, attempts=3, base_delay=0.01))
    assert calls["n"] == 1


def pytest_raises(exc_type):
    class CM:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {exc_type.__name__}")
            return issubclass(et, exc_type)

    return CM()
