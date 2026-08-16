"""call_resilient contract tests.

Regression for the Phase-6 finding: `call_resilient` passed the *coroutine
from `retry_with_backoff`* into `CircuitBreaker.call`, which then did
`await fn()` on it -> "'coroutine' object is not callable", breaking every
provider call that used a breaker (all of them). Also pins that HTTP 429 /
5xx status codes are treated as retryable even when raised as
httpx.HTTPStatusError (status lives on `.response`, not the exception).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from backend.core.retry import CircuitBreaker, call_resilient


def run(coro):
    return asyncio.run(coro)


def test_call_resilient_with_breaker_succeeds():
    async def fn():
        return "ok"

    result = run(call_resilient(fn, breaker=CircuitBreaker(), attempts=2))
    assert result == "ok"


def test_call_resilient_breaker_retries_transient():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return "recovered"

    result = run(call_resilient(flaky, breaker=CircuitBreaker(), attempts=3, retry_kwargs={"base_delay": 0.01}))
    assert result == "recovered"
    assert calls["n"] == 2


def test_call_resilient_breaker_trips_open():
    breaker = CircuitBreaker(failure_threshold=1, reset_seconds=60)

    async def always_fail():
        raise ConnectionError("boom")

    with pytest_raises(ConnectionError):
        run(call_resilient(always_fail, breaker=breaker, attempts=1))
    assert breaker.state == "open"


def test_429_status_is_retryable_from_httpstatuserror():
    exc = httpx.HTTPStatusError("429", request=httpx.Request("POST", "http://x"), response=httpx.Response(429))
    from backend.core.retry import _should_retry

    assert _should_retry(exc) is True


def test_400_status_is_not_retryable():
    exc = httpx.HTTPStatusError("400", request=httpx.Request("POST", "http://x"), response=httpx.Response(400))
    from backend.core.retry import _should_retry

    assert _should_retry(exc) is False


def pytest_raises(exc_type):
    class CM:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {exc_type.__name__}")
            return issubclass(et, exc_type)

    return CM()