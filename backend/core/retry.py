"""Resilience primitives for external calls: exponential backoff + jitter,
and a circuit breaker. Every provider call in this project must go through
these (see core/providers.py).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

RetryableExc = (ConnectionError, TimeoutError, OSError)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, RetryableExc):
        return True
    # HTTP 429/5xx style errors are retryable; anything else isn't.
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status is not None and (status == 429 or status >= 500)


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.15,
    max_delay: float = 2.0,
    jitter: float = 0.2,
    on_retry: Callable[[Exception, int], None] | None = None,
    **kwargs,
) -> T:
    """Call `fn(**kwargs)` with exponential backoff + jitter on retryable errors."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts or not _should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (1.0 + random.uniform(-jitter, jitter))
            if on_retry:
                on_retry(exc, attempt)
            logger.warning("Retry %d/%d for %s in %.3fs: %s", attempt, attempts, getattr(fn, "__name__", fn), delay, exc)
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover — unreachable


@dataclass
class CircuitBreaker:
    """Trips after `failure_threshold` consecutive failures; half-opens after
    `reset_seconds` so a single success closes it again."""

    failure_threshold: int = 5
    reset_seconds: float = 15.0
    _failures: int = field(default=0, init=False)
    _state: str = field(default="closed", init=False)  # closed | open | half_open
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> str:
        if self._state == "open" and (time.monotonic() - self._opened_at) >= self.reset_seconds:
            self._state = "half_open"
        return self._state

    def allow(self) -> bool:
        return self.state != "open"

    def on_success(self) -> None:
        self._failures = 0
        if self._state == "half_open":
            self._state = "closed"

    def on_failure(self) -> None:
        if self._state == "open":
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker OPEN (%s)", self.__class__.__name__)

    async def call(self, fn: Callable[..., Awaitable[T]], **kwargs) -> T:
        if not self.allow():
            raise ConnectionError(f"Circuit breaker {self.__class__.__name__} is OPEN")
        try:
            result = await fn(**kwargs)
            self.on_success()
            return result
        except Exception as exc:  # noqa: BLE001
            self.on_failure()
            raise


async def call_resilient(
    fn: Callable[..., Awaitable[T]],
    *,
    breaker: CircuitBreaker | None = None,
    attempts: int = 3,
    **kwargs,
) -> T:
    """Retry + circuit-breaker in one call. This is the only sanctioned way to
    hit an external provider in VakRAG."""
    async def inner() -> T:
        return await retry_with_backoff(fn, attempts=attempts, **kwargs.pop("retry_kwargs", {}))
    if breaker is not None:
        return await breaker.call(inner)
    return await inner()