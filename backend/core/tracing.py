"""Span instrumentation for the request path.

Every stage must record a span. Stages don't thread a tracer through their
signatures — they grab the current trace from a ContextVar, so pipelines and
benchmarks stay decoupled.

Compatible with OpenTelemetry/Langfuse-style exporters via a pluggable sink.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator

from .models import StageSpan

_trace_var: ContextVar["Trace | None"] = ContextVar("vakrag_trace", default=None)

# Optional external sink (e.g. Langfuse). Set at startup by main.py.
_sink: Callable[[dict[str, Any]], None] | None = None


def set_trace_sink(fn: Callable[[dict[str, Any]], None] | None) -> None:
    global _sink
    _sink = fn


class Trace:
    """Collects stage spans for a single request."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.spans: list[StageSpan] = []
        self._start = time.perf_counter()

    def record(self, name: str, duration_ms: float, ok: bool = True, detail: str | None = None) -> None:
        span = StageSpan(name=name, duration_ms=round(duration_ms, 3), ok=ok, detail=detail)
        self.spans.append(span)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def finish(self, payload: dict[str, Any] | None = None) -> None:
        if _sink is not None:
            _sink({"request_id": self.request_id, "spans": [s.model_dump() for s in self.spans], **({"payload": payload} if payload else {})})


def begin_trace(request_id: str) -> Trace:
    trace = Trace(request_id)
    _trace_var.set(trace)
    return trace


def current_trace() -> Trace | None:
    return _trace_var.get()


@contextmanager
def span(name: str, detail: str | None = None) -> Generator[None, None, None]:
    """Records the elapsed time of the wrapped block under `name`."""
    trace = current_trace()
    if trace is None:
        yield
        return
    start = time.perf_counter()
    ok = True
    try:
        yield
    except Exception as exc:
        ok = False
        raise
    finally:
        trace.record(name, (time.perf_counter() - start) * 1000.0, ok=ok, detail=detail)