"""In-memory latency tracker for the live P50/P70/P100 panel.

A bounded ring buffer of per-query total and per-stage latencies, sampled
continuously from the live request path. Exposed via GET /v1/telemetry and
shown in the frontend dashboard.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class LatencyTracker:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._totals: deque[float] = deque(maxlen=capacity)
        self._stages: dict[str, deque[float]] = {}
        self._count = 0
        self._refusals = 0

    def record(self, total_ms: float, stages: list[Any], *, refused: bool) -> None:
        with self._lock:
            self._totals.append(total_ms)
            self._count += 1
            if refused:
                self._refusals += 1
            for span in stages:
                buf = self._stages.setdefault(span.name, deque(maxlen=self.capacity))
                buf.append(float(span.duration_ms))

    def snapshot(self) -> dict:
        with self._lock:
            totals = list(self._totals)
            stage_stats: dict[str, dict[str, float]] = {}
            for name, buf in self._stages.items():
                vals = list(buf)
                stage_stats[name] = {
                    "p50": round(_percentile(vals, 50), 2),
                    "p70": round(_percentile(vals, 70), 2),
                    "p100": round(_percentile(vals, 100), 2),
                    "n": len(vals),
                }
            return {
                "requests": self._count,
                "refusals": self._refusals,
                "total_ms": {
                    "p50": round(_percentile(totals, 50), 2),
                    "p70": round(_percentile(totals, 70), 2),
                    "p100": round(_percentile(totals, 100), 2),
                    "n": len(totals),
                },
                "stages": stage_stats,
            }


tracker = LatencyTracker()