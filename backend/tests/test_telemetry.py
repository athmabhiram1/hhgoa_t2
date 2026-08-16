import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.models import StageSpan
from backend.harness.telemetry import LatencyTracker


def test_tracker_records_and_snapshots():
    t = LatencyTracker(capacity=100)
    t.record(120.0, [StageSpan(name="retrieval", duration_ms=60.0), StageSpan(name="generate", duration_ms=55.0)], refused=False)
    t.record(180.0, [StageSpan(name="retrieval", duration_ms=90.0), StageSpan(name="generate", duration_ms=85.0)], refused=False)
    snap = t.snapshot()
    assert snap["requests"] == 2
    assert snap["total_ms"]["p50"] == 150.0
    assert snap["total_ms"]["p100"] == 180.0
    assert snap["stages"]["retrieval"]["p100"] == 90.0


def test_tracker_tracks_refusals():
    t = LatencyTracker()
    t.record(50.0, [], refused=True)
    snap = t.snapshot()
    assert snap["refusals"] == 1


def test_tracker_empty_snapshot_safe():
    t = LatencyTracker()
    snap = t.snapshot()
    assert snap["requests"] == 0
    assert snap["total_ms"]["p50"] == 0.0