import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_eval import latency_stats


def _records(lats):
    return [{"latency_ms": v, "error": None} for v in lats]


def test_latency_stats_basic():
    stats = latency_stats(_records([100.0, 200.0, 300.0]))
    assert stats["count"] == 3
    assert stats["mean_ms"] == 200.0
    assert stats["p50_ms"] == 200.0
    assert stats["max_ms"] == 300.0


def test_p95_uses_nearest_rank():
    # nearest-rank: ceil(0.95 * 10) = 10th value
    lats = [float(i) for i in range(1, 11)]
    stats = latency_stats(_records(lats))
    assert stats["p95_ms"] == 10.0

    # ceil(0.95 * 48) = 46th value
    lats48 = [float(i) for i in range(1, 49)]
    stats48 = latency_stats(_records(lats48))
    assert stats48["p95_ms"] == 46.0


def test_latency_stats_skips_failures():
    records = _records([100.0, 200.0])
    records.append({"latency_ms": 9999.0, "error": "boom"})
    stats = latency_stats(records)
    assert stats["count"] == 2
    assert stats["max_ms"] == 200.0
