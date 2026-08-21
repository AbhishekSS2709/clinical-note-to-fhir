import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_serving import percentile


def test_percentile_p50_is_the_median_value():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_p95_does_not_interpolate():
    # statistics.quantiles would return a value BETWEEN samples; a latency p95
    # must be a latency that was actually observed.
    values = [float(i) for i in range(1, 21)]
    assert percentile(values, 95) in values
    assert percentile(values, 95) == 19.0


def test_percentile_p100_is_the_maximum():
    assert percentile([4, 1, 9, 3], 100) == 9


def test_percentile_of_empty_is_zero_not_an_error():
    assert percentile([], 95) == 0.0
