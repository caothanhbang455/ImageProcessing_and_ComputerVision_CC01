# tests/test_evaluation_metrics.py
import numpy as np
import pytest

from evaluation.metrics import MetricResult, finite_summary, pixel_difference_statistics
from evaluation.aggregation import aggregate_rows


def test_pixel_diff_correct_and_never_resizes() :
    a = np.array([[0, 10]], dtype=np.uint8)
    b = np.array([[2, 6]], dtype=np.uint8)
    result = pixel_difference_statistics(a, b)
    assert result["mae"] == 3.0 and result["max_abs"] == 4.0
    assert result["rmse"] == pytest.approx(np.sqrt(10.0))
    with pytest.raises(ValueError) :
        pixel_difference_statistics(a, b.reshape((2, 1)))
    with pytest.raises(ValueError) :
        pixel_difference_statistics(a.astype(float), np.full((1, 2), np.nan))


def test_missing_metrics_are_not_zero() :
    stats = finite_summary([1, None, 3, np.nan])
    assert stats["n_expected"] == 4 and stats["n_valid"] == 2
    assert stats["mean"] == 2
    assert finite_summary([None])["mean"] is None
    with pytest.raises(ValueError) :
        MetricResult("sample", "model", "protocol", "metric", None, "px", False, "OK")
    with pytest.raises(ValueError) :
        MetricResult("sample", "model", "protocol", "metric", 0., "px", False, "METRIC_FAILED", "failure")


def test_aggregate_keeps_failed_denominator() :
    base = dict(sample_key="a", method_id="model", protocol_id="p", metric_name="h", unit="px",
                higher_is_better=False, reference_verified=False)
    rows = [MetricResult(**base, value=2.0, status="OK").as_row(),
            MetricResult(**{**base, "sample_key": "b"}, value=None, status="UNAVAILABLE_REFERENCE",
                         reason="missing").as_row()]
    result = aggregate_rows(rows)[0]
    assert result["n_attempted"] == 2 and result["n_metric_ok"] == 1
    assert result["coverage_fraction"] == 0.5 and result["mean"] == 2.0
