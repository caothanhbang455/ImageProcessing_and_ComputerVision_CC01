# evaluation/metrics.py
"""Numerical diagnostics without hidden resizing, registration, or cropping."""

from dataclasses import dataclass
from typing import Any

import numpy as np


METRIC_STATUSES = frozenset({
    "OK", "UNAVAILABLE_PREDICTION", "UNAVAILABLE_REFERENCE", "UNAVAILABLE_DEPENDENCY",
    "INVALID_INPUT", "METRIC_FAILED", "NOT_APPLICABLE", "REVIEW_REQUIRED",
})


@dataclass(frozen=True)
class MetricResult :
    sample_key : str
    method_id : str
    protocol_id : str
    metric_name : str
    value : float | None
    unit : str
    higher_is_better : bool
    status : str
    reason : str = ""
    reference_verified : bool = False

    def __post_init__(self) :
        if (not all((self.sample_key, self.method_id, self.protocol_id, self.metric_name, self.unit))) :
            raise ValueError("Metric identities and unit must be nonempty")
        if (self.status not in METRIC_STATUSES) :
            raise ValueError(f"Unknown metric status: {self.status}")
        if (self.status == "OK") :
            if (self.value is None or not np.isfinite(self.value) or self.reason) :
                raise ValueError("OK metric needs a finite value and no failure reason")
        elif (self.value is not None or not self.reason) :
            raise ValueError("Unavailable metrics need no value and a reason")

    def as_row(self) -> dict[str, Any] :
        return vars(self).copy()


def pixel_difference_statistics(a : np.ndarray, b : np.ndarray) -> dict[str, float] :
    """Compute raw pixel deviations, without interpreting them as benchmark scores."""
    a = np.asarray(a)
    b = np.asarray(b)
    if (a.shape != b.shape or a.size == 0 or a.ndim not in (2, 3)) :
        raise ValueError("Pixel diagnostics require equally shaped nonempty images")
    if (not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number)) :
        raise TypeError("Pixel diagnostics require numeric arrays")
    if (not np.isfinite(a).all() or not np.isfinite(b).all()) :
        raise ValueError("Pixel diagnostics require finite inputs")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"mae" : float(delta.mean()), "max_abs" : float(delta.max()),
            "rmse" : float(np.sqrt(np.mean(delta ** 2)))}


def finite_summary(values : list[float | None]) -> dict[str, float | int | None] :
    """Summarize only finite measurements and report missing measurements separately."""
    values = list(values)
    valid = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    result : dict[str, float | int | None] = {"n_expected" : len(values), "n_valid" : len(valid),
                                             "n_missing" : len(values) - len(valid)}
    for metric in ("mean", "median", "std", "min", "max", "p05", "p95") :
        result[metric] = None
    if (valid.size) :
        result.update(mean=float(valid.mean()), median=float(np.median(valid)),
                      std=float(valid.std()), min=float(valid.min()), max=float(valid.max()),
                      p05=float(np.percentile(valid, 5)), p95=float(np.percentile(valid, 95)))
    return result
