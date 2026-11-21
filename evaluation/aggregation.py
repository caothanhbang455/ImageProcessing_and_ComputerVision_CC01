# evaluation/aggregation.py
"""Coverage-aware summaries of per-sample results."""

from collections import Counter
from evaluation.metrics import finite_summary


def aggregate_rows(rows : list[dict]) -> list[dict] :
    """Keep one row per protocol/metric and retain missing measurements in counts."""
    grouped : dict[tuple[str, str], list[dict]] = {}
    for row in rows :
        key = (str(row["protocol_id"]), str(row["metric_name"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (protocol, metric), group in sorted(grouped.items()) :
        statuses = Counter(row["status"] for row in group)
        measured = [float(row["value"]) if row["status"] == "OK" else None for row in group]
        stats = finite_summary(measured)
        summaries.append({"protocol_id" : protocol, "metric_name" : metric,
                          "unit" : group[0]["unit"], "higher_is_better" : group[0]["higher_is_better"],
                          "n_attempted" : len(group), "n_metric_ok" : statuses["OK"],
                          "n_metric_missing" : len(group) - statuses["OK"],
                          "coverage_fraction" : statuses["OK"] / len(group),
                          "reference_verified" : all(row["reference_verified"] for row in group),
                          **stats})
    return summaries
