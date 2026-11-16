"""Precision@5 evaluation and machine-readable result export."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def evaluate_precision_at_k(results: Sequence[Dict], k: int = 5) -> Dict:
    if k < 1:
        raise ValueError("k must be positive")
    grouped: Dict[str, List[Dict]] = {}
    for row in results:
        grouped.setdefault(row["query_id"], []).append(row)
    if not grouped:
        raise ValueError("No retrieval results to evaluate")

    per_query = []
    for query_id in sorted(grouped):
        rows = sorted(grouped[query_id], key=lambda row: row["rank"])
        if len(rows) != k or [row["rank"] for row in rows] != list(range(1, k + 1)):
            raise ValueError(
                "Query {} must have exactly ranks 1..{} to calculate P@{}".format(
                    query_id, k, k
                )
            )
        if any(not isinstance(row["relevant"], bool) for row in rows):
            raise ValueError("Relevance must be boolean, not a CSV string")
        ids = [row["image_id"] for row in rows if "image_id" in row]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate gallery result for query {}".format(query_id))
        precision = sum(bool(row["relevant"]) for row in rows) / float(k)
        per_query.append({"query_id": query_id, "precision_at_5": precision})

    values = [row["precision_at_5"] for row in per_query]
    distribution = Counter("{:.1f}".format(value) for value in values)
    return {
        "queries_evaluated": len(per_query),
        "precision_at_5_per_query": per_query,
        "mean_precision_at_5": sum(values) / len(values),
        "queries_precision_at_5_equal_1": sum(value == 1.0 for value in values),
        "queries_precision_at_5_equal_0": sum(value == 0.0 for value in values),
        "precision_at_5_distribution": dict(sorted(distribution.items())),
    }


def write_results_csv(results: Iterable[Dict], path: Path) -> None:
    rows = list(results)
    if not rows:
        raise ValueError("Cannot save an empty results file")
    _write_dict_csv(rows, path)


def write_metrics(metrics: Dict, csv_path: Path, json_path: Path) -> None:
    per_query = metrics["precision_at_5_per_query"]
    _write_dict_csv(per_query, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_dict_csv(rows: List[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
