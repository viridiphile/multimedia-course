"""Report-ready retrieval and precision figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _open_image(path: str):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _query_groups(results: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in results:
        grouped.setdefault(row["query_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["rank"])
    return grouped


def plot_retrieval_examples(
    records_by_id: Dict[str, Dict], results: Sequence[Dict], output_path: Path, count: int = 10
) -> None:
    grouped = _query_groups(results)
    query_ids = sorted(grouped)
    if not query_ids:
        raise ValueError("No retrieval results available for visualization")
    indexes = np.linspace(0, len(query_ids) - 1, min(count, len(query_ids)), dtype=int)
    selected = [query_ids[index] for index in indexes]
    figure, axes = plt.subplots(len(selected), 6, figsize=(18, 3.2 * len(selected)))
    axes = np.atleast_2d(axes)
    for row_index, query_id in enumerate(selected):
        results_for_query = grouped[query_id]
        query_record = records_by_id[query_id]
        axes[row_index, 0].imshow(_open_image(query_record["path"]))
        axes[row_index, 0].set_title(
            "QUERY {} · obj{} · {}°".format(
                query_id, query_record["object_id"], query_record["angle"]
            ),
            fontsize=9,
        )
        for spine in axes[row_index, 0].spines.values():
            spine.set_edgecolor("#245bdb")
            spine.set_linewidth(4)
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        for column, result in enumerate(results_for_query, start=1):
            record = records_by_id[result["image_id"]]
            axes[row_index, column].imshow(_open_image(record["path"]))
            color = "#16833b" if result["relevant"] else "#c62828"
            axes[row_index, column].set_title(
                "Top-{} · {} · {:.5f}".format(
                    result["rank"], result["image_id"], result["similarity"]
                ),
                fontsize=8,
                color=color,
            )
            axes[row_index, column].set_xlabel(
                "obj{} · {}° · {}".format(
                    result["object_id"],
                    result["angle"],
                    "MATCH" if result["relevant"] else "INCORRECT",
                ),
                fontsize=8,
                color=color,
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    figure.suptitle("COIL-100 VGG16-GeM Top-5 retrieval examples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    _save_figure(figure, output_path, dpi=180)


def plot_precision(metrics: Dict, output_path: Path) -> None:
    values = [row["precision_at_5"] for row in metrics["precision_at_5_per_query"]]
    bins = np.arange(-0.05, 1.06, 0.1)
    mean = metrics["mean_precision_at_5"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(values, bins=bins, color="#4b79c9", edgecolor="white")
    axis.axvline(mean, color="#c62828", linestyle="--", linewidth=2,
                 label="Mean P@5 = {:.4f}".format(mean))
    axis.set(
        title="Query-level Precision@5 distribution",
        xlabel="Precision@5",
        ylabel="Number of queries",
        xlim=(-0.05, 1.05),
        xticks=np.arange(0, 1.01, 0.2),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi=220)


def plot_case_analysis(
    records_by_id: Dict[str, Dict],
    results: Sequence[Dict],
    precision_by_query: Dict[str, float],
    output_path: Path,
) -> Dict:
    grouped = _query_groups(results)
    case_id = min(
        grouped,
        key=lambda query_id: (precision_by_query[query_id], query_id),
    )
    rows = grouped[case_id]
    query = records_by_id[case_id]
    figure, axes = plt.subplots(1, 6, figsize=(22, 5.5))
    axes[0].imshow(_open_image(query["path"]))
    axes[0].set_title(
        "QUERY\n{} · obj{} · {}°\nP@5 = {:.1f}".format(
            case_id, query["object_id"], query["angle"], precision_by_query[case_id]
        ),
        fontsize=10,
    )
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#245bdb")
        spine.set_linewidth(4)
    for column, result in enumerate(rows, start=1):
        record = records_by_id[result["image_id"]]
        axes[column].imshow(_open_image(record["path"]))
        color = "#16833b" if result["relevant"] else "#c62828"
        axes[column].set_title(
            "Rank {} · {:.6f}\n{} · obj{} · {}°\n{}".format(
                result["rank"],
                result["similarity"],
                result["image_id"],
                result["object_id"],
                result["angle"],
                "RELEVANT" if result["relevant"] else "NOT RELEVANT",
            ),
            fontsize=9,
            color=color,
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Case analysis: {} (lowest observed P@5)".format(case_id), fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(figure, output_path, dpi=300)
    return {
        "query_id": case_id,
        "query_object_id": query["object_id"],
        "query_angle": query["angle"],
        "precision_at_5": precision_by_query[case_id],
        "results": rows,
    }


def _save_figure(figure, output_path: Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
