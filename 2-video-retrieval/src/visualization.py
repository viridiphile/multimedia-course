"""Render actual retrieval results and the ViSiL matrices used to score a pair."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def retrieval_examples(path, cases):
    figure, axes = plt.subplots(len(cases), 4, squeeze=False,
                                figsize=(11, 3.8 * len(cases)), layout="constrained")
    for row, case in zip(axes, cases):
        query, query_frames, query_times, candidates = case
        row[0].set_title(f"Query • {query['source_id']}\nsource start {query['start_seconds']} s",
                         color="#1e293b", fontsize=10)
        _frame_strip(row[0], query_frames, query_times)
        for ax, (rank, frames, times) in zip(row[1:], candidates):
            label, color = ("Relevant", "#16734c") if rank["relevant"] else ("Negative", "#a3392a")
            ax.set_title(f"Rank {rank['rank']} • {rank['candidate_id']}\n"
                         f"{rank['similarity_score']:.4f} · {label}", color=color, fontsize=10)
            _frame_strip(ax, frames, times)
    figure.suptitle("TVSum Top-3 retrieval with ViSiL", fontsize=17, fontweight="bold")
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _frame_strip(axis, frames, timestamps):
    axis.imshow(np.concatenate(frames, axis=0))
    size = frames.shape[1]
    axis.axhline(size - .5, color="white", linewidth=2)
    for index, time in enumerate(timestamps):
        axis.text(6, (index + 1) * size - 8, f"{time:.1f} s", color="white", fontsize=8,
                  bbox=dict(facecolor="#101827", alpha=.8, edgecolor="none", pad=2))
    axis.set_axis_off()


def heatmap(path, pair):
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), layout="constrained")
    query_times = np.asarray(pair["query_times"])
    gallery_times = np.asarray(pair["candidate_times"])
    for ax, key, group, label, limits in [
        (axes[0], "frame_similarity", 1, "Regional Chamfer similarity", (0, .5)),
        (axes[1], "temporal_similarity", 4, "Learned temporal similarity (clipped)", (-1, 1)),
    ]:
        matrix = np.asarray(pair[key])
        dt_x = (gallery_times[1] - gallery_times[0]) * group
        dt_y = (query_times[1] - query_times[0]) * group
        extent = [-.5 * dt_x / group, (matrix.shape[1] - .5 / group) * dt_x,
                  (matrix.shape[0] - .5 / group) * dt_y, -.5 * dt_y / group]
        if group == 1:
            limits = (0, max(.1, float(matrix.max())))
        image = ax.imshow(matrix, cmap="viridis" if group == 1 else "magma",
                          interpolation="nearest", aspect="auto", extent=extent,
                          vmin=limits[0], vmax=limits[1])
        start = pair["source_start_seconds"]
        ax.axvspan(start, start + pair["source_clip_seconds"], facecolor="none",
                   edgecolor="#29c8ef", linewidth=1.4, linestyle="--")
        ax.set_ylabel("Query time (s)")
        ax.set_title(label, loc="left", fontsize=11)
        figure.colorbar(image, ax=ax, fraction=.04, pad=.02)
    axes[1].set_xlabel("Original candidate time (s)")
    figure.suptitle(f"{pair['query_id']} → {pair['candidate_id']}\n"
                    f"ViSiL {pair['similarity_score']:.4f} · rank {pair['candidate_rank']}",
                    fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)
