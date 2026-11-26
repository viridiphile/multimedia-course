"""Plot the measured DSNet predictions and frames decoded from the MP4."""

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#155e75"
ORANGE = "#b45309"


def decode_examples(path, indices):
    wanted = set(indices)
    found = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                found[index] = frame.to_ndarray(format="rgb24")
                if len(found) == len(wanted):
                    break
    if set(found) != wanted:
        raise ValueError(f"Requested representative frames do not exist in {path}")
    return [found[index] for index in indices]


def draw_figures(directory, source, summary, metrics, segments, predictions, shots):
    directory.mkdir(parents=True, exist_ok=True)
    source_info, summary_info = metrics["source"], metrics["summary"]
    duration = source_info["duration_seconds"]

    fig, ax = plt.subplots(figsize=(11, 2.2), layout="constrained")
    ax.axhspan(0, 1, color="#e2e8f0")
    for row in segments:
        start, stop = row["start_seconds"], row["end_seconds"]
        ax.axvspan(start, stop, facecolor=BLUE, edgecolor="white", linewidth=.7)
        index = row["segment_index"]
        ax.text((start + stop) / 2, .38 if index % 2 else .65, str(index),
                color="white", ha="center", va="center", fontweight="bold", fontsize=9)
    ax.set(xlim=(0, duration), ylim=(0, 1), yticks=[], xlabel="Source time (s)")
    ax.set_title(f"Selected shots · {source_info['title']}", loc="left", fontweight="bold")
    fig.savefig(directory / "selected_timeline.png", dpi=170, facecolor="white")
    plt.close(fig)

    middle = [row["summary_start_frame"] + row["selected_frames"] // 2 for row in segments]
    summary_frames = decode_examples(summary, middle)
    columns = min(3, len(segments))
    rows = (len(segments) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 2.8 * rows),
                            squeeze=False, layout="constrained")
    for ax, image, row in zip(axes.flat, summary_frames, segments):
        fps = source_info["fps_numerator"] / source_info["fps_denominator"]
        timestamp = (row["start_frame"] + row["selected_frames"] // 2) / fps
        ax.imshow(image)
        ax.set_title(f"Segment {row['segment_index']} · source {timestamp:.2f} s",
                     loc="left", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Representative frames decoded from the summary", fontweight="bold")
    fig.savefig(directory / "summary_frames.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), layout="constrained")
    source_count = source_info["frame_count"]
    summary_count = summary_info["frame_count"]
    fps = source_info["fps_numerator"] / source_info["fps_denominator"]
    for column, index in enumerate((source_count // 6, source_count // 2, 5 * source_count // 6)):
        axes[0, column].set_title(f"Source {index / fps:.2f} s", loc="left")
    source_indices = (source_count // 6, source_count // 2, 5 * source_count // 6)
    summary_indices = (summary_count // 6, summary_count // 2, 5 * summary_count // 6)
    for ax, frame in zip(axes[0], decode_examples(source, source_indices)):
        ax.imshow(frame)
    for ax, index, frame in zip(axes[1], summary_indices, decode_examples(summary, summary_indices)):
        ax.imshow(frame)
        ax.set_title(f"Summary {index / fps:.2f} s", loc="left")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Original  {duration:.2f} s · {source_count:,} frames     "
                 f"Summary  {summary_info['duration_seconds']:.2f} s · {summary_count:,} frames\n"
                 f"{metrics['summary_ratio']:.2%} of original frames retained",
                 fontweight="bold", fontsize=14)
    fig.savefig(directory / "original_vs_summary.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, (ax, shot_ax) = plt.subplots(2, 1, sharex=True, figsize=(11, 5.1), layout="constrained")
    times = [row["source_seconds"] for row in predictions]
    ax.plot(times, [row["confidence"] for row in predictions], color="#7d8e9e",
            label="Detection confidence", linewidth=1)
    ax.step([*times, duration], [*(row["post_nms_bin_score"] for row in predictions),
            predictions[-1]["post_nms_bin_score"]], where="post", color=BLUE,
            label="Score after NMS and projection", linewidth=1.4)
    for shot in shots:
        shot_ax.hlines(shot["score"], shot["start_seconds"], shot["end_seconds"],
                      colors=ORANGE if shot["selected"] else "#7d8e9e", linewidth=2.7)
    for row in segments:
        for axis in (ax, shot_ax):
            axis.axvspan(row["start_seconds"], row["end_seconds"], color="#f59e0b", alpha=.18)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    fig.suptitle("Anchor-free DSNet scores through time", fontweight="bold")
    ax.set(ylabel="Normalized score", ylim=(0, 1.06))
    shot_ax.set(ylabel="Mean shot score", ylim=(0, 1.06), xlim=(0, duration),
                xlabel="Source time (s)")
    fig.savefig(directory / "dsnet_scores.png", dpi=170, facecolor="white")
    plt.close(fig)
