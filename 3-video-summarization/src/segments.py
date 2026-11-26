"""Project detection scores to source shots and select them within the frame budget."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    shot_index: int
    start_frame: int
    end_frame_exclusive: int
    score: float

    @property
    def frame_count(self):
        return self.end_frame_exclusive - self.start_frame


def frame_budget(source_frames):
    return source_frames * 15 // 100


def knapsack(values, weights, capacity):
    """Maximize integer shot scores. Keep the earlier decisions on equal objectives."""
    best = np.zeros(capacity + 1, np.int32)
    decisions = np.zeros((len(weights), capacity + 1), dtype=bool)
    for index, (weight, value) in enumerate(zip(weights, values)):
        if weight > capacity or value <= 0:
            continue
        alternatives = best[:capacity + 1 - weight] + value
        updated = alternatives > best[weight:]
        decisions[index, weight:] = updated
        best[weight:][updated] = alternatives[updated]
    selected = []
    space = capacity
    for index in range(len(weights) - 1, -1, -1):
        if decisions[index, space]:
            selected.append(index)
            space -= weights[index]
    return selected[::-1]


def choose_shots(record, bin_scores):
    if (bin_scores.shape != record.picks.shape or not np.isfinite(bin_scores).all()
            or (bin_scores < 0).any() or (bin_scores > 1).any()):
        raise ValueError("Need one finite normalized score per feature bin")
    widths = np.diff(np.append(record.picks, record.n_frames))
    frame_scores = np.repeat(bin_scores, widths)
    scores = np.array([frame_scores[start:end + 1].mean() for start, end in record.shots])
    # Match the reference integer shot values used in knapsack.
    values = [int(1000 * score) for score in scores]
    winners = knapsack(values, record.shot_lengths, frame_budget(record.n_frames))
    segments = [
        Segment(index + 1, int(record.shots[index, 0]), int(record.shots[index, 1]) + 1,
                float(scores[index]))
        for index in winners
    ]
    return scores, segments


def verify_intervals(intervals, source_frames):
    if not intervals:
        raise ValueError("DSNet selected no positive whole shots within the frame budget")
    previous = 0
    selected = 0
    for start, stop in intervals:
        if not previous <= start < stop <= source_frames:
            raise ValueError("Intervals must be in range, disjoint, and chronological")
        previous = stop
        selected += stop - start
    limit = frame_budget(source_frames)
    if selected > limit:
        raise ValueError(f"Selected {selected} frames exceeds the {limit}-frame budget")
    return selected
