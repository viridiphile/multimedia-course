"""Run official anchor-free DSNet prediction and reference temporal NMS."""

from dataclasses import dataclass

import numpy as np
import torch

from .model import FEATURE_DIM
from .reference.anchor_free_helper import offset2bbox
from .reference.bbox_helper import nms


NMS_THRESHOLD = 0.4


@dataclass(frozen=True)
class Predictions:
    class_scores: np.ndarray
    centerness: np.ndarray
    offsets: np.ndarray
    confidence: np.ndarray
    raw_boxes: np.ndarray
    clipped_boxes: np.ndarray
    kept_scores: np.ndarray
    kept_boxes: np.ndarray
    bin_scores: np.ndarray


@torch.inference_mode()
def predict(model, features):
    if (features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) < 1
            or features.dtype != np.float32 or not np.isfinite(features).all()):
        raise ValueError("Expected finite float32 GoogLeNet features with shape (N, 1024)")
    cls, offsets, center = model(torch.from_numpy(features).unsqueeze(0))
    scores = cls * center
    scores /= scores.max() + 1e-8
    tensors = [value.numpy() for value in (cls, offsets, center, scores)]
    if not all(np.isfinite(array).all() for array in tensors) or (tensors[1] <= 0).any():
        raise ValueError("DSNet returned invalid scores or nonpositive temporal offsets")
    cls, offsets, center, scores = tensors
    raw = offset2bbox(offsets)
    clipped = np.clip(raw, 0, len(features)).round().astype(np.int32)
    kept_scores, kept_boxes = nms(scores, clipped, NMS_THRESHOLD)

    bin_scores = np.zeros(len(features), np.float32)
    for score, (first, stop) in zip(kept_scores, kept_boxes):
        np.maximum(bin_scores[first:stop], score, out=bin_scores[first:stop])
    return Predictions(cls, center, offsets, scores, raw, clipped, kept_scores,
                       kept_boxes.reshape(-1, 2), bin_scores)
