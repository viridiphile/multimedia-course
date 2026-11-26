import numpy as np


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    """Convert LR offsets to bounding boxes.

    :param offsets: LR offsets. Sized [N, 2].
    :return: Bounding boxes corresponding to offsets. Sized [N, 2].
    """
    offset_left, offset_right = offsets[:, 0], offsets[:, 1]
    seq_len, _ = offsets.shape
    indices = np.arange(seq_len)
    bbox_left = indices - offset_left
    bbox_right = indices + offset_right + 1
    bboxes = np.vstack((bbox_left, bbox_right)).T
    return bboxes
