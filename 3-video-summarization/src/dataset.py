"""Validate complete source video and DSNet's temporal feature mapping."""

import csv
from dataclasses import dataclass
from fractions import Fraction

import av
import h5py
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    filename: str
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    duration_seconds: float
    width: int
    height: int
    codec: str
    format: str
    audio_streams: int
    max_timestamp_drift_seconds: float

    @property
    def fps(self):
        return Fraction(self.fps_numerator, self.fps_denominator)


@dataclass(frozen=True)
class FeatureRecord:
    key: str
    features: np.ndarray
    picks: np.ndarray
    shots: np.ndarray
    shot_lengths: np.ndarray
    n_frames: int
    gtscore: np.ndarray
    user_summary_shape: tuple
    fields: tuple


def probe_video(path):
    """Decode every frame and verify timestamps correspond to a constant cadence."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if not rate or rate <= 0:
            raise ValueError(f"Cannot establish video frame rate for {path}")
        count = 0
        worst_drift = 0.0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Decoded frame has no timestamp in {path}")
            timestamp = frame.pts * frame.time_base
            if count == 0:
                origin = timestamp
            elif timestamp <= previous:
                raise ValueError(f"Decoded timestamps are unordered in {path}")
            worst_drift = max(worst_drift, abs(float((timestamp - origin) - count / rate)))
            previous = timestamp
            count += 1
        if not count or worst_drift > float(Fraction(1, 2) / rate):
            raise ValueError(f"Expected constant-rate frames in {path}")
        duration = stream.duration * stream.time_base if stream.duration is not None else count / rate
        if abs(duration - count / rate) > Fraction(1, 2) / rate:
            raise ValueError(f"Container duration disagrees with decoded frame cadence in {path}")
        return VideoMetadata(path.name, count, rate.numerator, rate.denominator, float(duration),
                             stream.width, stream.height, stream.codec_context.name,
                             container.format.name, len(container.streams.audio), worst_drift)


def load_features(path, feature_key):
    with h5py.File(path, "r") as dataset:
        group = dataset[feature_key]
        record = FeatureRecord(
            feature_key, group["features"][:], group["picks"][:], group["change_points"][:],
            group["n_frame_per_seg"][:], int(group["n_frames"][()]), group["gtscore"][:],
            group["user_summary"].shape, tuple(sorted(group.keys())),
        )
        if int(group["n_steps"][()]) != len(record.features):
            raise ValueError(f"Wrong number of feature steps in {feature_key}")
    validate_mapping(record)
    return record


def validate_mapping(record):
    n = len(record.features)
    if (record.features.shape != (n, 1024) or n < 1 or record.features.dtype != np.float32
            or not np.isfinite(record.features).all()):
        raise ValueError("Invalid GoogLeNet feature matrix")
    if (record.picks.shape != (n,) or record.picks.dtype.kind not in "iu"
            or record.picks[0] != 0 or record.picks[-1] >= record.n_frames
            or np.any(record.picks[1:] <= record.picks[:-1])):
        raise ValueError("Feature picks must increase from frame zero inside the original")
    shots = record.shots
    if (shots.ndim != 2 or shots.shape[1] != 2 or len(shots) < 1 or shots.dtype.kind not in "iu"
            or shots[0, 0] != 0 or shots[-1, 1] != record.n_frames - 1
            or np.any(shots[:, 0] > shots[:, 1]) or np.any(shots[1:, 0] != shots[:-1, 1] + 1)
            or not np.array_equal(record.shot_lengths, shots[:, 1] - shots[:, 0] + 1)):
        raise ValueError("HDF5 inclusive shot ranges must partition the source")
    if record.user_summary_shape != (20, record.n_frames):
        raise ValueError("Expected twenty TVSum annotations with the original length")


def tvsum_entry(info_file, video_id):
    with info_file.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE))
    matches = [(index, row) for index, row in enumerate(rows, 1) if row["video_id"] == video_id]
    if len(rows) != 50 or len(matches) != 1:
        raise ValueError(f"{video_id} is not uniquely identified in the 50-video TVSum catalog")
    index, row = matches[0]
    return f"video_{index}", row


def annotation_scores(path, video_id):
    values = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            identifier, _, scores = line.rstrip().split("\t")
            if identifier == video_id:
                values.append(np.fromstring(scores, sep=",", dtype=np.int8))
    if len(values) != 20 or len({len(row) for row in values}) != 1:
        raise ValueError(f"Expected twenty equal-length annotations for {video_id}")
    return np.stack(values)
