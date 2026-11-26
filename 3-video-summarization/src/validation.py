"""Independently verify frame count, cadence, streams, and decoded source correspondence."""

from dataclasses import asdict
from fractions import Fraction
from itertools import zip_longest

import av
import numpy as np

from .dataset import probe_video
from .segments import frame_budget


MAX_FRAME_LUMA_ERROR = 4.0  # Mean absolute error in 8-bit luma after downsampling.


def _luma(frame):
    return frame.reformat(width=96, height=54, format="gray").to_ndarray().astype(np.int16)


def _selected_frames(container, intervals):
    current = 0
    for index, frame in enumerate(container.decode(video=0)):
        if index == intervals[current][1]:
            current += 1
            if current == len(intervals):
                break
        if intervals[current][0] <= index:
            yield frame


def verify_video(source, summary, source_metadata, intervals):
    # Independent selection audit before decoding the generated artifact.
    count, last_stop = 0, 0
    for first, stop in intervals:
        if first < last_stop or stop <= first or stop > source_metadata.frame_count:
            raise ValueError("Saved interval is invalid or out of chronological order")
        count += stop - first
        last_stop = stop
    limit = frame_budget(source_metadata.frame_count)
    if not 0 < count <= limit:
        raise ValueError(f"Invalid retained frame count {count}; budget is {limit}")
    actual = probe_video(summary)
    if ("mp4" not in actual.format.split(",") or actual.codec != "h264" or actual.audio_streams
            or actual.frame_count != count or actual.fps != source_metadata.fps
            or (actual.width, actual.height) != (source_metadata.width, source_metadata.height)):
        raise ValueError(f"Export metadata mismatch: {actual}")
    with av.open(str(summary)) as result:
        if len(result.streams) != 1 or result.streams.video[0].codec_context.pix_fmt != "yuv420p":
            raise ValueError("Expected a single H.264 yuv420p video stream")
        ratio = result.streams.video[0].sample_aspect_ratio or Fraction(1, 1)
        with av.open(str(source)) as original:
            if ratio != (original.streams.video[0].sample_aspect_ratio or Fraction(1, 1)):
                raise ValueError("Export changed the sample aspect ratio")
            errors = []
            for expected, exported in zip_longest(_selected_frames(original, intervals),
                                                   result.decode(video=0)):
                if expected is None or exported is None:
                    raise ValueError("Decoded source and summary lists differ in length")
                errors.append(float(np.abs(_luma(expected) - _luma(exported)).mean()))
    if len(errors) != count or max(errors) > MAX_FRAME_LUMA_ERROR:
        raise ValueError("A decoded summary frame differs from its expected source position")
    return {
        "validation_passed": True,
        "selected_source_frames": count,
        "total_source_frames": source_metadata.frame_count,
        "frame_budget": limit,
        "summary_ratio": count / source_metadata.frame_count,
        "summary": asdict(actual),
        "no_audio": actual.audio_streams == 0,
        "decoded_frame_matches": len(errors),
        "mean_frame_luma_error": float(np.mean(errors)),
        "max_frame_luma_error": max(errors),
        "frame_luma_error_limit": MAX_FRAME_LUMA_ERROR,
    }
