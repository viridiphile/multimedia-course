"""Synthetic checks for shot selection, frame timing, encoding and format integrity."""

from dataclasses import replace
from itertools import combinations
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import imageio_ffmpeg
import numpy as np
import pytest

from src.dataset import FeatureRecord, probe_video, validate_mapping
from src.reference.bbox_helper import nms
from src.resources import read_range_member
from src.segments import choose_shots, frame_budget, knapsack, verify_intervals
from src.validation import verify_video
from src.video_export import export


@pytest.fixture
def record():
    return FeatureRecord(
        "synthetic", np.ones((10, 1024), np.float32), np.arange(0, 100, 10),
        np.array([[0, 39], [40, 49], [50, 89], [90, 99]]), np.array([40, 10, 40, 10]),
        100, np.zeros(10), (20, 100), (),
    )


def test_feature_mapping_and_score_projection(record):
    validate_mapping(record)
    scores = np.full(10, .3, np.float32)
    scores[-1] = .9
    shot_scores, segments = choose_shots(record, scores)
    np.testing.assert_allclose(shot_scores, [.3, .3, .3, .9], atol=1e-6)
    assert [(item.shot_index, item.start_frame, item.end_frame_exclusive)
            for item in segments] == [(4, 90, 100)]


@pytest.mark.parametrize("field,value", [("picks", np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8])),
                                          ("shot_lengths", np.array([40, 0, 40, 20])),
                                          ("n_frames", 101)])
def test_mapping_rejects_mismatched_frames(record, field, value):
    with pytest.raises(ValueError):
        validate_mapping(replace(record, **{field: value}))


def test_reference_temporal_nms():
    scores = np.array([.99, .85, .75, .5])
    boxes = np.array([[0, 6], [1, 6], [9, 12], [3, 3]])
    kept_scores, kept = nms(scores, boxes, .4)
    np.testing.assert_array_equal(kept_scores, [.99, .75])
    np.testing.assert_array_equal(kept, [[0, 6], [9, 12]])


def test_knapsack_matches_exhaustive_search_and_breaks_ties():
    values = [8, 11, 6, 6, 3]
    weights = [4, 6, 3, 5, 1]
    capacity = 10
    feasible = [indices for size in range(len(weights) + 1)
                for indices in combinations(range(len(weights)), size)
                if sum(weights[i] for i in indices) <= capacity]
    selected = knapsack(values, weights, capacity)
    assert sum(values[i] for i in selected) == max(sum(values[i] for i in group) for group in feasible)
    assert selected == [1, 2, 4]
    # Equal maximum utility keeps the decision from the earlier shots.
    assert knapsack([8, 11, 5, 6, 3], weights, capacity) == [0, 1]
    assert knapsack([3, 3], [5, 5], 5) == [0]
    assert knapsack([0, 0], [5, 5], 0) == []


@pytest.mark.parametrize("intervals", [[], [(0, 16)], [(4, 8), (7, 10)], [(5, 10), (1, 4)],
                                         [(-1, 1)], [(98, 101)]])
def test_rejects_invalid_chronology_bounds_and_budget(intervals):
    with pytest.raises(ValueError):
        verify_intervals(intervals, 100)


def test_exact_integer_frame_budget():
    assert frame_budget(101) == 15
    assert verify_intervals([(0, 5), (5, 10), (70, 75)], 101) == 15
    assert frame_budget(6) == 0


def test_zip_range_member_crc(tmp_path):
    archive = tmp_path / "archive.zip"
    target = tmp_path / "range"
    payload = b"synthetic asset" * 200
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        output.writestr("asset.h5", payload)
    with ZipFile(archive) as output:
        target.write_bytes(archive.read_bytes()[:output.start_dir])
    dest = tmp_path / "extracted"
    read_range_member(target, "asset.h5", dest)
    assert dest.read_bytes() == payload
    with pytest.raises(ValueError):
        read_range_member(target, "other.h5", dest)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("source") / "source with audio.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x240:rate=30000/1001", "-f", "lavfi", "-i", "sine=frequency=440",
        "-frames:v", "100", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path),
    ], check=True)
    return path


def test_video_export_preserves_fractional_rate_and_source_order(synthetic_video, tmp_path):
    output = tmp_path / "synthetic summary.mp4"
    info = probe_video(synthetic_video)
    assert info.audio_streams == 1 and info.frame_count == 100
    intervals = [(3, 10), (30, 38)]
    assert export(synthetic_video, info, intervals, output) == 15
    actual = verify_video(synthetic_video, output, info, intervals)
    assert actual["no_audio"]
    assert actual["summary"]["frame_count"] == 15
    assert actual["summary"]["duration_seconds"] == float(15 / info.fps)
    assert actual["summary_ratio"] == .15
    assert actual["decoded_frame_matches"] == 15


def test_independent_decode_rejects_same_count_wrong_content(synthetic_video, tmp_path):
    output = tmp_path / "incorrect summary.mp4"
    info = probe_video(synthetic_video)
    export(synthetic_video, info, [(60, 75)], output)
    with pytest.raises(ValueError, match="expected source position"):
        verify_video(synthetic_video, output, info, [(0, 15)])
