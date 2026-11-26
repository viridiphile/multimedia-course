"""Tests requiring the original TVSum release and authors' pretrained ViSiL."""

import numpy as np
import pytest

from src import dataset, resources
from src.verify import verify


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tvsum():
    directory = resources.LAB / "data" / "tvsum"
    if not (directory / "info.tsv").exists():
        pytest.skip("Prepare official TVSum with ./run-lab2.sh setup")
    return directory


@pytest.fixture(scope="module")
def pretrained():
    from src.model import ViSiL

    checkpoint = resources.LAB / "weights" / "resnet"
    if not all((checkpoint / name).is_file() for name in resources.TF_FILES):
        pytest.skip("Prepare the pretrained ViSiL checkpoint with ./run-lab2.sh setup")
    with ViSiL(checkpoint) as model:
        yield model


@pytest.fixture(scope="module")
def manifest(tvsum):
    return dataset.inventory(tvsum)


def test_real_inventory(manifest, tvsum):
    assert len(manifest) == 50
    assert len({video.category for video in manifest}) == 10
    assert all(0 < video.duration_seconds and video.width > 0 and video.height > 0 for video in manifest)
    assert sum(video.size_bytes for video in manifest) > 600_000_000
    assert {item["source_id"] for item in dataset.select_protocol(manifest, "full")["queries"]}.issubset(
        {video.video_id for video in manifest})
    assert all((tvsum / video.file).is_file() for video in manifest)


def test_model_pretrained_real_clip(pretrained, manifest, tvsum, tmp_path):
    query = dataset.select_protocol(manifest, "smoke")["queries"][0]
    source = next(video for video in manifest if video.video_id == query["source_id"])
    clip, meta = dataset.make_query(tvsum, source, query, tmp_path)
    frames, _, _ = dataset.sample_frames(clip, meta["fps"], meta["frame_count"])
    descriptors = pretrained.extract_features(frames[:8])
    assert descriptors.shape == (8, 9, 3840)
    assert np.isfinite(descriptors).all()
    assert np.std(descriptors) > .0001
    assert np.linalg.norm(descriptors[0] - descriptors[-1]) > .01
    raw, temporal, score = pretrained.compare(descriptors, descriptors)
    assert raw.shape == (8, 8)
    assert temporal.shape == (2, 2)
    np.testing.assert_allclose(score, temporal.max(axis=1).mean(), atol=1e-6)
    # Independent regional Chamfer calculation for one pair of frames.
    regional = descriptors[0] @ descriptors[-1].T
    np.testing.assert_allclose(raw[0, -1], regional.max(axis=1).mean(), atol=1e-6)


@pytest.mark.parametrize("profile, expected", [("smoke", 2), ("full", 10)])
def test_saved_outputs(profile, expected):
    results = resources.LAB / "results"
    if profile == "smoke":
        results = results / "smoke"
    if not (results / "metrics" / "experiment_metrics.json").is_file():
        pytest.skip(f"Run ./run-lab2.sh {profile} to produce integration artifacts")
    verification = verify(results)
    assert verification["total_queries"] == expected
    assert verification["scored_pairs"] == expected * expected * 2
