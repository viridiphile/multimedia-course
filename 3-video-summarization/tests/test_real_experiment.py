"""Official resource, checkpoint, and full TVSum video integration tests."""

import numpy as np
import pytest
import torch

from scripts.prepare_dataset import prepare
from scripts.run_experiment import run
from scripts.verify_results import verify
from src.inference import predict
from src.model import CHECKPOINT_NAME, load_model
from src.resources import FEATURE_NAME, LAB


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_inputs():
    if not (LAB / "data" / FEATURE_NAME).is_file() or not (LAB / "weights" / CHECKPOINT_NAME).is_file():
        pytest.skip("Run ./run-lab3.sh setup for the official assets")
    return prepare()


def test_original_and_feature_identity(real_inputs):
    source, video, source_info, record, features, _ = real_inputs
    assert source.is_file() and source_info["video_id"] == "EYqVtI9YWJA"
    assert video.frame_count == record.n_frames == 5939
    assert (video.width, video.height) == (540, 360)
    assert features["pick_intervals"] == [15]
    assert features["shape"] == [396, 1024]
    assert record.key == "video_42" and features["fold_role"] == "test"
    assert features["sampled_annotation_score_pearson_r"] > 0.9999


def test_pretrained_predictions_match_reference(real_inputs):
    _, _, _, record, _, checkpoint = real_inputs
    model = load_model(checkpoint)
    pred = predict(model, record.features)
    assert 0 < len(pred.kept_boxes) < len(record.picks)
    assert np.isfinite(pred.bin_scores).all()
    assert np.all(pred.offsets > 0)
    assert pred.confidence.max() > .99
    with torch.inference_mode():
        scores, boxes = model.predict(torch.from_numpy(record.features).unsqueeze(0))
    np.testing.assert_array_equal(pred.raw_boxes, boxes)
    np.testing.assert_array_equal(pred.confidence, scores)


def test_full_export_and_independent_verification(real_inputs, tmp_path):
    generated = run(tmp_path / "isolated integration run")
    verified = verify(tmp_path / "isolated integration run")
    assert generated["selected_segments"] == verified["selected_segments"]
    assert generated["summary"]["frame_count"] == verified["decoded_frame_matches"] == 890
    assert verified["summary_ratio"] <= .15
    assert verified["no_audio"] and verified["figures_checked"] == 4
