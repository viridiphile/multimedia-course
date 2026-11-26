"""Synthetic tests for video preparation, ranking and plotting."""

from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image
import pytest

from src import dataset
from src.evaluation import hit_metrics, rank
from src.visualization import heatmap, retrieval_examples


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic source.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x240:rate=12", "-t", "40", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    return path


def test_decode_and_probe(video):
    meta = dataset.probe(video)
    assert meta == dict(width=320, height=240, fps=12, frame_count=480, duration_seconds=40)
    frames, times, decoded = dataset.sample_frames(video, meta["fps"], meta["frame_count"])
    assert frames.shape == (40, 240, 320, 3)
    np.testing.assert_array_equal(times, np.arange(40))
    assert decoded == 480
    assert not np.array_equal(frames[0], frames[-1])


def test_probe_rejects_corrupt_video(tmp_path):
    video = tmp_path / "invalid.mp4"
    video.write_bytes(b"invalid container")
    with pytest.raises(ValueError, match="Cannot open"):
        dataset.probe(video)


def test_query_is_a_processed_subset(video, tmp_path):
    meta = dataset.probe(video)
    source = dataset.Video("source", "category", "Title", "", video.name, **meta,
                           size_bytes=video.stat().st_size, sha256=dataset.sha256(video))
    query = dict(query_id="source__clip", start_seconds=5, duration_seconds=16)
    path, clip_meta = dataset.make_query(video.parent, source, query, tmp_path / "test clips")
    assert path != video
    assert clip_meta["duration_seconds"] == 16
    assert clip_meta["width"] < meta["width"]
    assert clip_meta["height"] == 240
    assert dataset.sha256(path) != source.sha256
    assert dataset.make_query(video.parent, source, query, path.parent) == (path, clip_meta)


def test_protocol_is_balanced_and_lexically_ordered():
    from types import SimpleNamespace
    videos = [SimpleNamespace(video_id=f"{category}{index}", category=category, duration_seconds=60)
              for category in "CBA" for index in (3, 2, 1)]
    protocol = dataset.select_protocol(videos, "full")
    assert protocol["gallery_ids"] == ["A1", "A2", "B1", "B2", "C1", "C2"]
    assert [query["relevant_ids"] for query in protocol["queries"]] == [["A1"], ["B1"], ["C1"]]
    assert [query["start_seconds"] for query in protocol["queries"]] == [11, 20, 29]
    assert len(dataset.select_protocol(videos, "smoke")["gallery_ids"]) == 4


def test_ranking_relevance_and_hit_rate():
    scores = {"c": 0.5, "b": 0.6, "a": 0.6, "d": -0.1}
    hit = rank(dict(query_id="first", source_id="c", relevant_ids=["c"]), scores)
    miss = rank(dict(query_id="second", source_id="d", relevant_ids=["d"]), scores)
    assert [row["candidate_id"] for row in hit] == ["a", "b", "c", "d"]
    assert [row["relevant"] for row in hit] == [False, False, True, False]
    assert all(row["top3_success"] for row in hit)
    assert all(not row["top3_success"] for row in miss)
    assert hit_metrics(hit + miss) == dict(successful_queries=1, total_queries=2, top3_hit_rate=.5)


@pytest.mark.parametrize("scores", [{"a": 0, "b": 0, "c": 0}, {"a": 0, "b": 0, "d": float("nan")}])
def test_ranking_rejects_missing_ground_truth_or_nonfinite_score(scores):
    with pytest.raises(ValueError):
        rank(dict(query_id="test", source_id="d", relevant_ids=["d"]), scores)


def test_figures_can_be_rendered(tmp_path):
    frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    times = np.array([0, 1])
    ranking = rank(dict(query_id="q", source_id="a", relevant_ids=["a"]), {"a": .9, "b": .4, "c": .1})
    cases = [(dict(source_id="a", start_seconds=10), frames, times,
              [(row, frames, times) for row in ranking])]
    retrieval_examples(tmp_path / "top3.png", cases)
    heatmap(tmp_path / "heatmap.png", dict(query_id="q", candidate_id="a", candidate_rank=1,
            query_times=list(range(8)), candidate_times=list(range(12)), source_start_seconds=3,
            source_clip_seconds=8, similarity_score=.8, frame_similarity=np.ones((8, 12)) * .4,
            temporal_similarity=[[.8, .3, .2], [.3, .8, .2]]))
    for path in sorted(Path(tmp_path).glob("*.png")):
        with Image.open(path) as image:
            assert image.format == "PNG"
            image.verify()
