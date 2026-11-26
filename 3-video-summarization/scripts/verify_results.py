"""Validate the saved experiment by decoding source and summary independently."""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image

from src.dataset import probe_video
from src.resources import LAB, prepare_tvsum, sha256
from src.validation import verify_video


def read_csv(directory, name):
    with (directory / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def verify(output):
    directory = output / "metrics"
    expected = json.loads((directory / "experiment_metrics.json").read_text(encoding="utf-8"))
    rows = read_csv(directory, "selected_segments.csv")
    segment_objects = json.loads((directory / "selected_segments.json").read_text(encoding="utf-8"))
    if [{key: str(value) for key, value in item.items()} for item in segment_objects] != rows:
        raise ValueError("Selected segments differ between CSV and JSON")
    source, _ = prepare_tvsum(expected["source"]["video_id"])
    video = probe_video(source)
    verify_path = output / "summary.mp4"
    if sha256(source) != expected["source"]["sha256"] or sha256(verify_path) != expected["summary_sha256"]:
        raise ValueError("Source or summary SHA-256 differs from the recorded run")

    intervals = [(int(row["start_frame"]), int(row["end_frame_exclusive"])) for row in rows]
    actual = verify_video(source, verify_path, video, intervals)
    if len(rows) != expected["selected_segments"] or any(expected[key] != value for key, value in actual.items()):
        raise ValueError("Measured output disagrees with saved summary metrics")
    if video.frame_count != expected["source"]["frame_count"]:
        raise ValueError("Saved source frame count differs from the decode")

    summary_csv = read_csv(directory, "run_summary.csv")
    if (len(summary_csv) != 1 or int(summary_csv[0]["summary_frames"]) != actual["summary"]["frame_count"]
            or float(summary_csv[0]["summary_ratio"]) != actual["summary_ratio"]
            or summary_csv[0]["no_audio"] != str(actual["no_audio"])):
        raise ValueError("Run summary CSV disagrees with independent decode")

    shots = read_csv(directory, "shot_scores.csv")
    predictions = read_csv(directory, "predictions.csv")
    boxes = read_csv(directory, "nms_detections.csv")
    if (len(shots) != expected["features"]["hdf5_shots"] or
            len(predictions) != expected["features"]["shape"][0] or
            len(boxes) != expected["nms_kept_detections"]):
        raise ValueError("The exported probability, NMS or shot tables are incomplete")
    selected_shots = [row for row in shots if row["selected"] == "True"]
    if len(selected_shots) != len(rows):
        raise ValueError("Shot and selection tables differ")
    cursor = 0
    for index, (row, shot, (start, stop)) in enumerate(zip(rows, selected_shots, intervals), 1):
        if (int(row["segment_index"]) != index or int(row["selected_frames"]) != stop - start
                or int(row["summary_start_frame"]) != cursor
                or int(row["summary_end_frame_exclusive"]) != cursor + stop - start
                or not math.isclose(float(row["start_seconds"]), float(start / video.fps), abs_tol=1e-9)
                or not math.isclose(float(row["end_seconds"]), float(stop / video.fps), abs_tol=1e-9)
                or any(row[key] != shot[key] for key in
                       ("shot_index", "start_frame", "end_frame_exclusive", "score"))):
            raise ValueError(f"Exported segment {index} does not match its shot and time mapping")
        cursor += stop - start
    for name in ("selected_timeline", "summary_frames", "original_vs_summary", "dsnet_scores"):
        with Image.open(output / "figures" / f"{name}.png") as image:
            if image.format != "PNG":
                raise ValueError(f"Invalid figure {name}")
            image.verify()
    return {**actual, "selected_segments": len(rows), "predictions_checked": len(predictions),
            "nms_detections_checked": len(boxes), "figures_checked": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LAB / "results")
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
