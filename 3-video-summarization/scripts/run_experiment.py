"""Summarize the complete selected TVSum video with the official anchor-free DSNet."""

import argparse
import csv
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import av
import imageio_ffmpeg
import torch

from scripts.prepare_dataset import prepare
from src.inference import NMS_THRESHOLD, predict
from src.model import ATTENTION_HEADS, FOLD, HIDDEN_DIM, REVISION, load_model
from src.resources import LAB, sha256
from src.segments import choose_shots
from src.validation import verify_video
from src.video_export import export
from src.visualization import draw_figures


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def exports(record, pred, scores, segments, fps):
    shot_rows = []
    selected = {segment.shot_index for segment in segments}
    for index, ((start, end), score) in enumerate(zip(record.shots, scores), 1):
        start, stop = int(start), int(end) + 1
        shot_rows.append(dict(shot_index=index, start_frame=start,
                              end_frame_exclusive=stop, start_seconds=float(start / fps),
                              end_seconds=float(stop / fps), frame_count=stop - start,
                              score=float(score), selected=index in selected))
    segment_rows, cursor = [], 0
    for index, segment in enumerate(segments, 1):
        first, stop = segment.start_frame, segment.end_frame_exclusive
        segment_rows.append(dict(segment_index=index, shot_index=segment.shot_index,
                                 start_frame=first, end_frame_exclusive=stop,
                                 start_seconds=float(first / fps), end_seconds=float(stop / fps),
                                 selected_frames=segment.frame_count, score=segment.score,
                                 summary_start_frame=cursor,
                                 summary_end_frame_exclusive=cursor + segment.frame_count))
        cursor += segment.frame_count
    prediction_rows = []
    for i, pick in enumerate(record.picks):
        frame = int(pick)
        raw_left, raw_right = map(float, pred.raw_boxes[i])
        left, right = map(int, pred.clipped_boxes[i])
        prediction_rows.append(dict(feature_index=i, source_frame=frame, source_seconds=float(frame / fps),
                                    class_probability=float(pred.class_scores[i]),
                                    centerness=float(pred.centerness[i]),
                                    confidence=float(pred.confidence[i]),
                                    left_offset=float(pred.offsets[i, 0]),
                                    right_offset=float(pred.offsets[i, 1]),
                                    raw_left=raw_left, raw_right_exclusive=raw_right,
                                    clipped_left=left, clipped_right_exclusive=right,
                                    post_nms_bin_score=float(pred.bin_scores[i])))
    edges = [*map(int, record.picks), record.n_frames]
    detections = [
        dict(rank=index, score=float(score), left_bin=int(left), right_bin_exclusive=int(right),
             start_source_frame=edges[left], end_source_frame_exclusive=edges[right])
        for index, (score, (left, right)) in enumerate(zip(pred.kept_scores, pred.kept_boxes), 1)
    ]
    return shot_rows, segment_rows, prediction_rows, detections


def run(output):
    source, video, source_info, record, feature_info, checkpoint = prepare()
    print(f"Predicting {record.key} · {source_info['title']} · {video.frame_count} decoded source frames",
          flush=True)
    pred = predict(load_model(checkpoint), record.features)
    scores, segments = choose_shots(record, pred.bin_scores)
    intervals = [(segment.start_frame, segment.end_frame_exclusive) for segment in segments]
    shots, selected, predictions, detections = exports(record, pred, scores, segments, video.fps)
    output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output) as work:
        stage = Path(work)
        summary = stage / "summary.mp4"
        export(source, video, intervals, summary)
        verification = verify_video(source, summary, video, intervals)
        metrics = dict(
            source=source_info, features=feature_info,
            model=dict(name="DSNet anchor-free basic attention", revision=REVISION, fold_index=FOLD,
                       checkpoint=f"pretrain_af_basic/checkpoint/{checkpoint.name}",
                       checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha256(checkpoint),
                       attention_heads=ATTENTION_HEADS, hidden_dim=HIDDEN_DIM,
                       nms_threshold=NMS_THRESHOLD, device="cpu", pytorch=torch.__version__),
            environment=dict(python=platform.python_version(), platform=platform.platform(),
                             pyav=av.__version__, ffmpeg=imageio_ffmpeg.get_ffmpeg_version()),
            nms_kept_detections=len(detections), selected_segments=len(selected),
            selection_adjustment="none",
            summary_sha256=sha256(summary), **verification,
        )
        draw_figures(stage / "figures", source, summary, metrics, selected, predictions, shots)
        metrics_dir = stage / "metrics"
        metrics_dir.mkdir()
        write_csv(metrics_dir / "selected_segments.csv", selected)
        write_csv(metrics_dir / "shot_scores.csv", shots)
        write_csv(metrics_dir / "predictions.csv", predictions)
        write_csv(metrics_dir / "nms_detections.csv", detections)
        write_json(metrics_dir / "selected_segments.json", selected)
        write_csv(metrics_dir / "run_summary.csv", [dict(
            source_id=source_info["video_id"], source_title=source_info["title"],
            source_filename=source_info["filename"], source_frames=video.frame_count,
            source_duration_seconds=video.duration_seconds, source_fps=str(video.fps),
            width=video.width, height=video.height, checkpoint=metrics["model"]["checkpoint"],
            feature_key=record.key, feature_count=len(record.features), feature_dim=record.features.shape[1],
            feature_pick_stride=feature_info["pick_intervals"][0],
            selected_segments=len(selected), frame_budget=verification["frame_budget"],
            selected_source_frames=verification["selected_source_frames"],
            summary_frames=verification["summary"]["frame_count"],
            summary_seconds=verification["summary"]["duration_seconds"],
            summary_ratio=verification["summary_ratio"], output_format="MP4/H.264",
            no_audio=verification["no_audio"], validation_passed=verification["validation_passed"],
        )])
        write_json(metrics_dir / "experiment_metrics.json", metrics)
        for item in stage.rglob("*"):
            if item.is_file():
                target = output / item.relative_to(stage)
                target.parent.mkdir(exist_ok=True)
                item.replace(target)
    print(f"Verified {metrics['summary']['frame_count']}/{video.frame_count} frames "
          f"({metrics['summary_ratio']:.2%}); {len(selected)} selected shots; no audio.", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LAB / "results")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
