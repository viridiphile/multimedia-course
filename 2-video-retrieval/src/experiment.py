"""Run TVSum clip-to-source retrieval and write auditable evidence."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import cv2
import numpy as np

from . import dataset, resources
from .evaluation import hit_metrics, rank


@dataclass
class Descriptor:
    path: Path
    metadata: dict
    features: np.ndarray
    times: np.ndarray
    decoded_frames: int


def describe(model, path, metadata, cache):
    key = hashlib.sha256(
        f"{metadata['sha256']}:{model.cache_key}:{dataset.sha256(Path(dataset.__file__))}".encode()
    ).hexdigest()
    output = cache / f"{key}.npz"
    if not output.is_file():
        print(f"Extracting ViSiL features: {path.name}", flush=True)
        frames, times, decoded = dataset.sample_frames(path, metadata["fps"], metadata["frame_count"])
        features = model.extract_features(frames)
        cache.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as file:
            temporary = Path(file.name)
        try:
            with temporary.open("wb") as file:
                np.savez(file, features=features, times=times, decoded_frames=decoded)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(output, allow_pickle=False) as saved:
        features, times, decoded = saved["features"], saved["times"], int(saved["decoded_frames"])
    if features.shape != (len(times), 9, 3840) or not np.isfinite(features).all():
        raise ValueError(f"Invalid cached regional features in {output}")
    return Descriptor(path, metadata, features, times, decoded)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_display_frames(stream, indices):
    """Decode only the representative frames requested for the figure."""
    cap = dataset.open_video(stream.path)
    cadence = round(stream.metadata["fps"])
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index) * cadence)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Cannot render frame {index} of {stream.path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            scale = 224 / max(height, width)
            thumb = cv2.resize(rgb, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
            canvas = np.full((224, 224, 3), 25, np.uint8)
            top, left = (224 - thumb.shape[0]) // 2, (224 - thumb.shape[1]) // 2
            canvas[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb
            frames.append(canvas)
    finally:
        cap.release()
    return np.stack(frames)


def render(model, out, queries, gallery, rankings):
    from .visualization import heatmap, retrieval_examples

    top = {query["query_id"]: [row for row in rankings
           if row["query_id"] == query["query_id"] and row["rank"] <= 3] for query, _ in queries}
    # Use the strongest true-source score and the weakest true-source rank, fixed by the exported rankings.
    relevant = {row["query_id"]: row for row in rankings if row["relevant"]}
    weakest = max(queries, key=lambda item: (
        relevant[item[0]["query_id"]]["rank"], -relevant[item[0]["query_id"]]["similarity_score"]))
    strongest = max((item for item in queries if item[0] != weakest[0]), default=weakest,
                    key=lambda item: relevant[item[0]["query_id"]]["similarity_score"])
    cases = []
    for query, stream in (strongest, weakest):
        rows = top[query["query_id"]]
        candidates = []
        indices = [len(stream.times) // 3, 2 * len(stream.times) // 3]
        for row in rows:
            candidate = gallery[row["candidate_id"]]
            frames, _, _ = model.compare(stream.features, candidate.features)
            match = frames[indices].argmax(axis=1)
            candidates.append((row, read_display_frames(candidate, match), candidate.times[match]))
        cases.append((query, read_display_frames(stream, indices), stream.times[indices], candidates))
    retrieval_examples(out / "figures" / "top3_examples.png", cases)

    query, stream = strongest
    row = relevant[query["query_id"]]
    candidate = gallery[row["candidate_id"]]
    frame_matrix, temporal, score = model.compare(stream.features, candidate.features)
    pair = dict(query_id=query["query_id"], candidate_id=row["candidate_id"],
                candidate_rank=row["rank"], source_start_seconds=query["start_seconds"],
                source_clip_seconds=query["duration_seconds"], similarity_score=score,
                query_times=stream.times.tolist(), candidate_times=candidate.times.tolist(),
                frame_similarity=frame_matrix.tolist(), temporal_similarity=temporal.tolist())
    write_json(out / "metrics" / "heatmap_pair.json", pair)
    heatmap(out / "figures" / "frame_similarity.png", pair)


def run(profile, out):
    from .model import ViSiL

    start = time.perf_counter()
    tvsum = resources.prepare_dataset()
    checkpoint = resources.prepare_checkpoint()
    manifest = dataset.inventory(tvsum)
    protocol = dataset.select_protocol(manifest, profile)
    sources = {video.video_id: video for video in manifest}
    cache = resources.LAB / ".cache" / "descriptors"
    gallery, queries, rankings = {}, [], []
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    with ViSiL(checkpoint) as model:
        for video_id in protocol["gallery_ids"]:
            meta = vars(sources[video_id])
            gallery[video_id] = describe(model, tvsum / meta["file"], meta, cache)
        for query in protocol["queries"]:
            path, meta = dataset.make_query(tvsum, sources[query["source_id"]], query,
                                            resources.LAB / "data" / "queries")
            meta["sha256"] = dataset.sha256(path)
            query["file"] = path.relative_to(resources.LAB).as_posix()
            query["clip_metadata"] = meta
            stream = describe(model, path, meta, cache)
            queries.append((query, stream))
            scores = {key: model.compare(stream.features, item.features)[2]
                      for key, item in gallery.items()}
            rankings.extend(rank(query, scores))
            truth = next(row for row in rankings if row["query_id"] == query["query_id"] and row["relevant"])
            print(f"{query['query_id']} · source rank {truth['rank']} · {truth['similarity_score']:.5f}",
                  flush=True)
        render(model, out, queries, gallery, rankings)
        model_info = model.provenance

    metrics = dict(**hit_metrics(rankings), profile=profile, gallery_count=len(gallery),
                   total_source_videos=len(manifest), dataset_bytes=sum(v.size_bytes for v in manifest),
                   source_duration_seconds=sum(v.duration_seconds for v in manifest),
                   scored_pairs=len(rankings), device="cpu", python=platform.python_version(),
                   platform=platform.platform(), elapsed_seconds=round(time.perf_counter() - start, 3),
                   model=model_info)
    protocol["gallery_decode"] = {
        key: dict(decoded_frames=stream.decoded_frames, sampled_frames=len(stream.times))
        for key, stream in gallery.items()
    }
    for query, stream in queries:
        query["decoded_frames"] = stream.decoded_frames
        query["sampled_frames"] = len(stream.times)
    write_csv(out / "metrics" / "tvsum_manifest.csv", dataset.manifest_rows(manifest))
    write_csv(out / "metrics" / "all_rankings.csv", rankings)
    write_csv(out / "metrics" / "top3.csv", [row for row in rankings if row["rank"] <= 3])
    write_json(out / "metrics" / "protocol.json", protocol)
    write_json(out / "metrics" / "experiment_metrics.json", metrics)
    print(f"Top-3 Hit Rate: {metrics['successful_queries']}/{metrics['total_queries']} "
          f"({metrics['top3_hit_rate']:.1%})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("smoke", "full"))
    parser.add_argument("--output", type=Path, help="Override the profile's default result directory")
    args = parser.parse_args()
    out = args.output or resources.LAB / "results"
    if args.output is None and args.profile == "smoke":
        out = out / "smoke"
    run(args.profile, out)


if __name__ == "__main__":
    main()
