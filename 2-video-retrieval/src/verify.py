"""Check a saved Lab 2 run without invoking the retrieval pipeline."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from PIL import Image


def load_json(metrics, name):
    return json.loads((metrics / name).read_text(encoding="utf-8"))


def load_csv(metrics, name):
    with (metrics / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(out):
    metrics = out / "metrics"
    summary = load_json(metrics, "experiment_metrics.json")
    protocol = load_json(metrics, "protocol.json")
    manifest = load_csv(metrics, "tvsum_manifest.csv")
    rows = load_csv(metrics, "all_rankings.csv")
    top = load_csv(metrics, "top3.csv")
    pair = load_json(metrics, "heatmap_pair.json")

    ids = {item["video_id"]: item for item in manifest}
    require(len(ids) == len(manifest) == summary["total_source_videos"] == 50, "Wrong inventory size")
    categories = Counter(item["category"] for item in manifest)
    require(len(categories) == 10 and set(categories.values()) == {5}, "Invalid TVSum categories")
    require(sum(int(item["size_bytes"]) for item in manifest) == summary["dataset_bytes"],
            "Video bytes mismatch")
    for item in manifest:
        require(all(float(item[key]) > 0 for key in
                    ("width", "height", "fps", "frame_count", "duration_seconds", "size_bytes")),
                "Invalid video metadata")

    gallery = protocol["gallery_ids"]
    queries = protocol["queries"]
    require(protocol["profile"] == summary["profile"] in {"smoke", "full"}, "Unknown profile")
    categories = sorted({item["category"] for item in manifest})
    categories = categories[:2] if summary["profile"] == "smoke" else categories
    selection = [sorted(key for key, item in ids.items() if item["category"] == category)[:2]
                 for category in categories]
    require(gallery == [item for pair_ids in selection for item in pair_ids], "Wrong gallery")
    require([query["source_id"] for query in queries] == [group[0] for group in selection],
            "Wrong query selection")
    require(len(gallery) == summary["gallery_count"], "Wrong gallery count")
    require(len(rows) == len(queries) * len(gallery) == summary["scored_pairs"], "Pair count mismatch")
    require(top == [row for row in rows if int(row["rank"]) <= 3], "Top-3 CSV differs from all ranks")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    require(set(grouped) == {item["query_id"] for item in queries}, "Unexpected query IDs")

    successes = 0
    for index, query in enumerate(queries):
        key = query["query_id"]
        source = query["source_id"]
        duration = float(ids[source]["duration_seconds"])
        expected_start = round((0.25, 0.45, 0.65)[index % 3] * (duration - 16))
        require(query["duration_seconds"] == 16 and query["start_seconds"] == expected_start
                and 0 < expected_start < expected_start + 16 < duration,
                f"Invalid clip interval for {key}")
        require(query["relevant_ids"] == [source], f"Wrong relevance ground truth for {key}")
        require(query["clip_metadata"]["sha256"] != ids[source]["sha256"], f"Identical query {key}")
        group = grouped[key]
        require(Counter(row["candidate_id"] for row in group) == Counter(gallery),
                f"Candidate coverage mismatch for {key}")
        require([int(row["rank"]) for row in group] == list(range(1, len(gallery)+1)),
                f"Rank positions invalid for {key}")
        require(all(math.isfinite(float(row["similarity_score"])) and
                    abs(float(row["similarity_score"])) <= 1 for row in group),
                f"Out-of-range scores for {key}")
        require(group == sorted(group, key=lambda row: (-float(row["similarity_score"]),
                                                        row["candidate_id"])),
                f"Score ordering mismatch for {key}")
        require(all(row["source_id"] == source and
                    row["relevant"] == str(row["candidate_id"] == source) for row in group),
                f"Relevance classification mismatch for {key}")
        hit = any(row["candidate_id"] == source for row in group[:3])
        require(all(row["top3_success"] == str(hit) for row in group), f"Wrong hit flag for {key}")
        successes += hit

    require(successes == summary["successful_queries"] and len(queries) == summary["total_queries"]
            and successes / len(queries) == summary["top3_hit_rate"], "Hit-rate mismatch")
    pair_row = next(row for row in grouped[pair["query_id"]]
                    if row["candidate_id"] == pair["candidate_id"])
    require(pair_row["relevant"] == "True" and int(pair_row["rank"]) == pair["candidate_rank"],
            "Unexpected heatmap pair")
    query_count, candidate_count = len(pair["query_times"]), len(pair["candidate_times"])
    for key, divisor in [("frame_similarity", 1), ("temporal_similarity", 4)]:
        matrix = pair[key]
        require(len(matrix) == query_count // divisor and
                all(len(row) == candidate_count // divisor for row in matrix),
                f"Unexpected {key} dimensions")
        require(all(math.isfinite(score) and -1 <= score <= 1 for row in matrix for score in row),
                f"Invalid {key} values")
    temporal = pair["temporal_similarity"]
    reconstructed = math.fsum(map(max, temporal)) / len(temporal)
    require(all(math.isclose(reconstructed, score, abs_tol=1e-6) for score in
                (pair["similarity_score"], float(pair_row["similarity_score"]))),
            "Heatmap does not reproduce the ranked similarity")
    for name in ("top3_examples.png", "frame_similarity.png"):
        with Image.open(out / "figures" / name) as image:
            require(image.format == "PNG" and image.width >= 1000 and image.height >= 500, name)
            image.verify()
    return dict(scored_pairs=len(rows), top3_rows=len(top), successful_queries=successes,
                total_queries=len(queries), top3_hit_rate=successes / len(queries),
                heatmap_score_from_exported_values=reconstructed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
