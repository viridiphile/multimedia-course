#!/usr/bin/env python3
"""Extract official VGG16-GeM descriptors, retrieve Top-5, and evaluate P@5."""

from __future__ import annotations

import argparse
import platform
import json
import time
from datetime import datetime, timezone
import importlib.metadata

import numpy as np
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset import (build_manifest, read_manifest, write_manifest, dataset_fingerprint, OFFICIAL_DATASET_URL)
from src.evaluation import evaluate_precision_at_k, write_metrics, write_results_csv
from src.features import extract_features
from src.model import CHECKPOINT_NAME, load_retrieval_model, resolve_device
from src.report import write_report
from src.retrieval import rank_top_k
from src.visualization import (
    plot_case_analysis,
    plot_precision,
    plot_retrieval_examples,
)


def _record_map(records):
    return {
        record.image_id: {
            "image_id": record.image_id,
            "filename": record.filename,
            "path": record.path,
            "object_id": record.object_id,
            "angle": record.angle,
            "split": record.split,
        }
        for record in records
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/coil-100"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/metrics/coil100_manifest.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--weights", type=Path, default=Path("weights/retrievalSfM120k-vgg16-gem.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Optional smoke-run cap. Omit for the complete 200-query experiment.",
    )
    parser.add_argument("--smoke-objects", type=int, help="Limit extraction to first N objects; smoke runs only")
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.smoke_objects is not None and not 1 <= args.smoke_objects <= 100:
        parser.error("--smoke-objects must be in 1..100")
    if (args.max_queries or args.smoke_objects) and args.output_dir == Path("results"):
        parser.error("Smoke runs require a separate --output-dir (for example results/smoke)")
    np.random.seed(0)
    torch.manual_seed(0)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")

    # Always recheck the real inventory, hashes and decoded image dimensions.
    records = build_manifest(args.data_dir, strict_dataset=True, validate_content=True)
    if args.manifest.exists() and read_manifest(args.manifest) != records:
        raise ValueError("Prepared manifest does not match validated dataset; regenerate it")
    write_manifest(records, args.manifest)
    fingerprint = dataset_fingerprint(records)
    queries = [row for row in records if row.split == "query"]
    gallery = [row for row in records if row.split == "gallery"]
    all_ids = [row.image_id for row in records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Manifest contains duplicate image IDs")
    if len(queries) != 200 or len(gallery) != 7000:
        raise ValueError(
            "Expected a full COIL-100 manifest (200 queries, 7,000 gallery); "
            "found {} queries and {} gallery images".format(len(queries), len(gallery))
        )
    if args.smoke_objects is not None:
        queries = [r for r in queries if r.object_id <= args.smoke_objects]
        gallery = [r for r in gallery if r.object_id <= args.smoke_objects]
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if len(gallery) < 5:
        raise ValueError("At least five gallery images are required")

    device = resolve_device(args.device)
    model, meta, checkpoint_sha256 = load_retrieval_model(args.weights, device)
    input_dims = _check_image_dimensions(records)
    print(
        "Checkpoint: {}\nSource: {}\nLocal checkpoint: {}\n"
        "Input dimensions: {}\nDescriptor dimensions: {}\nDevice: {}".format(
            meta["checkpoint_name"],
            meta["checkpoint_url"],
            meta["checkpoint_path"],
            input_dims,
            meta["outputdim"],
            device,
        )
    )

    inference_started = time.perf_counter()
    query_ids, query_features = extract_features(
        queries, model, meta, device, args.batch_size
    )
    gallery_ids, gallery_features = extract_features(
        gallery, model, meta, device, args.batch_size
    )
    if query_ids != [row.image_id for row in queries] or gallery_ids != [
        row.image_id for row in gallery
    ]:
        raise RuntimeError("Feature descriptors do not align with their image records")

    inference_seconds = time.perf_counter() - inference_started
    results = rank_top_k(queries, gallery, query_features, gallery_features, k=5)
    metrics = evaluate_precision_at_k(results, k=5)
    metrics.update(
        {
            "gallery_images": len(gallery),
            "mean_precision_at_5": float(metrics["mean_precision_at_5"]),
            "checkpoint_name": CHECKPOINT_NAME,
        }
    )

    output = args.output_dir
    results_path = output / "metrics/top5_results.csv"
    precision_path = output / "metrics/per_query_precision.csv"
    metrics_path = output / "metrics/experiment_metrics.json"
    write_results_csv(results, results_path)

    config: Dict = {
        "checkpoint_name": meta["checkpoint_name"],
        "checkpoint_url": meta["checkpoint_url"],
        "checkpoint_path": meta["checkpoint_path"],
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "batch_size": args.batch_size,
        "input_dimensions": input_dims,
        "descriptor_dimensions": meta["outputdim"],
        "mean": meta["mean"],
        "std": meta["std"],
        "architecture": meta["architecture"],
        "pooling": meta["pooling"],
        "gallery_count": len(gallery),
        "query_count": len(queries),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "is_smoke_run": args.max_queries is not None or args.smoke_objects is not None,
        "gem_p": meta["gem_p"],
        "upstream_revision": meta["upstream_revision"],
        "dataset_url": OFFICIAL_DATASET_URL,
        "valid_dataset_images": len(records),
        "dataset_fingerprint_sha256": fingerprint,
        "query_angles": [0, 180],
        "relevance": "same COIL-100 object_id",
        "seed": 0,
        "split_randomized": False,
        "started_at_utc": started_at,
        "command": [sys.executable] + sys.argv,
        "inference_seconds": inference_seconds,
        "gem_epsilon": 1e-6,
        "preprocessing": "native 128x128 RGB; ToTensor; checkpoint mean/std; no augmentation",
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "torchvision", "numpy", "Pillow", "matplotlib", "scipy", "pytest", "tqdm")},
    }
    output.mkdir(parents=True, exist_ok=True)
    # Local descriptors enable an independent exact ranking check; ignored by Git.
    np.savez_compressed(output / "descriptors.npz", query_ids=query_ids,
                        gallery_ids=gallery_ids, query_features=query_features,
                        gallery_features=gallery_features)
    metrics["configuration"] = config
    write_metrics(metrics, precision_path, metrics_path)

    records_by_id = _record_map(records)
    figure_dir = output / "figures"
    plot_retrieval_examples(
        records_by_id, results, figure_dir / "retrieval_examples.png", count=10
    )
    plot_precision(metrics, figure_dir / "precision_at_5.png")
    precision_by_query = {
        row["query_id"]: row["precision_at_5"]
        for row in metrics["precision_at_5_per_query"]
    }
    case = plot_case_analysis(
        records_by_id,
        results,
        precision_by_query,
        figure_dir / "case_analysis.png",
    )
    (output / "metrics/case_analysis.json").write_text(json.dumps(case, indent=2) + "\n")
    write_report(
        metrics,
        config,
        case,
        results,
        output / "report/lab1_report.md",
    )
    config["elapsed_seconds"] = time.perf_counter() - started
    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_metrics(metrics, precision_path, metrics_path)
    print(
        "Evaluated {} queries against {} gallery images. Mean P@5: {:.4f}".format(
            metrics["queries_evaluated"],
            len(gallery),
            metrics["mean_precision_at_5"],
        )
    )
    print("Results written under {}".format(output.resolve()))


def _check_image_dimensions(records) -> str:
    from PIL import Image

    dimensions = set()
    for record in records:
        try:
            with Image.open(record.path) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ValueError("Unsupported image format in {}".format(record.path))
                image.verify()
                dimensions.add(image.size)
        except OSError as exc:
            raise ValueError("Invalid image {}: {}".format(record.path, exc)) from exc
    if dimensions != {(128, 128)}:
        raise ValueError("COIL-100 images must all be 128x128; found {}".format(dimensions))
    return "128x128 RGB"


if __name__ == "__main__":
    main()
