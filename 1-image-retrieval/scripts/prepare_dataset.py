#!/usr/bin/env python3
"""Download/validate COIL-100 and save its deterministic split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_manifest, download_and_extract, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/coil-100"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/metrics/coil100_manifest.csv"),
    )
    args = parser.parse_args()

    dataset_dir = download_and_extract(args.data_dir, args.archive)
    records = build_manifest(dataset_dir, strict_dataset=True, validate_content=True)
    write_manifest(records, args.manifest)
    print(
        "Validated {} COIL-100 images: {} queries, {} gallery images.".format(
            len(records),
            sum(record.split == "query" for record in records),
            sum(record.split == "gallery" for record in records),
        )
    )
    print("Manifest: {}".format(args.manifest.resolve()))


if __name__ == "__main__":
    main()
