"""COIL-100 discovery, validation, and deterministic query/gallery split."""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


OFFICIAL_DATASET_URL = (
    "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/"
    "coil-100/coil-100.zip"
)
FILENAME_PATTERN = re.compile(
    r"^obj(?P<object_id>\d+)__(?P<angle>\d+)\.(?P<extension>png|jpe?g)$",
    re.IGNORECASE,
)
EXPECTED_OBJECT_IDS = set(range(1, 101))
EXPECTED_ANGLES = set(range(0, 360, 5))
QUERY_ANGLES = {0, 180}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str
    path: str
    object_id: int
    angle: int
    split: str


def parse_filename(filename: str) -> Tuple[int, int]:
    """Return (object ID, angle) from a standard COIL-100 filename."""
    match = FILENAME_PATTERN.fullmatch(Path(filename).name)
    if not match:
        raise ValueError(
            "Invalid COIL-100 filename {!r}; expected obj<ID>__<angle>.png".format(
                filename
            )
        )
    object_id, angle = int(match.group("object_id")), int(match.group("angle"))
    if object_id not in EXPECTED_OBJECT_IDS:
        raise ValueError("Object ID {} is outside the COIL-100 range 1..100".format(object_id))
    if angle not in EXPECTED_ANGLES:
        raise ValueError(
            "Viewing angle {} is not a COIL-100 five-degree angle".format(angle)
        )
    return object_id, angle


def _image_files(data_dir: Path) -> List[Path]:
    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and FILENAME_PATTERN.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(
            "No COIL-100 images found under {}. Run prepare_dataset.py first.".format(
                data_dir
            )
        )
    return files


def _validate_content(files: Sequence[Path]) -> None:
    seen: dict = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(
                "Duplicate image content found in {} and {}".format(seen[digest], path)
            )
        seen[digest] = path


def build_manifest(
    data_dir: Path,
    strict_dataset: bool = True,
    validate_content: bool = True,
) -> List[ImageRecord]:
    """Build split records and verify the standard dataset protocol."""
    data_dir = Path(data_dir).expanduser().resolve()
    files = _image_files(data_dir)
    records: List[ImageRecord] = []
    found = set()

    for path in files:
        object_id, angle = parse_filename(path.name)
        key = (object_id, angle)
        if key in found:
            raise ValueError(
                "Duplicate object/view entry for object {} at {} degrees".format(
                    object_id, angle
                )
            )
        found.add(key)
        records.append(
            ImageRecord(
                image_id=path.stem,
                filename=path.name,
                path=str(path),
                object_id=object_id,
                angle=angle,
                split="query" if angle in QUERY_ANGLES else "gallery",
            )
        )

    if validate_content:
        _validate_content(files)

    if strict_dataset:
        expected = {
            (object_id, angle)
            for object_id in EXPECTED_OBJECT_IDS
            for angle in EXPECTED_ANGLES
        }
        missing = expected - found
        unexpected = found - expected
        queries = [record for record in records if record.split == "query"]
        gallery = [record for record in records if record.split == "gallery"]
        if missing or unexpected:
            raise ValueError(
                "COIL-100 view inventory mismatch: {} missing, {} unexpected".format(
                    len(missing), len(unexpected)
                )
            )
        if len(records) != 7200 or len(queries) != 200 or len(gallery) != 7000:
            raise ValueError(
                "Expected 7,200 images (200 queries, 7,000 gallery); got "
                "{} ({} queries, {} gallery)".format(
                    len(records), len(queries), len(gallery)
                )
            )
        _validate_dimensions(files)

    query_ids = {record.image_id for record in records if record.split == "query"}
    gallery_ids = {record.image_id for record in records if record.split == "gallery"}
    if query_ids & gallery_ids:
        raise ValueError("Query/gallery overlap detected")
    return sorted(records, key=lambda row: (row.object_id, row.angle))


def _validate_dimensions(files: Sequence[Path]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate COIL-100 images") from exc

    for path in files:
        try:
            with Image.open(path) as image:
                if image.size != (128, 128):
                    raise ValueError(
                        "Expected 128x128 image at {}, found {}".format(
                            path, image.size
                        )
                    )
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError("Invalid image file: {}".format(path)) from exc


def write_manifest(records: Iterable[ImageRecord], output_path: Path) -> None:
    records = list(records)
    if not records:
        raise ValueError("Cannot write an empty manifest")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            row = asdict(record)
            import os
            row["path"] = os.path.relpath(record.path, output_path.resolve().parent)
            writer.writerow(row)


def read_manifest(manifest_path: Path) -> List[ImageRecord]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty: {}".format(manifest_path))
    return [
        ImageRecord(
            image_id=row["image_id"],
            filename=row["filename"],
            path=str((Path(manifest_path).resolve().parent / row["path"]).resolve()),
            object_id=int(row["object_id"]),
            angle=int(row["angle"]),
            split=row["split"],
        )
        for row in rows
    ]


def download_and_extract(data_dir: Path, archive_path: Optional[Path] = None) -> Path:
    """Download the official Columbia ZIP, or validate an explicit local archive."""
    data_dir = Path(data_dir).expanduser().resolve()
    if _has_images(data_dir):
        return data_dir
    if archive_path is None:
        archive_path = data_dir.parent / "downloads" / "coil-100.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not archive_path.is_file():
            temporary = archive_path.with_suffix(".zip.part")
            try:
                with urllib.request.urlopen(OFFICIAL_DATASET_URL, timeout=120) as response:
                    with temporary.open("wb") as handle:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                temporary.replace(archive_path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
    return _extract_archive(data_dir, archive_path)


def _extract_archive(data_dir: Path, archive_path: Path) -> Path:
    """Extract a local COIL-100 zip, rejecting unsafe archive paths."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError("COIL-100 archive not found: {}".format(archive))

    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("Unsafe path in dataset archive: {}".format(member.filename))
        bundle.extractall(root)

    if not _has_images(data_dir):
        raise ValueError("Archive contained no COIL-100 images: {}".format(archive))
    return data_dir


def _has_images(directory: Path) -> bool:
    return directory.exists() and any(
        path.is_file() and FILENAME_PATTERN.fullmatch(path.name)
        for path in directory.rglob("*")
    )


def dataset_fingerprint(records: Sequence[ImageRecord]) -> str:
    """Hash relative identifiers and image bytes in protocol order."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.filename.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(Path(record.path).read_bytes()).hexdigest()))
    return digest.hexdigest()
