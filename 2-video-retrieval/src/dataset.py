"""Inventory TVSum originals, create separate query clips, and sample video frames."""

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np


CLIP_SECONDS = 16
CROP_FILTER = "crop=trunc(iw*0.92/2)*2:trunc(ih/2)*2:iw-ow:0,scale=-2:240,setsar=1"
QUERY_TRANSFORM = "Remove left ~8% of frame, scale to 240 pixels high, H.264 CRF 26, mute"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Video:
    video_id: str
    category: str
    title: str
    url: str
    file: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    size_bytes: int
    sha256: str


def open_video(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")
    return cap


def probe(path):
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or not math.isfinite(fps) or fps < 1 or count < 1:
            raise ValueError(f"Invalid video metadata or first frame: {path}")
        height, width = frame.shape[:2]
        return dict(width=width, height=height, fps=fps, frame_count=count,
                    duration_seconds=count / fps)
    finally:
        cap.release()


def inventory(dataset):
    with (dataset / "info.tsv").open(encoding="utf-8-sig", newline="") as file:
        entries = list(csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE))
    ids = [row["video_id"] for row in entries]
    if (len(ids) != 50 or len(set(ids)) != 50
            or any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", key) for key in ids)):
        raise ValueError("Expected the 50 unique source IDs from TVSum 1.1")
    files = {}
    for path in (dataset / "videos").iterdir():
        if path.suffix.lower() in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
            if path.stem in files:
                raise ValueError(f"Duplicate TVSum file ID: {path.stem}")
            files[path.stem] = path
    if set(ids) != set(files):
        raise ValueError(
            f"TVSum files differ from info.tsv. Missing: {sorted(set(ids)-files.keys())}; "
            f"unexpected: {sorted(files.keys()-set(ids))}"
        )
    manifest = []
    for row in entries:
        key = row["video_id"]
        path = files[key]
        manifest.append(Video(key, row["category"], row["title"], row["url"],
                              path.relative_to(dataset).as_posix(), **probe(path),
                              size_bytes=path.stat().st_size, sha256=sha256(path)))
    if len({video.sha256 for video in manifest}) != len(manifest):
        raise ValueError("TVSum inventory contains byte-identical videos under different IDs")
    return manifest


def select_protocol(manifest, profile):
    """Pick two sources per category in video ID order, before any scoring."""
    groups = {}
    for video in manifest:
        groups.setdefault(video.category, []).append(video)
    if profile not in {"smoke", "full"}:
        raise ValueError(f"Unknown profile: {profile}")
    gallery, queries = [], []
    for index, category in enumerate(sorted(groups)):
        if profile == "smoke" and index == 2:
            break
        source, negative = sorted(groups[category], key=lambda video: video.video_id)[:2]
        gallery.extend([source.video_id, negative.video_id])
        fraction = (0.25, 0.45, 0.65)[index % 3]
        start = round(fraction * (source.duration_seconds - CLIP_SECONDS))
        if start <= 0 or start + CLIP_SECONDS >= source.duration_seconds:
            raise ValueError(f"Cannot take a strict interior clip of {source.video_id}")
        queries.append(dict(query_id=f"{source.video_id}__clip", source_id=source.video_id,
                            category=category, start_seconds=start, duration_seconds=CLIP_SECONDS,
                            relevant_ids=[source.video_id], transform=QUERY_TRANSFORM))
    return dict(profile=profile, gallery_ids=gallery, queries=queries,
                relevance="Same TVSum source as the query's overlapping clip",
                tie_break="Ascending candidate ID for equal scores",
                selection="Two lexicographically first IDs per category, first used as query source",
                frame_sampling="Every round(fps) decoded frames, starting at frame zero",
                query_start_rule="round(fraction * (source_seconds - 16)); 25%, 45%, 65% cycle")


def make_query(dataset, source, query, directory):
    directory.mkdir(parents=True, exist_ok=True)
    # Include inputs in the filename so an interrupted or changed preparation cannot reuse a stale clip.
    fingerprint = hashlib.sha256(
        f"{source.sha256}:{query['start_seconds']}:{query['duration_seconds']}:{CROP_FILTER}:crf26"
        .encode()
    ).hexdigest()[:16]
    destination = directory / f"{query['query_id']}-{fingerprint}.mp4"
    if not destination.exists():
        with TemporaryDirectory(dir=directory) as work:
            temporary = Path(work) / destination.name
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-xerror", "-y",
                "-ss", str(query["start_seconds"]), "-i", str(dataset / source.file),
                "-t", str(query["duration_seconds"]), "-map", "0:v:0", "-an", "-vf", CROP_FILTER,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "26",
                str(temporary),
            ], check=True)
            probe(temporary)
            temporary.replace(destination)
    metadata = probe(destination)
    if abs(metadata["duration_seconds"] - query["duration_seconds"]) > 2 / metadata["fps"]:
        raise ValueError(f"Unexpected query clip duration: {destination}")
    return destination, metadata


def sample_frames(path, fps, expected_frames):
    """Decode sampled RGB frames at the authors' rounded-FPS cadence."""
    cap = open_video(path)
    cadence = round(fps)
    frames, times = [], []
    index = 0
    try:
        while cap.grab():
            if index % cadence == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode frame {index} from {path}")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                times.append(index / fps)
            index += 1
    finally:
        cap.release()
    if len(frames) < 4 or index < expected_frames - max(2, round(0.01 * expected_frames)):
        raise ValueError(f"Video incomplete or too short for ViSiL: {path} ({index}/{expected_frames})")
    return np.stack(frames), np.asarray(times), index


def manifest_rows(manifest):
    return [asdict(video) for video in manifest]
