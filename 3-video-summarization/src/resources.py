"""Prepare the exact DSNet releases and locate the original TVSum media."""

import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import zlib


LAB = Path(__file__).resolve().parents[1]
TVSUM_ARCHIVE = "tvsum50_ver_1_1.tgz"
TVSUM_URL = "https://people.csail.mit.edu/yalesong/tvsum/" + TVSUM_ARCHIVE
TVSUM_SHA256 = "407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea"
FEATURE_NAME = "eccv16_dataset_tvsum_google_pool5.h5"
FEATURE_ZIP = "dsnet_datasets.zip"
FEATURE_URL = "https://dl.dropboxusercontent.com/s/tdknvkpz1jp6iuz/" + FEATURE_ZIP
FEATURE_MIRROR = "https://drive.usercontent.google.com/download?id=11ulsvk1MZI7iDqymw9cfL7csAYS0cDYH&export=download&confirm=t"
FEATURE_SHA256 = "ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b"
CHECKPOINT_ZIP = "pretrain_af_basic.zip"
CHECKPOINT_URL = "https://dl.dropboxusercontent.com/s/2hjngmb0f97nxj0/" + CHECKPOINT_ZIP
CHECKPOINT_SHA256 = "93112811c36c8a841f05ee7b6e18f6c3526647c0d12b0b74a96e49bacc98606b"
# ZIP member byte intervals, read from the authors' published central directories.
FEATURE_RANGE = (58_071_699, 132_274_790)
CHECKPOINT_RANGE = (163, 15_976_843)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_digest(path, expected):
    if expected and sha256(path) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}. Replace it with the published release.")


def fetch(url, destination, byte_range=None):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
               "--connect-timeout", "20", "--max-time", "1000"]
    if byte_range:
        first, last = byte_range
        command.extend(["--range", f"{first}-{last}", "--max-filesize", str(last - first + 1)])
    print(f"Fetching {'ZIP member from ' if byte_range else ''}{url}", flush=True)
    subprocess.run([*command, "--output", str(destination), url], check=True)
    if byte_range and destination.stat().st_size != last - first + 1:
        raise ValueError("The server did not return the requested ZIP member range")


def read_range_member(path, member, destination):
    """Decode one ZIP local entry from a pre-identified HTTP byte range."""
    with path.open("rb") as source:
        signature, _, flags, method, _, _, crc, size, length, name_length, extra_length = (
            struct.unpack("<4s5H3I2H", source.read(30))
        )
        name = source.read(name_length).decode("utf-8")
        source.seek(extra_length, 1)
        if signature != b"PK\x03\x04" or name != member or flags & 9 or method != 8:
            raise ValueError("The published ZIP member layout changed")
        if source.tell() + size != path.stat().st_size:
            raise ValueError("Unexpected compressed member size")
        payload = zlib.decompress(source.read(size), -15)
    if len(payload) != length or zlib.crc32(payload) != crc:
        raise ValueError("The downloaded ZIP member failed its CRC check")
    destination.write_bytes(payload)


def zip_member(destination, zip_name, member, digest, urls, byte_range):
    if destination.is_file():
        check_digest(destination, digest)
        return destination
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    provided_zip = directory / zip_name
    with TemporaryDirectory(dir=directory) as work:
        pending = Path(work) / destination.name
        if provided_zip.is_file():
            with ZipFile(provided_zip) as archive, archive.open(member) as source, pending.open("wb") as target:
                shutil.copyfileobj(source, target)
        else:
            try:
                download = Path(work) / "archive.part"
                fetch(urls[0], download, byte_range)
                read_range_member(download, member, pending)
                check_digest(pending, digest)
            except (subprocess.CalledProcessError, ValueError, struct.error, zlib.error):
                print("Direct member download failed. Trying the published full ZIP.", flush=True)
                for url in urls:
                    try:
                        fetch(url, download)
                        with ZipFile(download) as archive, archive.open(member) as source, pending.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        check_digest(pending, digest)
                        break
                    except (subprocess.CalledProcessError, ValueError, KeyError):
                        continue
                else:
                    raise RuntimeError(f"Unable to obtain {member}. Place the official ZIP at {provided_zip}.")
        check_digest(pending, digest)
        pending.replace(destination)
    return destination


def prepare_dsnet(checkpoint_name):
    features = zip_member(LAB / "data" / FEATURE_NAME, FEATURE_ZIP, FEATURE_NAME, FEATURE_SHA256,
                          (FEATURE_URL, FEATURE_MIRROR), FEATURE_RANGE)
    checkpoint = zip_member(LAB / "weights" / checkpoint_name, CHECKPOINT_ZIP,
                             f"pretrain_af_basic/checkpoint/{checkpoint_name}", CHECKPOINT_SHA256,
                             (CHECKPOINT_URL, "https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip?dl=1"),
                             CHECKPOINT_RANGE)
    return features, checkpoint


def tvsum_archive():
    local = LAB / "data" / TVSUM_ARCHIVE
    sibling = LAB.parent / "2-video-retrieval" / "data" / TVSUM_ARCHIVE
    archive = next((path for path in (local, sibling) if path.is_file()), None)
    if archive:
        check_digest(archive, TVSUM_SHA256)
        return archive
    with TemporaryDirectory(dir=local.parent) as work:
        pending = Path(work) / TVSUM_ARCHIVE
        try:
            fetch(TVSUM_URL, pending)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"TVSum download failed. Place the official archive at {local}.") from error
        check_digest(pending, TVSUM_SHA256)
        pending.replace(local)
    return local


def prepare_tvsum(video_id):
    """Use existing originals where available; extract only the requested video otherwise."""
    data = LAB / "data"
    data.mkdir(parents=True, exist_ok=True)
    cache = data / "tvsum"
    sibling = LAB.parent / "2-video-retrieval" / "data" / "tvsum"
    candidates = [directory / "videos" / f"{video_id}.mp4" for directory in (cache, sibling)]
    video = next((path for path in candidates if path.is_file()), None)
    names = ("info.tsv", "anno.tsv")
    if video and all((cache / name).is_file() for name in names):
        return video, cache
    archive = tvsum_archive()
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=data) as work, tarfile.open(archive, "r:gz") as tar:
        work = Path(work)
        metadata_needed = not all((cache / name).is_file() for name in names)
        for kind, needed in (("data", metadata_needed), ("video", video is None)):
            if not needed:
                continue
            zip_name = f"ydata-tvsum50-{kind}.zip"
            nested = work / zip_name
            with tar.extractfile(f"./ydata-tvsum50-v1_1/{zip_name}") as source, nested.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            with ZipFile(nested) as zip_file:
                if kind == "data":
                    members = [(f"data/ydata-tvsum50-{name}", cache / name) for name in names]
                else:
                    video = candidates[0]
                    video.parent.mkdir(exist_ok=True)
                    members = [(f"video/{video.name}", video)]
                for member, destination in members:
                    pending = work / destination.name
                    with zip_file.open(member) as source, pending.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    pending.replace(destination)
    return video, cache
