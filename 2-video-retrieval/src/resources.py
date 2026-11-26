"""Download published assets and extract only the members used by Lab 2."""

from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .dataset import sha256


LAB = Path(__file__).resolve().parents[1]
ARCHIVE = "tvsum50_ver_1_1.tgz"
DATA_URL = f"https://people.csail.mit.edu/yalesong/tvsum/{ARCHIVE}"
MIRROR = "https://drive.usercontent.google.com/download?id=1kHe-PKGRX9IJYtSIs1Ot_D0Zj67U1CjG&export=download&confirm=t"
TF_REVISION = "0971e54fb8325fceb1bc9748ecbfe4c66e5dabd2"
TF_DIGESTS = {
    "model.index": "84169ac6f98b7ff375ed7a130ce45276bbe2d3d8f3901bfd26bf54ee8ff4efb1",
    "model.data-00000-of-00001": "9b45ee5022406713b083a6ce31cfa354bc4ad03678e8015259727c214a30dbea",
    "model.meta": "e419d7bf63b3a1507253b52e95f70ee7f4185e91ed01aaa8240a004bb0e91f4b",
}
TF_FILES = tuple(TF_DIGESTS)
# Digests for the exact release artifacts used in this experiment.
DATA_SHA256 = "407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea"
CHECKPOINT_SHA256 = "d235e1ce56d39782a7faf81a0713262672aeb7564b831572acc0c7a9ee8da764"


def download(url, path, expected):
    def verify(file):
        if expected and sha256(file) != expected:
            raise ValueError(f"SHA-256 mismatch for {file}. Replace it with the published asset.")
    if path.is_file():
        verify(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {path.relative_to(LAB)}", flush=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        partial = Path(temporary) / path.name
        try:
            subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout",
                            "20", "--max-time", "900", "--output", str(partial), url], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Download failed. Place the published file at {path}") from exc
        verify(partial)
        partial.replace(path)


def extract_member(archive, name, destination):
    with archive.open(name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def prepare_dataset():
    data = LAB / "data"
    dataset = data / "tvsum"
    if (dataset / "info.tsv").is_file() and (dataset / "videos").is_dir():
        return dataset
    path = data / ARCHIVE
    download(DATA_URL, path, DATA_SHA256)
    with TemporaryDirectory(dir=data) as temporary:
        stage = Path(temporary) / "tvsum"
        (stage / "videos").mkdir(parents=True)
        with tarfile.open(path, "r:gz") as tar:
            for kind in ("data", "video"):
                name = f"ydata-tvsum50-{kind}.zip"
                nested = Path(temporary) / name
                member = tar.getmember(f"./ydata-tvsum50-v1_1/{name}")
                with tar.extractfile(member) as source, nested.open("wb") as target:
                    shutil.copyfileobj(source, target)
                with ZipFile(nested) as archive:
                    if kind == "data":
                        extract_member(archive, "data/ydata-tvsum50-info.tsv", stage / "info.tsv")
                        continue
                    for entry in archive.namelist():
                        parsed = Path(entry)
                        if parsed.parent == Path("video") and parsed.suffix.lower() == ".mp4":
                            extract_member(archive, entry, stage / "videos" / parsed.name)
        stage.rename(dataset)
    return dataset


def prepare_checkpoint():
    weights = LAB / "weights"
    checkpoint = weights / "resnet"
    if all((checkpoint / name).is_file() for name in TF_FILES):
        return checkpoint
    bundle = weights / "visil-tf.zip"
    download(MIRROR, bundle, CHECKPOINT_SHA256)
    with TemporaryDirectory(dir=weights) as temporary:
        stage = Path(temporary) / "resnet"
        stage.mkdir()
        with ZipFile(bundle) as archive:
            for name in TF_FILES:
                extract_member(archive, f"ckpt/resnet/{name}", stage / name)
        stage.rename(checkpoint)
    return checkpoint
