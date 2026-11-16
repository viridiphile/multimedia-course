"""Loading the genuine VGG16-GeM checkpoint from the authors' implementation."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


CHECKPOINT_NAME = "retrievalSfM120k-vgg16-gem"
OFFICIAL_REPOSITORY_REVISION = "1e66a417afa2247edde6d35f3a9a2a465778a3a8"
OFFICIAL_SOURCE_URL = (
    "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/"
    "{}.zip".format(OFFICIAL_REPOSITORY_REVISION)
)
CHECKPOINT_URL = (
    "https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/"
    "retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth"
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("Unsupported device {!r}; choose auto, mps, or cpu".format(requested))


def _download_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, str(temporary))
        temporary.replace(path)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the official {} checkpoint from {}: {}".format(
                CHECKPOINT_NAME, CHECKPOINT_URL, exc
            )
        ) from exc


def load_retrieval_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """Load exact checkpoint weights using the official cirtorch network builder."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        _download_checkpoint(checkpoint_path)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise RuntimeError("Checkpoint is missing or empty: {}".format(checkpoint_path))

    try:
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not read the official checkpoint {}. It may be incomplete or "
            "incompatible with this PyTorch version: {}".format(checkpoint_path, exc)
        ) from exc

    if not isinstance(state, dict) or "meta" not in state or "state_dict" not in state:
        raise RuntimeError(
            "Checkpoint does not contain the authors' expected meta/state_dict fields"
        )
    meta = state["meta"]
    required = {"architecture", "pooling", "mean", "std"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise RuntimeError("Checkpoint metadata is incomplete: expected {}".format(required))
    if meta["architecture"] != "vgg16" or meta["pooling"] != "gem":
        raise RuntimeError(
            "Expected the official VGG16-GeM checkpoint, found {}-{}".format(
                meta["architecture"], meta["pooling"]
            )
        )

    _add_official_source()
    try:
        from cirtorch.networks.imageretrievalnet import init_network
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the pinned official cirtorch network implementation. "
            "Check that PyTorch and torchvision are installed; generic torchvision "
            "VGG16 is not an acceptable substitute."
        ) from exc

    params = {
        "architecture": meta["architecture"],
        "pooling": meta["pooling"],
        "local_whitening": meta.get("local_whitening", False),
        "regional": meta.get("regional", False),
        "whitening": meta.get("whitening", False),
        "mean": meta["mean"],
        "std": meta["std"],
        "pretrained": False,
    }
    try:
        model = init_network(params)
        model.load_state_dict(state["state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            "The official checkpoint could not be applied to the authors' network "
            "implementation. No alternate model was loaded: {}".format(exc)
        ) from exc

    model.to(device)
    model.eval()
    descriptor_meta = dict(model.meta)
    descriptor_meta["checkpoint_name"] = CHECKPOINT_NAME
    descriptor_meta["checkpoint_url"] = CHECKPOINT_URL
    descriptor_meta["checkpoint_sha256"] = _sha256(checkpoint_path)
    descriptor_meta["checkpoint_path"] = str(checkpoint_path.resolve())
    descriptor_meta["gem_p"] = float(model.pool.p.detach().cpu().item())
    descriptor_meta["upstream_revision"] = OFFICIAL_REPOSITORY_REVISION
    return model, descriptor_meta, descriptor_meta["checkpoint_sha256"]


def _add_official_source() -> None:
    """Fetch the exact upstream source without installing its obsolete torch pin."""
    source_root = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / ("cnnimageretrieval-pytorch-" + OFFICIAL_REPOSITORY_REVISION)
    )
    if not (source_root / "cirtorch" / "networks" / "imageretrievalnet.py").is_file():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="cirtorch-source-", dir=str(source_root.parent)
            ) as temporary:
                archive = Path(temporary) / "source.zip"
                urllib.request.urlretrieve(OFFICIAL_SOURCE_URL, str(archive))
                extract_root = Path(temporary) / "extract"
                extract_root.mkdir()
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        target = (extract_root / member.filename).resolve()
                        if extract_root.resolve() not in target.parents:
                            raise RuntimeError(
                                "Unsafe path in official source archive: {}".format(
                                    member.filename
                                )
                            )
                    bundle.extractall(extract_root)
                candidates = list(extract_root.iterdir())
                if len(candidates) != 1 or not (
                    candidates[0] / "cirtorch" / "networks" / "imageretrievalnet.py"
                ).is_file():
                    raise RuntimeError(
                        "Official source archive has an unexpected directory layout"
                    )
                shutil.move(str(candidates[0]), str(source_root))
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                "Could not retrieve the pinned authors' source revision {} from {}: {}"
                .format(OFFICIAL_REPOSITORY_REVISION, OFFICIAL_SOURCE_URL, exc)
            ) from exc
    sys.path.insert(0, str(source_root))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
