"""Batched RGB feature extraction and descriptor validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.dataset import ImageRecord


def l2_normalize(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError("Descriptors must be a 2D matrix")
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptors contain NaN or infinite values")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Descriptors contain a zero-length vector")
    return descriptors / norms


class _ImageDataset:
    def __init__(self, records: Sequence[ImageRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(Path(record.path)) as image:
                rgb = image.convert("RGB")
                return self.transform(rgb), record.image_id
        except OSError as exc:
            raise RuntimeError("Could not read image {}: {}".format(record.path, exc)) from exc


def checkpoint_transform(meta):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=meta["mean"], std=meta["std"]),
        ]
    )


def extract_features(
    records: Sequence[ImageRecord],
    model: torch.nn.Module,
    meta,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not records:
        raise ValueError("Cannot extract features from an empty image set")

    loader = DataLoader(
        _ImageDataset(records, checkpoint_transform(meta)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    identifiers: List[str] = []
    descriptor_batches = []
    expected_dim = int(meta["outputdim"])

    try:
        with torch.inference_mode():
            for images, image_ids in tqdm(loader, desc="Extracting features", unit="batch"):
                outputs = model(images.to(device))
                expected_shape = (expected_dim, len(image_ids))
                if outputs.ndim != 2 or tuple(outputs.shape) != expected_shape:
                    raise ValueError(
                        "The authors' network returns descriptors shaped {}, got {}".format(
                            expected_shape, tuple(outputs.shape)
                        )
                    )
                if not torch.isfinite(outputs).all():
                    raise ValueError("Model produced NaN or infinite descriptors")
                descriptor_batches.append(
                    outputs.transpose(0, 1).detach().to("cpu").numpy()
                )
                identifiers.extend(list(image_ids))
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError("Feature extraction failed: {}".format(exc)) from exc

    if identifiers != [record.image_id for record in records]:
        raise RuntimeError("Feature-to-image mapping changed during extraction")
    descriptors = l2_normalize(np.concatenate(descriptor_batches, axis=0))
    return identifiers, descriptors
