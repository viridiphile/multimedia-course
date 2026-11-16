"""Cosine-similarity ranking with stable deterministic tie-breaking."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.dataset import ImageRecord
from src.features import l2_normalize


def rank_top_k(
    query_records: Sequence[ImageRecord],
    gallery_records: Sequence[ImageRecord],
    query_descriptors: np.ndarray,
    gallery_descriptors: np.ndarray,
    k: int = 5,
) -> List[Dict]:
    if k < 1:
        raise ValueError("k must be positive")
    if len(gallery_records) < k:
        raise ValueError("Gallery has {} images; at least {} are required".format(len(gallery_records), k))
    if query_descriptors.shape[0] != len(query_records):
        raise ValueError("Query descriptor count does not match query records")
    if gallery_descriptors.shape[0] != len(gallery_records):
        raise ValueError("Gallery descriptor count does not match gallery records")
    if query_descriptors.shape[1] != gallery_descriptors.shape[1]:
        raise ValueError("Query and gallery descriptor dimensions differ")
    if not np.isfinite(query_descriptors).all() or not np.isfinite(gallery_descriptors).all():
        raise ValueError("Descriptors must be finite")

    query_ids = {row.image_id for row in query_records}
    gallery_ids = {row.image_id for row in gallery_records}
    if query_ids & gallery_ids:
        raise ValueError("A query image is present in the gallery")

    # Ensure this API computes cosine similarity even for unnormalized callers.
    scores = np.clip(
        l2_normalize(query_descriptors) @ l2_normalize(gallery_descriptors).T,
        -1.0, 1.0,
    )
    ranked: List[Dict] = []
    for query_index, query in enumerate(query_records):
        # lexsort's final key is primary: negative score gives descending order.
        order = np.lexsort(
            (
                np.asarray([row.image_id for row in gallery_records]),
                -scores[query_index],
            )
        )[:k]
        for rank, gallery_index in enumerate(order, start=1):
            result = gallery_records[int(gallery_index)]
            ranked.append(
                {
                    "query_id": query.image_id,
                    "query_filename": query.filename,
                    "query_object_id": query.object_id,
                    "query_angle": query.angle,
                    "rank": rank,
                    "image_id": result.image_id,
                    "filename": result.filename,
                    "object_id": result.object_id,
                    "angle": result.angle,
                    "similarity": float(scores[query_index, gallery_index]),
                    "relevant": result.object_id == query.object_id,
                }
            )
    return ranked
