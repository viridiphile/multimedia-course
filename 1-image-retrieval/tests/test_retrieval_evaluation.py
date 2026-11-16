import numpy as np
import pytest

from src.dataset import ImageRecord
from src.evaluation import evaluate_precision_at_k
from src.features import l2_normalize
from src.retrieval import rank_top_k


def record(image_id, object_id, split, angle=0):
    return ImageRecord(
        image_id=image_id,
        filename=image_id + ".png",
        path="/unused/" + image_id + ".png",
        object_id=object_id,
        angle=angle,
        split=split,
    )


def test_l2_normalize_outputs_unit_vectors():
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


def test_l2_normalize_rejects_invalid_or_zero_vectors():
    with pytest.raises(ValueError, match="zero-length"):
        l2_normalize(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="NaN"):
        l2_normalize(np.array([[float("nan"), 1.0]]))


def test_cosine_ranking_mapping_and_deterministic_ties():
    queries = [record("q1", 1, "query")]
    gallery = [
        record("g-z", 2, "gallery"),
        record("g-b", 1, "gallery"),
        record("g-a", 2, "gallery"),
        record("g-d", 1, "gallery"),
        record("g-c", 1, "gallery"),
        record("g-e", 3, "gallery"),
    ]
    query_vectors = np.array([[1.0, 0.0]])
    gallery_vectors = np.array(
        [[0.5, 0.866], [1.0, 0.0], [0.5, 0.866], [0.8, 0.6], [0.7, 0.714], [-1.0, 0.0]]
    )
    results = rank_top_k(queries, gallery, query_vectors, gallery_vectors, k=5)
    assert [row["rank"] for row in results] == [1, 2, 3, 4, 5]
    assert [row["image_id"] for row in results] == [
        "g-b",
        "g-d",
        "g-c",
        "g-a",
        "g-z",
    ]
    assert all(row["query_id"] == "q1" for row in results)
    assert [row["relevant"] for row in results] == [True, True, True, False, False]
    assert results[0]["similarity"] >= results[-1]["similarity"]


def test_precision_at_5_and_mean_are_exact():
    results = []
    relevant_by_query = {
        "q1": [True, True, True, True, True],
        "q2": [False, False, False, False, False],
        "q3": [True, True, False, False, False],
    }
    for query_id, relevant in relevant_by_query.items():
        for rank, is_relevant in enumerate(relevant, start=1):
            results.append(
                {"query_id": query_id, "rank": rank, "relevant": is_relevant}
            )
    metrics = evaluate_precision_at_k(results)
    assert metrics["queries_evaluated"] == 3
    assert metrics["precision_at_5_per_query"] == [
        {"query_id": "q1", "precision_at_5": 1.0},
        {"query_id": "q2", "precision_at_5": 0.0},
        {"query_id": "q3", "precision_at_5": 0.4},
    ]
    assert metrics["mean_precision_at_5"] == pytest.approx(1.4 / 3)
    assert metrics["queries_precision_at_5_equal_1"] == 1
    assert metrics["queries_precision_at_5_equal_0"] == 1
