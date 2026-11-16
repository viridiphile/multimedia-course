from src.report import write_report


def test_report_uses_actual_metrics_and_case_observations(tmp_path):
    metrics = {
        "queries_evaluated": 1,
        "precision_at_5_per_query": [
            {"query_id": "obj1__0", "precision_at_5": 0.4}
        ],
        "mean_precision_at_5": 0.4,
        "queries_precision_at_5_equal_1": 0,
        "queries_precision_at_5_equal_0": 0,
    }
    config = {
        "query_count": 1,
        "gallery_count": 7000,
        "checkpoint_name": "retrievalSfM120k-vgg16-gem",
        "python_version": "3.test",
        "platform": "test-platform",
        "device": "cpu",
        "torch_version": "test",
        "batch_size": 2,
        "input_dimensions": "128x128 RGB",
        "descriptor_dimensions": 512,
        "checkpoint_sha256": "abc",
    }
    case = {
        "query_id": "obj1__0",
        "query_object_id": 1,
        "query_angle": 0,
        "precision_at_5": 0.4,
        "results": [
            {
                "rank": rank,
                "image_id": "obj{}__{}".format(1 if rank <= 2 else rank, rank * 5),
                "object_id": 1 if rank <= 2 else rank,
                "angle": rank * 5,
                "similarity": 0.9 - rank / 100,
                "relevant": rank <= 2,
            }
            for rank in range(1, 6)
        ],
    }
    results = [
        {
            "query_id": "obj1__0",
            "image_id": "obj1__5",
            "similarity": 0.89,
            "relevant": True,
        }
    ]

    destination = tmp_path / "lab1_report.md"
    write_report(metrics, config, case, results, destination)
    report = destination.read_text(encoding="utf-8")
    assert "Mean P@5 over 1 evaluated queries: **0.4000**" in report
    assert "| obj1__0 | 0.4 |" in report
    assert "their viewing angles are 5°, 10°" in report
    assert "object IDs 3, 4, 5" in report
