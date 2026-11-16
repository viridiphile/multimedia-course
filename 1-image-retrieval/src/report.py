"""Create a results-grounded Markdown report draft after a real experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def write_report(
    metrics: Dict,
    config: Dict,
    case: Dict,
    results: List[Dict],
    output_path: Path,
) -> None:
    if metrics["queries_evaluated"] != len(metrics["precision_at_5_per_query"]):
        raise ValueError("Report metrics do not include every evaluated query")
    examples = []
    seen = set()
    for row in results:
        if row["query_id"] not in seen:
            seen.add(row["query_id"])
            examples.append(row["query_id"])
        if len(examples) == 5:
            break
    example_text = ", ".join(examples)
    result_rows = [
        row for row in results if row["query_id"] in examples
    ]
    per_query = {
        row["query_id"]: row["precision_at_5"]
        for row in metrics["precision_at_5_per_query"]
    }
    per_query_lines = [
        "| {} | {:.1f} |".format(row["query_id"], row["precision_at_5"])
        for row in metrics["precision_at_5_per_query"]
    ]
    example_lines = [
        "| {} | {:.1f} | {} |".format(
            query_id, per_query[query_id], ", ".join(
                "{} ({:.5f}, {})".format(
                    row["image_id"], row["similarity"],
                    "relevant" if row["relevant"] else "not relevant",
                )
                for row in result_rows
                if row["query_id"] == query_id
            )
        )
        for query_id in examples
    ]
    case_results = "; ".join(
        "rank {}: {} (object {}, {}°, score {:.6f}, {})".format(
            row["rank"], row["image_id"], row["object_id"], row["angle"],
            row["similarity"], "relevant" if row["relevant"] else "not relevant"
        )
        for row in case["results"]
    )
    relevant_angles = [
        row["angle"] for row in case["results"] if row["relevant"]
    ]
    incorrect_objects = sorted(
        {row["object_id"] for row in case["results"] if not row["relevant"]}
    )
    case_observation = (
        "Of the five results, {} depict the query object{}; their viewing angles "
        "are {}. The incorrect results belong to object ID{} {}. These are "
        "direct observations from the ranked output; inspect the case figure to "
        "compare appearance and viewpoint before attributing a cause."
    ).format(
        len(relevant_angles),
        "s" if len(relevant_angles) != 1 else "",
        ", ".join("{}°".format(angle) for angle in relevant_angles) or "none",
        "s" if len(incorrect_objects) != 1 else "",
        ", ".join(str(object_id) for object_id in incorrect_objects) or "none",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Lab 1 — COIL-100 Image Retrieval

## Objective

Retrieve the five most visually similar COIL-100 gallery images for each query
using an inference-only pretrained image-retrieval descriptor, then measure
same-object Precision@5.

## Dataset

COIL-100 contains 100 physical objects and 72 RGB views per object (7,200
images, 128 × 128 pixels). The source is the
[Columbia Object Image Library](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php).
The fixed split used here has {query_count} query images (0° and 180° for each
object) and {gallery_count} gallery images (the remaining views). Relevance is
defined exclusively by matching object ID. There is no training/validation
split because this experiment performs no model training.

## Method

The network is the authors' pretrained **{checkpoint}** model: VGG16 with
Generalized-Mean pooling, trained for image retrieval on retrieval-SfM-120k.
The learned GeM exponent is **{gem_p}**, with epsilon 1e-6; this is not
average pooling. The unmodified authors' network source is pinned to revision
`{upstream_revision}`. No multiscale aggregation or additional discriminative
whitening is applied. Images stay at native 128×128 resolution, an explicit
choice differing from resized evaluation settings.
RGB images are converted to tensors and normalized using channel mean/std
stored in the official checkpoint. Inference is performed with gradients
disabled. Descriptors are L2-normalized and ranked by cosine similarity
`q · g`; ties are resolved by gallery image ID. Exactly five gallery results
are retained per query.

## Experiment setup

- Python: {python_version}
- Platform: {platform}
- Device: {device}
- PyTorch: {torch_version}
- Batch size: {batch_size}
- Input dimensions: {input_dimensions}
- Descriptor dimensions: {descriptor_dimensions}
- Checkpoint SHA-256: `{checkpoint_sha256}`
- Queries evaluated: {query_count}; gallery images: {gallery_count}

## Results

Mean P@5 over {evaluated} evaluated queries: **{mean_precision:.4f}**.
Queries with P@5 = 1: {perfect}; queries with P@5 = 0: {zero}.

Example query results (similarity scores and relevance are actual run outputs):

| Query | P@5 | Top-5 gallery results (score, relevance) |
|---|---:|---|
{example_lines}

![Top-5 retrieval examples](../figures/retrieval_examples.png)

Figure 1. Selected actual queries and their five ranked gallery images. Each result
shows its ID, cosine score and relevance. Images: Columbia COIL-100.

![Distribution of P@5](../figures/precision_at_5.png)

Figure 2. Query-level precision distribution for this run.

Per-query values are in `../metrics/per_query_precision.csv`; all ranked results
are in `../metrics/top5_results.csv`.

All evaluated query-level measurements:

| Query | P@5 |
|---|---:|
{per_query_lines}

## Case analysis

The selected query is **{case_query}** (object {case_object}, angle
{case_angle}°), with P@5 = {case_precision:.1f}. Its observed ranked results:
{case_results}.

{case_observation}

![Lowest-P@5 case](../figures/case_analysis.png)

Figure 3. Lowest observed P@5, selected deterministically by score then query ID.

Interpret
errors by comparing object appearance and viewpoint in that figure; possible
factors include viewpoint change, shape/texture similarity to another object,
and the difference between COIL-100's turntable imagery and the landmark
retrieval data used to train this descriptor. Do not treat these as proven
causes without visual evidence.

## Limitations

This is a single pretrained model and a single fixed viewpoint split. The gallery
contains neighboring 5° views, so this is a relatively favorable dense-view
protocol and does not establish robustness to large viewpoint gaps. P@5
depends on the definition of relevance and does not capture all ranking
behavior. No model adaptation or parameter tuning is performed. A landmark-
trained embedding may not optimally represent small controlled object images;
qualitative interpretation should be based on the generated case figure.

## References

1. F. Radenović, G. Tolias, and O. Chum, “Fine-Tuning CNN Image Retrieval with
   No Human Annotation,” *IEEE TPAMI*, 41(7), 1655–1668, 2019.
   https://doi.org/10.1109/TPAMI.2018.2846566
2. [Authors' PyTorch implementation](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
   and `retrievalSfM120k-vgg16-gem` checkpoint.
3. S. Nene, S. Nayar, and H. Murase, *Columbia Object Image Library (COIL-100)*,
   Columbia University.
""".format(
            query_count=config["query_count"],
            gallery_count=config["gallery_count"],
            checkpoint=config["checkpoint_name"],
            gem_p=config.get("gem_p", "see saved experiment metadata"),
            upstream_revision=config.get("upstream_revision", "see model source"),
            python_version=config["python_version"],
            platform=config["platform"],
            device=config["device"],
            torch_version=config["torch_version"],
            batch_size=config["batch_size"],
            input_dimensions=config["input_dimensions"],
            descriptor_dimensions=config["descriptor_dimensions"],
            checkpoint_sha256=config["checkpoint_sha256"],
            evaluated=metrics["queries_evaluated"],
            mean_precision=metrics["mean_precision_at_5"],
            perfect=metrics["queries_precision_at_5_equal_1"],
            zero=metrics["queries_precision_at_5_equal_0"],
            example_lines="\n".join(example_lines),
            per_query_lines="\n".join(per_query_lines),
            case_query=case["query_id"],
            case_object=case["query_object_id"],
            case_angle=case["query_angle"],
            case_precision=case["precision_at_5"],
            case_results=case_results,
            case_observation=case_observation,
        ),
        encoding="utf-8",
    )

    if config.get("is_smoke_run"):
        text = output_path.read_text(encoding="utf-8")
        output_path.write_text("**SMOKE RUN — limited queries/gallery; not final benchmark results.**\n\n" + text,
                               encoding="utf-8")
