# Lab 1 — Image retrieval using GeM

Inference with the authors' **retrievalSfM120k-vgg16-gem** checkpoint on the real COIL-100 image dataset. Existing course code was preserved and corrected; see [the implementation audit](../docs/implementation_audit.md).

## Installation

Tested target: CPython 3.9.6, macOS arm64. CPU works; `--device auto` prefers Apple MPS when available. Allow approximately 2 GB for the environment, data, weights and source cache.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
```

The pinned upstream implementation is downloaded automatically into ignored `vendor/`, bypassing its obsolete package installation pins. We load the authors' exact model and state dictionary strictly. Generic ImageNet descriptors are never substituted. Network access is needed only to prepare data/source/weights; subsequent runs use local files.

## Prepare and run

```bash
python scripts/prepare_dataset.py
python scripts/run_experiment.py --smoke-objects 3 --max-queries 4 --batch-size 8 --device auto --output-dir results/smoke
python scripts/run_experiment.py --batch-size 16 --device auto
python -m pytest -q
python scripts/verify_results.py --descriptors
```

Dataset preparation downloads the [official Columbia ZIP](https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip), validates the complete image inventory, and creates `results/metrics/coil100_manifest.csv`. A local archive can be supplied with `--archive PATH`, and an existing data directory with `--data-dir PATH`. Pass the same data directory to the experiment when overriding the default. Delete/regenerate a stale manifest if the dataset directory changes.

The smoke run validates all source data, then extracts only 4 queries and 210 gallery images. Its scores are **not** the final benchmark. Smoke flags require a separate output directory to protect final metrics. The final command omits both limits and evaluates every intended query.

The CLI also supports `--weights PATH`, `--manifest PATH`, `--output-dir PATH`, `--batch-size N` and `--device auto|cpu|mps`. For CPU execution use `--device cpu`. Exact rerun command/device/versions are recorded in the saved experiment JSON; CPU and MPS can differ slightly numerically.

## Dataset and protocol

- Total: 7,200 real 128×128 images, 100 objects, 72 angles per object (0°–355° in 5° steps).
- Queries: the 0° and 180° views of each object, **200** total.
- Gallery: the other 70 views of each object, **7,000** total.
- Relevant means the same physical object ID. The query itself cannot enter the gallery.
- The split is deterministic, ordered by numeric object ID and angle; no random selection. Execution seed is 0.
- Inventory, decoded format/dimensions, duplicate IDs/content, and manifest consistency are checked before inference. Manifests store relative paths. An ordered SHA-256 fingerprint identifies the image bytes.

This protocol includes nearby 5° views in the gallery. Strong scores measure retrieval under this dense-view protocol; they do not demonstrate invariance to large viewpoint gaps or generalization to unseen objects.

## Method

VGG16 convolutional features (last max-pool and classifier omitted by the authors' network builder) feed actual learned GeM pooling:

`GeM(x)_c = mean(max(x_c, epsilon) ** p) ** (1/p)`

The exponent `p` is loaded from the retrieval-SfM-120k checkpoint, not set to average pooling (`p=1`). The model returns 512-dimensional descriptors. Native 128×128 RGB images are scaled to [0,1], then normalized with checkpoint channel means/stds. No crop, augmentation, multiscale extraction, additional discriminative whitening, or COIL training is performed. Native resolution is an explicit experiment choice: the authors' evaluation helper ordinarily accepts a maximum side length; this pipeline applies no resizing.

Inference runs in evaluation mode with gradients disabled. Descriptors are L2 normalized; cosine similarity ranks all gallery images in descending order, using image ID to break exact ties. For each query, `P@5 = relevant Top-5 results / 5`. Mean P@5 is the arithmetic mean across actual evaluated queries.

## Artifacts and verification

| File under `results/` | Contents |
|---|---|
| `metrics/coil100_manifest.csv` | All image IDs, object IDs, angles, splits, portable paths |
| `metrics/top5_results.csv` | Five ranked IDs/scores/relevance labels per query |
| `metrics/per_query_precision.csv` | Every evaluated query's P@5 |
| `metrics/experiment_metrics.json` | Aggregate/per-query metrics, dataset fingerprint, checkpoint SHA, learned p, device, dependencies and timing |
| `metrics/case_analysis.json` | Actual lowest-P@5 case, selected deterministically |
| `metrics/verification.json` | Independent checks of exported metrics/rankings |
| `figures/retrieval_examples.png` | Ten query-plus-Top-5 rows with IDs, scores and match labels |
| `figures/case_analysis.png` | Detailed worst-case row |
| `figures/precision_at_5.png` | Distribution across all queries |
| `report/lab1_report.md` | Generated Lab 1-only draft with embedded figures and all query P@5 values |
| `descriptors.npz` | Local-only normalized descriptors for exact ranking verification; excluded from Git |
| `smoke/` | Clearly separated small initial experiment |

`verify_results.py` independently checks CSV versus JSON P@5, same-object relevance, disjoint splits, query completeness, rank order, duplicate results and figure files. Add `--descriptors` to recompute cosine rankings for every query from the local descriptor cache, without using the retrieval implementation. Without that flag, the committed CSV/JSON/manifest/figures can be checked on a fresh clone before downloading the data. Run the verifier from this Lab 1 directory.

Tests cover dataset inventory/duplicates, preprocessing values/dimensions, learned GeM against the mathematical power mean, real pretrained descriptor dimensions/finite values/unit norm, cosine scores, ranking/ties, query exclusion, P@5 and real-image figure rendering. Real-asset tests skip when data/weights/source have not been prepared; the final local validation must report their actual status. Synthetic unit fixtures are not used as retrieval accuracy measurements.

See [experiment log](../docs/experiment_log.md), [Lab 1 report draft](../reports/lab1_report.md), and [AI disclosure/export guidance](../docs/ai_usage.md). The final combined Word report is deferred until Labs 2 and 3 are authorized and complete.

## Sources

- Radenović, Tolias and Chum (2019), *Fine-Tuning CNN Image Retrieval with No Human Annotation*, IEEE TPAMI 41(7), 1655–1668. [Paper](https://arxiv.org/abs/1711.02512), [DOI](https://doi.org/10.1109/TPAMI.2018.2846566).
- [Authors' MIT-licensed implementation](https://github.com/filipradenovic/cnnimageretrieval-pytorch), pinned revision `1e66a417afa2247edde6d35f3a9a2a465778a3a8`. Source remains unmodified in the ignored local cache, including its license. [Official checkpoint](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth).
- Nene, Nayar and Murase (1996), *Columbia Object Image Library (COIL-100)*, CUCS-006-96. [Official source](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php). Dataset thumbnails appear only as credited experimental result figures.
