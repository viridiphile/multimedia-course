# Implementation audit — 2026-11-16

## Existing work found

The accessible course workspace contained `../lab1/`, a 116 KiB non-Git folder, last filesystem modification dated October 21. It had eight source modules, two command-line scripts, three test modules, README, technical notes, requirements, an output placeholder and pytest cache. There were no notebooks, data, checkpoint files, descriptors, saved retrieval figures, metrics or reports. A pytest cache proves only that tests were invoked previously; it does not establish success or a real-data experiment. File hashes and sizes are preserved in `prior_work_inventory.json`. The original directory remains intact.

No Lab 2/3 implementation was discovered in this workspace. Those stages receive placeholder README files only.

## Reusable components

| Component | Prior implementation | Assessment before execution |
|---|---|---|
| Dataset | `src/dataset.py` | Parses COIL IDs and angles, enforces full 7,200-view inventory, 0°/180° query split, duplicate checks, dimensions, ZIP traversal checks. Reuse protocol and inventory validation. |
| Model | `src/model.py` | Requests the authors' VGG16-GeM checkpoint, pins upstream source revision `1e66a417afa2247edde6d35f3a9a2a465778a3a8`, loads state strictly. This is a genuine GeM plan, but no evidence weights were downloaded or inference ran. |
| Features | `src/features.py` | RGB conversion, checkpoint normalization, ordered batching, official D×N output handling and L2 normalization. Reuse. |
| Ranking | `src/retrieval.py` | Matrix-product similarity, deterministic score/ID ordering, rejects query/gallery overlap. Reuse; enforce unit vectors and add explicit exclusion tests. |
| Metrics | `src/evaluation.py` | Per-query and mean P@5, CSV/JSON output. Reuse; validate malformed result sets. |
| Figures/report | `src/visualization.py`, `src/report.py` | Real-image Top-5, relevance colors, distribution and lowest-P@5 case; results-populated Markdown. Reuse with embedded report figures and validated qualitative discussion. |
| CLI | `scripts/prepare_dataset.py`, `scripts/run_experiment.py` | Preparation, smoke cap, complete evaluation. Reuse with project-local caches, run provenance, portable manifests and separate smoke artifacts. |
| Tests | Three files | Synthetic unit coverage for inventory/splits, ranking, normalization, metrics and report rendering; not pretrained-model validation. |

## Changes required

1. Use Columbia's official dataset ZIP first. Previous Hugging Face/Kaggle download functions imported undeclared optional dependencies and did not use the requested official source.
2. Keep all downloads/caches inside ignored workspace paths. The previous model source loader wrote to the home cache, outside this workspace's writable roots. Verify transport endpoints and record hashes.
3. Pin a tested Python/PyTorch/torchvision environment and record exact installed versions.
4. Add automated checks for preprocessing, actual model descriptor dimensions/numerics, GeM behavior, cosine scores, explicit query exclusion and figure generation.
5. Validate actual image data even if a manifest exists; avoid trusting stale absolute paths or edited split labels. Save portable split records and dataset fingerprints.
6. Add measured runtime/device, learned GeM exponent and complete run metadata; support independent verification of saved results.
7. Expand README, experiment log, genuine AI record preservation and a Lab 1-only report draft. No combined three-lab Word report in this phase.

## Status at audit commit

No current-session tests, data downloads, checkpoint loading or retrieval evaluation have completed yet. Measurements will be recorded only after execution. Existing code will be copied into `1-image-retrieval/` and corrected in later commits; it is not represented as newly authored from scratch.

## Completion status for user review

The reused pipeline is now executed against the official complete COIL-100 archive
and the named authors' SHA-256-verified checkpoint. The complete 200-query /
7,000-gallery experiment measured mean P@5 0.9970. All 7,200 images were validated;
198 queries scored 1.0, one 0.8 and one 0.6. All 18 automated checks ran without
skips, including official GeM behavior and actual descriptors. Independent
verification checked all 1,000 ranked rows and exact descriptor rankings.

The main corrections were official HTTPS dataset preparation, local source/cache
paths, portable validated manifests, checkpoint hash/legacy-metadata handling,
cosine normalization at the ranking API boundary, explicit smoke segregation,
run provenance and an independent verifier. Figure spacing was corrected after
visual inspection using saved actual rankings. Original-file hashes still match;
no prior implementation was discarded or overwritten.

The environment and results are validated on this macOS arm64 machine (MPS for
full inference, CPU for model unit checks). Other platforms and sparse-angle
protocols have not been experimentally validated. Dependency deprecation warnings
are recorded. The final combined Word report, Labs 2/3 execution and final
student submission remain future work outside this authorization. There is no
remaining Lab 1 execution blocker; the next required step is the user's review,
not automatic progression to Lab 2.
