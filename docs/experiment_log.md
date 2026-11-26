# Experiment log

Dates use Asia/Shanghai (UTC+08:00). Commands run from the repository root unless specified.

## 2026-11-16 — Setup

- Read the user technical request and extracted the assignment document with macOS textutil. The request limits current work to setup, audit and Lab 1.
- Workspace contained `../lab1/` (116 KiB); no Git metadata or saved experimental outputs.
- GitHub connector confirmed `viridiphile/multimedia-course` is public, empty, default branch main, with push permission.
- Sandboxed Git failed DNS; network-enabled `git clone https://github.com/viridiphile/multimedia-course.git multimedia-course` succeeded with an empty-repository warning.
- Created an isolated environment using `python3 -m venv multimedia-course-env` in the parent workspace.

## 2026-11-16 — Implementation and asset preparation

- Preserved all original `../lab1/` files; copied modules/scripts/tests into the requested Lab 1 directory.
- Verified HTTP 200 over HTTPS for the official COIL archive, named authors' checkpoint and pinned upstream source archive.
- Downloaded with `curl -fL --retry 2 --connect-timeout 30 --max-time 600 -o <local-file> <official-URL>`. COIL ZIP: 130,688,843 bytes, SHA-256 `fcb4b8b01c0778f416d8d7a09f45082c4f7075cc74743dcddd5c44307bd4afe7`. Checkpoint: 65,121,467 bytes, SHA-256 `b4dcdc6b1dc22d9291a27417034a9fa608f697556d26116fafbfa99b5ce0ab95`.
- Downloaded pinned source ZIP to a temporary path and safely extracted it to ignored `1-image-retrieval/vendor/`; source ZIP SHA-256 `85fdcaaa717ebee0ab95e2ce580cdf8e32a8e4991cf6fd2e7bc1fc96f314a978`. The source is unmodified and retains its MIT license.
- From `1-image-retrieval/`: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare_dataset.py --archive data/downloads/coil-100.zip` completed: **7,200 valid images, 200 queries, 7,000 gallery images**.
- Initial system-Python `compileall` could not write Apple's global bytecode cache under the sandbox. Replaced that check with `ast.parse` for every project source, which passed, and disabled bytecode during subsequent runs. This was an environment/cache restriction, not a source syntax error.
- Dependency installation began with `../multimedia-course-env/bin/python -m pip install --no-cache-dir` and the exact versions in Lab 1 requirements. The torch wheel download was unusually slow; direct official PyPI download was attempted as a transport fix. No alternate model or synthetic evaluation dataset was substituted.
- Implementation records learned GeM p, exact model/dataset identifiers, normalization, device, versions and timings. Smoke outputs require a separate directory. Added an independent saved-results verifier.

### Preliminary unit validation

From `1-image-retrieval/`: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_dataset.py tests/test_retrieval_evaluation.py tests/test_report.py` — **10 passed in 0.69 s** using the preexisting system scientific packages. This check covers reused dataset/ranking/metrics/report logic; it is not pretrained inference or retrieval accuracy. The full isolated-environment suite and actual evaluation remain pending.

The alternative official PyTorch wheel-host URL returned HTTP 403. A 1,024-byte range request to the PyPI wheel succeeded, so concurrent range downloading with SHA-256 validation against official PyPI metadata was started as a transport workaround.

### Dependency transport fix

The single-stream PyPI torch transfer remained slow. Downloaded official wheels to temporary local wheel directories using concurrent HTTP byte ranges; every resulting file was checked against its SHA-256 from the package's official PyPI version JSON. Torch wheel hash: `e9f071f5b52a9f6970dc8a919694b27a91ae9dc08898b2b988abbef5eddfd1ae`. No alternative package index or model was used. Stopped the two redundant transfers started by this task. One metadata request for pyparsing failed transient TLS and was retried. An initial offline install failed cleanly because that one wheel was missing; installation was retried after fetching it.

Offline environment installation command from parent workspace:

```bash
multimedia-course-env/bin/python -m pip install --no-index \
  --find-links /private/tmp/multimedia-torch-chunks \
  --find-links /private/tmp/multimedia-wheelhouse \
  -r multimedia-course/1-image-retrieval/requirements.txt
```

A normal network-connected machine can use `python -m pip install -r requirements.txt`; the workaround changes only package transport, not installed artifacts.

### First pretrained smoke attempt and correction

`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.mplconfig" TQDM_MININTERVAL=5 ../../multimedia-course-env/bin/python -u scripts/run_experiment.py --smoke-objects 3 --max-queries 4 --batch-size 8 --device auto --output-dir results/smoke` initially failed before extraction. Modern PyTorch's `weights_only=True` rejected NumPy reconstruction metadata embedded in the original authors' checkpoint. The exact traceback is retained as `results/smoke_initial_failure.log`. Corrected the loader to verify the complete official checkpoint SHA-256 before legacy `weights_only=False` deserialization, which the previous implementation had used. No network or architecture substitution occurred. The same smoke command was rerun.

`pip check` after installation: **No broken requirements found**. `pip freeze` saved all 31 installed runtime/test packages to `requirements-lock.txt`.

## Smoke run and automated validation

- Smoke command above completed on **Apple MPS** at 2026-11-16 22:07 (Asia/Shanghai): 4 queries, 210 gallery images, mean P@5 **1.0000**. Descriptor extraction 3.3166 s; complete smoke pipeline 15.6178 s. These are smoke-only measurements.
- Checkpoint's learned GeM exponent: **2.9208486080169678**. Descriptors: 512 dimensions; mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; native 128×128 RGB.
- `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.mplconfig" ../../multimedia-course-env/bin/python -m pytest -q --junitxml=results/metrics/pytest.xml`: **18 passed, 0 skipped, 16 deprecation warnings in 9.32 s**. Included all real-data and official-model checks. Warnings come from the unmodified upstream torchvision construction and matplotlib/pyparsing; none caused failure. Full console output and JUnit XML retained.
- `../../multimedia-course-env/bin/python scripts/verify_results.py --results results/smoke --descriptors`: independently verified all 20 ranked rows, relevance, P@5, figures and exact descriptor rankings.
- Ordered dataset fingerprint: `8cc29268d9b56fba8f6b1787e8cd191b38e84e01a6dc937a4a4e3e438c5c1d3d`.

A report-only check launched from repository root initially could not import `src`. Added root `pytest.ini` with the Lab 1 Python path/test location so root and Lab 1 invocations resolve consistently. The report-only check then passed. The generated report now labels smoke runs prominently and records learned GeM p/source revision.

## Complete intended evaluation — 2026-11-16 22:09–22:10 Asia/Shanghai

From `1-image-retrieval/`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.mplconfig" TQDM_MININTERVAL=5 \
  ../../multimedia-course-env/bin/python -u scripts/run_experiment.py \
  --batch-size 16 --device mps
```

- Source commit: `4144793d2b766da136489e8a69a2e3ffd271379d`.
- **7,200 valid real images; 200 queries; 7,000 gallery images; 1,000 ranked result rows.**
- **Mean P@5 0.9970**, 198 queries at 1.0, `obj98__180` at 0.8 and `obj92__180` at 0.6; 0 queries at 0.0.
- Apple MPS, batch 16. Descriptor extraction 44.9552835 s; pipeline 54.715422125 s. Started 14:09:19.576607 UTC, finished 14:10:14.292053 UTC.
- Saved complete ranked CSV, per-query CSV, aggregate/config JSON, descriptors (local-only), case JSON, three figures, generated report draft and execution log.
- Inspected the lowest-P@5 image figure: query object 92 and false results from object 61 are white cylindrical bottles with pale caps and dense dark labeling. Correct ranks 1–3 are nearby angles. The observation supports discussing fine identity discrimination; no causal feature attribution is claimed.
- Copied actual figures into `reports/figures/`, prepared the Lab 1-only Markdown draft with all per-query values and a checked discussion. Final combined Word report remains deferred.
- Validation-stage Git push initially failed with HTTP 408; checked remote head (still `25d90eb`) and retried a normal fast-forward push with `git -c http.postBuffer=52428800 -c http.version=HTTP/1.1 push origin main`. No force push or history rewrite.

### Independent export verification and figure inspection

`../../multimedia-course-env/bin/python scripts/verify_results.py --descriptors` passed for **all 200 queries and 1,000 ranked rows**, including recomputed exact cosine rankings, relevance, scores, per-query/mean P@5 and figure decoding. Result CSV SHA-256: `994832090a5a1e23929cd2e9c44e9a2753857a3e073dce8cfabfc207682d03f3`.

Visually inspected the actual case figure and ten-example grid. The grid's overall title overlapped its first-row labels; adjusted only figure spacing and regenerated full/smoke grids from the saved actual CSV/manifest. No descriptors or numeric results changed. Copied refreshed figures into the report folder.

A complete post-evaluation suite passed 18 tests (0 skips), 16 dependency deprecation warnings in 9.49 s. One final suite after the spacing fix follows in the final validation record. Original prior-work inventory hashes all still match; Labs 2 and 3 contain only their placeholder README files. All report Markdown figure links resolve.

The validation-stage Git push retry succeeded: remote main advanced normally from `25d90eb` to `4144793`. Subsequent pushes use the successful HTTP/1.1/buffer settings without changing repository history.

## Final validation and approval checkpoint

After the figure spacing correction, reran the full suite with the same command:
**18 passed, 0 skipped, 16 dependency deprecation warnings in 9.24 s**. Final
console evidence is `results/final_test_execution.log` and JUnit evidence is
`results/metrics/final_pytest.xml`. Repeated the independent descriptor verifier:
**200 queries, 1,000 ranks, mean P@5 0.997**, all exact rankings and scores
verified. The result CSV SHA-256 remains unchanged.

Inspected the revised example grid: title and first-row IDs/scores are separated.
Report figures match the actual saved results and all local Markdown figure links
resolve. Original `../lab1/` hashes match the initial inventory. No data, weights,
source cache, descriptors, environment, secrets or raw conversation backup are
tracked. Labs 2 and 3 contain only placeholder README files.

The results/documentation stage `f6a3c4c` was pushed successfully. The final
`docs: finalize lab 1 validation and reproducibility` stage adds the disclosure,
current audit status, genuine conversation export helper and final test evidence.
See `git log --oneline --reverse` for the exact six stage commits. Push status
and the final remote HEAD are independently checked before the completion reply.

A genuine raw JSONL snapshot and visible-message extraction are preserved locally
under ignored `docs/ai_logs/`, with explicit cutoff and SHA-256 provenance. The
snapshot predates the completion response; refresh after this turn using the
command in `docs/ai_usage.md` or use Codex CLI `/export session.md` when supported.

**Lab 1 is implemented and validated, awaiting the user's explicit review.
No user approval has been inferred. Stop here; Lab 2/3 remain unauthorized.**

## Lab 2 — 2026-11-27

### Resources and compatibility

- Lab 2 started from a README placeholder. The isolated environment was
  created under `2-video-retrieval/.venv` with CPython 3.9.6 on macOS arm64.
- The official TVSum 1.1 archive was downloaded from MIT and hashed as
  `407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea`.
  All 50 original videos were inventoried, totaling 681,259,038 bytes.
- The original ViSiL model host failed over HTTP and HTTPS. The author's
  PyTorch mirror supplied the learned head, but the matching PCA file
  was hosted separately. The author's TensorFlow archive contained the
  complete pretrained ResNet inference graph and matching weights.
  [Checkpoint mirror](https://github.com/MKLab-ITI/visil/issues/24)
- The first PyAV installation attempt required a source build with
  `pkg-config` and FFmpeg development files. The pinned OpenCV and
  bundled FFmpeg wheels provided decoding and encoding.
- The first restore attempt failed with `No variables to save` because
  the legacy Saver executed outside the imported graph context.
  Restoring inside that context resolved the error. TensorFlow 2.16.2
  imported the published graph through `tf.compat.v1`, clearing the
  saved device assignments for CPU inference.

### Measured runs

| Profile | Inventoried originals | Queries | Gallery | Scored pairs | Top-3 hits | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 50 | 2 | 4 | 8 | 2/2 (100%) | 24.9 s |
| Full | 50 | 10 | 20 | 200 | 10/10 (100%) | 182.5 s |

The full run reused descriptors from the smoke profile for four gallery
videos and two queries. All 10 source originals ranked first. The full
gallery decode read 115,960 frames and sampled 4,163. The transformed
queries sampled 160 frames in total.

The standalone verifier checked 200 ranked rows, the exact Top-3
subset, source overlap labels, ties and the hit numerator and denominator.
The featured `98MoyGZKHXc` temporal matrix independently reproduced a
score of 0.9522336. Both figures were opened and visually inspected.
[Full verification](../2-video-retrieval/results/metrics/verification.json) ·
[Smoke verification](../2-video-retrieval/results/smoke/metrics/verification.json)

### Testing and rerun

Eight synthetic checks passed. The complete test suite passed 12 tests
with zero failures and zero skips, including actual TVSum inventory,
checkpoint restore, region descriptors, independent regional Chamfer
calculation, and saved smoke/full output verification. Matplotlib
reported pyparsing deprecations, and TensorFlow's imports reported
urllib3's warning about the system Python's LibreSSL. The launcher uses
curl for dataset and checkpoint transfers. [JUnit output](../2-video-retrieval/results/metrics/pytest.xml)

The README launcher was exercised for setup, smoke, full, test, and
both verification profiles. A full replay from the workspace parent
into an isolated output directory reproduced the rankings and exported
frame similarity values from the saved descriptors.
[Reproduction commands](../2-video-retrieval/README.md)

The Lab 2 work is ready for the student's local review. Lab 3 awaits
explicit approval of Lab 2.

## Lab 3 — 2026-11-27

### Resources and identity

- Created `3-video-summarization/.venv` with native CPython 3.9.6 and
  PyTorch 2.8.0 on macOS arm64. Reinstalled the exact lockfile in a
  separate fresh scratch environment and completed dataset preparation
  there. Dependency checks passed.
- Pinned DSNet to `1804176e2e8b57846beb063667448982273fca89`. The
  Dropbox sharing endpoint reset during the first checkpoint attempt.
  The direct-content host returned the published archive. Its ZIP
  directory identified `pretrain_af_basic/checkpoint/tvsum.yml.3.pt`,
  which was extracted and verified with CRC and SHA-256.
- The TVSum feature ZIP directory identified the 125,126,256-byte
  `eccv16_dataset_tvsum_google_pool5.h5` member. The exact member was
  obtained by HTTP range and verified by CRC and SHA-256. The launcher
  supports that member download, complete official archives and the
  author's Google Drive feature backup. [Official DSNet resources](https://github.com/li-plus/DSNet)
- Reused the complete TVSum original `EYqVtI9YWJA` and read the release
  archive to obtain its metadata and annotations. Its catalog index
  matches `video_42`. The HDF5 count, 20 official annotation lengths,
  and decoded video all match at 5,939 frames. The correlation between
  the per-pick mean official annotation and released HDF5 ground truth
  was 0.9999999999999982. This video is in fold 3's `test_keys`, used
  for validation by the authors.

### Execution and measured output

The official anchor-free model loaded the checkpoint strictly. Its
confidence and raw intervals matched the upstream `predict` method on
synthetic and real features. The full experiment evaluated all 396
feature positions, retained 35 detections after 0.4 IoU NMS, and scored
all 40 HDF5 shots. Knapsack selected 12 whole shots with no trimming.

| Measurement | Actual result |
|---|---:|
| Source frames and duration | 5,939 frames · 198.165 s |
| Frame budget | 890 |
| Summary decoded frames and duration | 890 frames · 29.696 s |
| Summary ratio | 14.9857% |
| MP4 audio streams | 0 |
| Decoded frames matched to source positions | 890 |

The initial synthetic export used passthrough timing. Its stream duration
was one frame short despite sequential presentation timestamps. Setting
the encoder to the measured source CFR corrected the duration. The
separate PyAV verifier decoded the actual MP4 and each expected source
position. It checked count, cadence, resolution, aspect and stream types,
and measured a maximum downsampled luma error of 0.459 on the 8-bit scale.
Repeated launcher runs produced the same MP4 SHA-256. The figures were
opened to inspect the decoded examples and model timeline.
[Run data](../3-video-summarization/results/metrics/experiment_metrics.json) ·
[Verification](../3-video-summarization/results/metrics/verification.json)

### Tests and artifacts

The first complete test run passed 18 of 19 tests. The remaining synthetic
assertion expected a later solution for two equally optimal knapsack
packs. The exhaustive case now has a unique optimum, and a separate
assertion verifies the documented earlier-shot tie rule. The final run
passed **19 tests (16 synthetic and 3 integration)** with zero failures
and zero skips. The real integration test exported and validated the
complete video in a separate output directory. Matplotlib emitted
pyparsing deprecation warnings. [JUnit results](../3-video-summarization/results/metrics/pytest.xml)

The local MP4 is at `3-video-summarization/results/summary.mp4`.
The CSV and JSON results, four figures, report draft, and source code
are available on the Lab 3 review branch. The combined Word report
awaits the student's explicit Lab 3 approval.
[Report draft](../reports/lab3_report.md) · [Reproduction instructions](../3-video-summarization/README.md)
