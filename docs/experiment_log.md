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
