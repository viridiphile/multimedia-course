# Lab 3 — Video summarization with anchor-free DSNet

The complete TVSum original *Smage Bros. Motorcycle Stunt Show*
(`EYqVtI9YWJA`) contains 5,939 frames over 198.165 s. The pretrained
model selected 12 shots. The local MP4 contains **890 frames over
29.696 s (14.9857%)**, with one H.264 video stream and zero audio streams.

## Run on Apple Silicon macOS

Run these commands from the repository root.

```bash
./run-lab3.sh setup
./run-lab3.sh
./run-lab3.sh verify
./run-lab3.sh test
```

The launcher uses `3-video-summarization/.venv/bin/python` and resolves
paths relative to its own location. The measured environment was native
macOS arm64 with Python 3.9.6, PyTorch 2.8.0 on CPU, PyAV 15.1.0, and
the FFmpeg bundled with `imageio-ffmpeg`. Set `LAB3_PYTHON` to a native
Python 3.9–3.12 interpreter when creating the environment if needed.
The launcher checks the lockfile before installing dependencies.

### Manual setup

```bash
cd 3-video-summarization
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-lock.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
.venv/bin/python -m scripts.prepare_dataset
.venv/bin/python -m scripts.run_experiment
.venv/bin/python -m scripts.verify_results
.venv/bin/python -m pytest
```

### Resources and cached media

Setup reuses the original TVSum file and release archive when they are
available in the earlier lab's data cache. The only shared inputs are the
original media and dataset archive. With an empty cache, it obtains
TVSum 1.1 from MIT and extracts the two metadata files and the single
original. The HDF5 and fold-3 checkpoint ZIP members are fetched from
the authors' archives. Cached assets are reused, and file hashes identify
the exact used release data.

If automatic download fails, place the official `tvsum50_ver_1_1.tgz`
and `dsnet_datasets.zip` under Lab 3's `data/` and
`pretrain_af_basic.zip` under `weights/`, then run setup again. Setup
also accepts the exact extracted HDF5 file at
`data/eccv16_dataset_tvsum_google_pool5.h5` and checkpoint at
`weights/tvsum.yml.3.pt`. [TVSum release](https://people.csail.mit.edu/yalesong/tvsum/) ·
[DSNet resources](https://github.com/li-plus/DSNet#datasets-preparation)

## Model and frame selection

Official TVSum catalog entry 42 maps to HDF5 group `video_42`. Setup
checks the 5,939-frame identity against a full decode and 20 official
annotation traces, and matches the sampled average annotation to HDF5
ground truth. The 396 input vectors are 1,024-dimensional GoogLeNet
pool5 features with picks every 15 original frames, from 0 to 5,925.
Inference consumes the complete feature sequence.

The authors' basic anchor-free DSNet uses eight attention heads and a
128-unit hidden layer. The pinned source revision is
`1804176e2e8b57846beb063667448982273fca89`. The checkpoint is
`pretrain_af_basic/checkpoint/tvsum.yml.3.pt` (zero-based fold 3).
The video is in that fold's `test_keys`, which the authors use for
checkpoint validation. [DSNet revision](https://github.com/li-plus/DSNet/tree/1804176e2e8b57846beb063667448982273fca89)

The pipeline multiplies predicted class probability and centerness,
normalizes the scores, and applies temporal NMS at IoU 0.4 to rounded
and clipped offsets. Each surviving detection contributes its score
to the feature bins it covers. The maximum score per bin is expanded
to source frames using the picks, then averaged within the 40 released
shot ranges. Zero-one knapsack selects whole shots under
`floor(0.15 × 5,939) = 890` frames. Saved start indices are inclusive
and end indices are exclusive, with timestamps in seconds. FFmpeg
extracts those intervals in source order at 30,000/1,001 fps.

## Local output and verification

- The playable MP4 is `3-video-summarization/results/summary.mp4`.
- [Run metrics](results/metrics/experiment_metrics.json),
  [CSV summary](results/metrics/run_summary.csv), and
  [selected segments](results/metrics/selected_segments.csv) include
  frame counts, timing, scores, and model signature.
- [Raw predictions and projected scores](results/metrics/predictions.csv),
  [NMS detections](results/metrics/nms_detections.csv), and
  [shot scores](results/metrics/shot_scores.csv) expose the selection inputs.
- [Timeline](results/figures/selected_timeline.png),
  [decoded summary frames](results/figures/summary_frames.png),
  [source comparison](results/figures/original_vs_summary.png), and
  [model scores](results/figures/dsnet_scores.png) are embedded in the
  [Lab 3 report draft](../reports/lab3_report.md).

The verifier checks the saved intervals, frame count, playback rate,
resolution and audio absence, then compares each decoded output frame
to its corresponding decoded source frame. Synthetic tests run with
`./run-lab3.sh test -m 'not integration'`. End-to-end integration
checks use the complete video and published checkpoint.
[Verification output](results/metrics/verification.json) ·
[Automated test results](results/metrics/pytest.xml)
