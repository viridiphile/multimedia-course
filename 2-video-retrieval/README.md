# Lab 2 — Video retrieval with ViSiL

Rank original TVSum videos against transformed clips using the authors'
pretrained ViSiL model. The full profile uses 10 queries and 20 gallery
videos. The smoke profile uses 2 queries and 4 gallery videos.

## Run on Apple Silicon macOS

Run these commands from the root of the multimedia-course checkout.

```bash
./2-video-retrieval/run-lab2.sh setup
./2-video-retrieval/run-lab2.sh smoke
./2-video-retrieval/run-lab2.sh full
./2-video-retrieval/run-lab2.sh test
./2-video-retrieval/run-lab2.sh verify results/smoke
./2-video-retrieval/run-lab2.sh verify results
```

The launcher always enters the Lab 2 directory and uses
`2-video-retrieval/.venv/bin/python`. Run it by its quoted absolute path
from any other directory. Python 3.9.6 on macOS arm64 is the tested setup.
For a new environment, set `LAB2_PYTHON` to the path of a compatible native
Python 3.9–3.12 interpreter if `python3` resolves elsewhere. The locked wheels
include OpenCV and a bundled FFmpeg, which handles the query encoding.
Inference runs on CPU with batches of 16 frames.

`setup` prepares assets ahead of time. `smoke` and `full` also prepare
missing resources. The first run needs network access to PyPI, the MIT
TVSum archive (641 MiB), and the author's Google Drive model mirror
(342 MiB). Downloads stay in `data/` and `weights/`, and descriptor caches
stay in `.cache/`. Keep approximately 4 GiB free for the environment,
archives, extracted assets, clips, and descriptors. The launcher checks
installed versions before running pip.

If a download fails, place the published downloads at
`2-video-retrieval/data/tvsum50_ver_1_1.tgz` and
`2-video-retrieval/weights/visil-tf.zip`, then rerun setup. The setup
verifies both archive SHA-256 values. An already extracted dataset can be
placed at `data/tvsum/` with its `info.tsv` and original files in `videos/`.
An already extracted ResNet checkpoint can be placed at `weights/resnet/`
with `model.meta`, `model.index`, and `model.data-00000-of-00001`.
Each checkpoint file is checked against the release digest before inference.
[TVSum downloads](https://people.csail.mit.edu/yalesong/tvsum/) ·
[Author's checkpoint mirror](https://github.com/MKLab-ITI/visil/issues/24)

## Protocol

The manifest includes the ID, category, title, URL, dimensions, average
frame rate, nominal duration, encoded frame count, file size, and SHA-256
for all 50 originals. OpenCV verifies that each file opens and decodes a
frame. For the gallery and queries, sampling traverses each video and
records the decoded and sampled frame counts.

The gallery uses the two lexicographically first IDs in each TVSum category.
The first of those IDs supplies the query. The smoke profile uses categories
BK and BT. Each query is a 16-second interior clip beginning at
`round(fraction × (source_duration − 16))`, where the fraction cycles
through 0.25, 0.45, and 0.65 across alphabetical categories. FFmpeg
removes approximately the left 8% of the image, resizes to 240 pixels high,
and encodes muted H.264 at CRF 26. Each corresponding original is the
one relevant candidate. The second original per category provides a
same-category negative, and other categories add more negatives.

RGB frames are sampled every `round(fps)` decoded frames starting at frame
zero, yielding approximately 1 Hz. ViSiL applies the saved graph's RGB
resize, crop, and mean subtraction, ResNet-50 regional pooling, PCA
whitening and regional attention. For each query–candidate pair, the
graph computes directional regional Chamfer similarity, processes the
frame similarity matrix with its trained four-convolution temporal network,
clips to [−1, 1], and averages the maximum over candidate time for each
downsampled query row. Scores are sorted descending, with equal scores
ordered by candidate ID.

The model runs the author's serialized ResNet inference graph and weights
from `ckpt/resnet/model`. TensorFlow 2.16.2 restores the TensorFlow 1 graph
through `tf.compat.v1`, clears device assignments, and runs in CPU mode.
The reference source reviewed was
`0971e54fb8325fceb1bc9748ecbfe4c66e5dabd2`; the serialized graph and
checkpoint files govern the exact deployed computation.
[ViSiL reference code](https://github.com/MKLab-ITI/visil/tree/0971e54fb8325fceb1bc9748ecbfe4c66e5dabd2) ·
[ICCV 2019 ViSiL paper](https://arxiv.org/abs/1908.07410)

## Results and checks

The full run measured **10/10 (100%) Top-3 Hit Rate**, with every source
ranked first. Smoke measured **2/2 (100%)**. All 12 tests passed.

- [Top-3 CSV](results/metrics/top3.csv) identifies ranks, scores, relevance and
  whether each query hit. [All rankings](results/metrics/all_rankings.csv)
  includes every scored pair.
- [Metrics](results/metrics/experiment_metrics.json),
  [manifest](results/metrics/tvsum_manifest.csv), and
  [protocol](results/metrics/protocol.json) record the run and its inputs.
- [Top-3 examples](results/figures/top3_examples.png) show query frames and
  each ranked video's highest-scoring raw matches for those frames.
  [Similarity figure](results/figures/frame_similarity.png) shows the
  actual raw and learned temporal matrices. The numerical arrays are
  in [heatmap_pair.json](results/metrics/heatmap_pair.json).
- [Report draft](../reports/lab2_report.md) interprets the measured results,
  scores, and figures. Smoke results are under [results/smoke](results/smoke).

Use `test -m 'not integration'` with the launcher for synthetic checks and
`test -m integration` for real TVSum, checkpoint, and output checks. The
standalone verifier reconstructs hit rate from the saved rankings and
recomputes the featured video score from the exported temporal matrix.
