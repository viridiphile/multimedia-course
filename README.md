# Multimedia course laboratories

Repository for the individual Multimedia assignments. Current authorized stage is **Lab 3 — video summarization using DSNet on TVSum**.

- [Lab 1](1-image-retrieval/README.md): implementation, tests, actual measurements and figures.
- [Lab 2](2-video-retrieval/README.md) includes the CPU launcher, TVSum protocol, scores, and similarity figures.
- [Lab 3](3-video-summarization/README.md) includes the verified extractive summary, frame budget, and launcher.
- `reports/` contains the [Lab 1](reports/lab1_report.md), [Lab 2](reports/lab2_report.md), and [Lab 3](reports/lab3_report.md) drafts.
- `docs/`: implementation audit, experiment log, AI assistance provenance.

The combined Word report awaits explicit approval of Lab 3.

## Lab 1 result — awaiting explicit review

All **7,200** COIL-100 images were validated; **200** query views were ranked
against **7,000** gallery views using the authors' VGG16-GeM checkpoint.
**Mean P@5: 0.9970**; 198 queries achieved 1.0. The independent verifier checked
all 1,000 returned positions and exact cosine rankings. All 18 tests passed
with real-asset checks included. See [Lab 1 report draft](reports/lab1_report.md),
[experiment summary](1-image-retrieval/results/summary.md),
[experiment log](docs/experiment_log.md) and [AI records](docs/ai_usage.md).

![Lowest-P@5 case](1-image-retrieval/results/figures/case_analysis.png)

## Lab 2 result — awaiting explicit review

ViSiL retrieved the overlapping original at rank 1 for all 10 transformed
clips against a 20-video TVSum gallery. **Top-3 Hit Rate was 10/10 (100%)**.
All 12 checks passed. See the [Lab 2 report draft](reports/lab2_report.md),
[Top-3 rankings](2-video-retrieval/results/metrics/top3.csv), and
[frame similarity figure](2-video-retrieval/results/figures/frame_similarity.png).

## Lab 3 result — awaiting explicit review

Anchor-free DSNet retained **890 of 5,939 frames (14.9857%)** in twelve
selected shots from one complete TVSum video. The decoded summary runs
for 29.696 s and has no audio stream. See the
[Lab 3 report draft](reports/lab3_report.md) and the
[measured timeline](3-video-summarization/results/figures/selected_timeline.png).
