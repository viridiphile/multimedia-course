# Lab 1 measured experiment summary

Final evaluation: **200 queries × 7,000 gallery images**, all 7,200 real COIL-100
images validated. Fixed query angles: 0°/180°; relevance: shared object ID.

Model: authors' retrievalSfM120k VGG16-GeM, learned p=2.9208486080169678,
512-dimensional normalized descriptors; native 128×128 RGB; cosine ranking.

**Mean P@5: 0.9970 (99.70%)**. 198 queries scored 1.0, one scored 0.8
(`obj98__180`), one scored 0.6 (`obj92__180`). Exactly 997/1,000 returned
positions were relevant. No query was present in its gallery.

Apple MPS, batch 16; extraction 44.9553 s; full pipeline 54.7154 s.
All per-query values and ranked IDs/scores/relevance are saved under `metrics/`.
Three figures and the generated report draft accompany the numerical outputs.
The analyzed failure confuses two visually similar white-bottle object IDs.

Dense neighboring views make this a favorable retrieval protocol; these results
do not establish performance with large view gaps or unfamiliar objects.

See `../README.md`, `../../reports/lab1_report.md`, and `metrics/verification.json`.
