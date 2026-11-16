# Technical notes

## Descriptor and retrieval method

The model is the authors' `retrievalSfM120k-vgg16-gem` network trained for
image retrieval on retrieval-SfM-120k. It is not torchvision's generic
ImageNet classifier. The checkpoint's metadata is used to build the matching
authors' architecture and to obtain the input channel mean and standard
deviation. No training or fine-tuning is performed.

COIL-100 files are opened as RGB and converted to tensors, then normalized with
the checkpoint's mean/std. The model is evaluated without gradients. Its
output is L2-normalized; cosine similarity is then a matrix product of the
normalized query and gallery descriptors. Sorting is descending by score with
image ID as a deterministic tie-break. The query set is never part of the
gallery.

## Protocol and interpretation

The 0°/180° viewpoints are held out as queries for every object. Relevance is
based solely on shared object identity, so a correct result need not have a
similar viewing angle. This is a fixed retrieval benchmark, not a train/val/test
learning experiment. P@5 is the number of same-object results among the five
nearest gallery images, divided by five; the mean is computed over the actual
number of evaluated queries.

The model was trained on landmark/retrieval imagery rather than COIL-100's
controlled turntable views. Its descriptor may therefore rank visually similar
objects above a strongly changed view of the same object. The generated report
is populated from the saved experiment outputs; any qualitative explanation
must be checked against the generated real-image case figure.

## Reproducibility and boundaries

The dataset split and result ordering are deterministic. The experiment JSON
records the checkpoint identifier, device, batch size, preprocessing metadata,
input/output dimensions, and counts. The project does not bundle COIL-100,
checkpoint files, or extracted feature caches.

The authors' upstream implementation targets older PyTorch versions. This
project avoids its CUDA-only extraction helper and calls the official network
directly, allowing CPU and MPS inference. If a future torchvision release
breaks construction of the upstream VGG network, the run fails with a surfaced
error; no fallback architecture is substituted.


## Current-session corrections

Official Columbia HTTPS download replaces optional undeclared mirror clients.
The pinned unmodified upstream source is cached under local `vendor/`.
Checkpoint deserialization uses `weights_only=True`; strict loading rejects
incompatible architectures. The learned GeM exponent and exact dataset fingerprint
are saved. All source data and manifest labels are checked before extraction.
Smoke output is kept separate. `verify_results.py` independently recomputes exact
rankings and P@5 from saved descriptors; descriptors remain local and ignored.

No multi-scale aggregation or additional learned whitening is applied. Native
128×128 resolution is an explicit difference from resized evaluation settings.
CPU/MPS numerical behavior can differ slightly; the actual execution device is
recorded rather than assuming bitwise cross-platform reproducibility.
