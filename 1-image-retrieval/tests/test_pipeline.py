"""Numerical and real-asset checks; synthetic unit inputs are not accuracy results."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import build_manifest, write_manifest, read_manifest
from src.evaluation import evaluate_precision_at_k
from src.features import checkpoint_transform, extract_features
from src.model import load_retrieval_model, OFFICIAL_REPOSITORY_REVISION
from src.retrieval import rank_top_k
from src.visualization import plot_case_analysis, plot_precision, plot_retrieval_examples

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_records():
    data = ROOT / 'data/coil-100'
    if not data.exists():
        pytest.skip('Prepare real COIL-100 data to run integration checks')
    return build_manifest(data, strict_dataset=True, validate_content=True)


@pytest.fixture(scope="module")
def real_model():
    weights = ROOT / 'weights/retrievalSfM120k-vgg16-gem.pth'
    source = ROOT / 'vendor' / ('cnnimageretrieval-pytorch-' + OFFICIAL_REPOSITORY_REVISION)
    if not weights.is_file() or not source.is_dir():
        pytest.skip('Download official weights and pinned source to run model checks')
    return load_retrieval_model(weights, torch.device('cpu'))


def test_preprocessing_preserves_rgb_shape_and_checkpoint_normalization():
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    transform = checkpoint_transform({'mean': [.5, .25, .1], 'std': [.5, .5, .2]})
    x = transform(Image.fromarray(pixels))
    assert x.shape == (3, 128, 128)
    assert x.dtype == torch.float32
    torch.testing.assert_close(x[:, 0, 0], torch.tensor([1., -.5, -.5]))
    assert torch.isfinite(x).all()


def test_real_dataset_integrity_and_portable_manifest(real_records, tmp_path):
    assert len(real_records) == 7200
    assert sum(r.split == 'query' for r in real_records) == 200
    assert sum(r.split == 'gallery' for r in real_records) == 7000
    assert {r.object_id for r in real_records} == set(range(1, 101))
    manifest = tmp_path / 'manifest.csv'
    write_manifest(real_records, manifest)
    assert read_manifest(manifest) == real_records
    assert not any(Path(line.split(',')[2]).is_absolute()
                   for line in manifest.read_text().splitlines()[1:])


def test_official_gem_matches_power_mean_and_differs_from_average(real_model):
    from cirtorch.layers.pooling import GeM
    x = torch.tensor([[[[0., 1.], [2., 7.]]]])
    pool = GeM(p=3.)
    expected = x.clamp_min(1e-6).pow(3.).mean(dim=(-2, -1), keepdim=True).pow(1 / 3)
    torch.testing.assert_close(pool(x), expected)
    assert not torch.allclose(pool(x), x.mean(dim=(-2, -1), keepdim=True))
    torch.testing.assert_close(GeM(p=1.)(x), x.clamp_min(1e-6).mean(dim=(-2, -1), keepdim=True))
    model, meta, digest = real_model
    assert model.pool.__class__.__name__ == 'GeM'
    assert meta['gem_p'] > 1 and len(digest) == 64


def test_real_pretrained_descriptors_are_512_finite_and_normalized(real_model, real_records):
    model, meta, _ = real_model
    records = real_records[:2]
    ids, descriptors = extract_features(records, model, meta, torch.device('cpu'), 2)
    assert ids == [r.image_id for r in records]
    assert descriptors.shape == (2, 512)
    assert np.isfinite(descriptors).all()
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1, atol=1e-6)
    assert not np.allclose(descriptors[0], descriptors[1])


def test_cosine_scores_do_not_depend_on_vector_magnitude():
    from src.dataset import ImageRecord
    q = ImageRecord('q', 'q.png', '/q', 1, 0, 'query')
    g = [ImageRecord('g', 'g.png', '/g', 1, 5, 'gallery')]
    rows = rank_top_k([q], g, np.array([[3., 4.]]), np.array([[0., 20.]]), k=1)
    assert rows[0]['similarity'] == pytest.approx(.8)


def test_query_overlap_is_rejected(real_records):
    record = real_records[0]
    with pytest.raises(ValueError, match='query image is present'):
        rank_top_k([record], [record], np.ones((1, 2)), np.ones((1, 2)), k=1)


def test_invalid_metric_rows_are_rejected():
    with pytest.raises(ValueError, match='exactly ranks'):
        evaluate_precision_at_k([{'query_id': 'q', 'rank': 1, 'relevant': True}])
    with pytest.raises(ValueError, match='boolean'):
        evaluate_precision_at_k([{'query_id': 'q', 'rank': 1, 'relevant': 'False'}], k=1)


def test_figures_render_actual_coil_images(real_records, tmp_path):
    query = next(r for r in real_records if r.split == 'query')
    gallery = [r for r in real_records if r.object_id == query.object_id and r.split == 'gallery'][:5]
    # Fixed descriptors test plotting mechanics, not retrieval accuracy.
    rows = rank_top_k([query], gallery, np.ones((1, 2)), np.ones((5, 2)))
    metrics = evaluate_precision_at_k(rows)
    records = {r.image_id: vars(r) for r in [query] + gallery}
    plot_retrieval_examples(records, rows, tmp_path / 'examples.png', count=1)
    plot_precision(metrics, tmp_path / 'precision.png')
    case = plot_case_analysis(records, rows, {query.image_id: 1.}, tmp_path / 'case.png')
    assert case['query_id'] == query.image_id
    for name in ['examples.png', 'precision.png', 'case.png']:
        with Image.open(tmp_path / name) as image:
            assert image.width > 500 and image.height > 100
            image.verify()
