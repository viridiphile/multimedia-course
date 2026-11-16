#!/usr/bin/env python3
"""Independently recompute saved P@5 and optionally all exact descriptor rankings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_manifest


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--descriptors', action='store_true', help='Also verify rankings using ignored local NPZ')
    args = parser.parse_args()
    root = args.results
    metrics = json.loads((root / 'metrics/experiment_metrics.json').read_text())
    config = metrics['configuration']
    # The canonical manifest remains at results/metrics for smoke runs too.
    manifest = Path('results/metrics/coil100_manifest.csv')
    records = {r.image_id: r for r in read_manifest(manifest)}
    results_file = root / 'metrics/top5_results.csv'
    with results_file.open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(row['query_id'], []).append(row)
    values = {}
    for query_id, results in grouped.items():
        query = records[query_id]
        check(query.split == 'query', 'Query must be in query split')
        check([int(r['rank']) for r in results] == [1, 2, 3, 4, 5], 'Ranks must be 1..5')
        check(len({r['image_id'] for r in results}) == 5, 'Duplicate result IDs')
        scores = [float(r['similarity']) for r in results]
        check(all(np.isfinite(s) and -1.000001 <= s <= 1.000001 for s in scores), 'Invalid scores')
        check(scores == sorted(scores, reverse=True), 'Scores not descending')
        relevant = 0
        for row in results:
            gallery = records[row['image_id']]
            check(gallery.split == 'gallery' and gallery.image_id != query_id, 'Query leakage')
            actual = query.object_id == gallery.object_id
            check(row['relevant'] in ('True', 'False'), 'Invalid boolean encoding')
            check((row['relevant'] == 'True') == actual, 'Relevance mismatch')
            check(int(row['object_id']) == gallery.object_id and int(row['angle']) == gallery.angle, 'Gallery metadata mismatch')
            check(int(row['query_object_id']) == query.object_id and int(row['query_angle']) == query.angle, 'Query metadata mismatch')
            relevant += actual
        values[query_id] = relevant / 5
    saved = {r['query_id']: r['precision_at_5'] for r in metrics['precision_at_5_per_query']}
    check(saved == values, 'JSON query precision mismatch')
    with (root / 'metrics/per_query_precision.csv').open() as handle:
        csv_values = {r['query_id']: float(r['precision_at_5']) for r in csv.DictReader(handle)}
    check(csv_values == values, 'CSV query precision mismatch')
    mean = sum(values.values()) / len(values)
    check(abs(mean - metrics['mean_precision_at_5']) < 1e-12, 'Mean P@5 mismatch')
    check(len(values) == metrics['queries_evaluated'] == config['query_count'], 'Query count mismatch')
    if not config['is_smoke_run']:
        check(len(values) == 200 and config['gallery_count'] == 7000, 'Incomplete final evaluation')
        check(set(values) == {k for k,r in records.items() if r.split == 'query'}, 'Missing final queries')
    if args.descriptors:
        with np.load(root / 'descriptors.npz', allow_pickle=False) as data:
            q, g = data['query_features'], data['gallery_features']
            qi, gi = data['query_ids'].tolist(), data['gallery_ids'].tolist()
        check(q.shape == (len(qi), 512) and g.shape == (config['gallery_count'], 512), 'Descriptor shape mismatch')
        check(set(qi) == set(values) and len(gi) == len(set(gi)), 'Descriptor ID mismatch')
        for matrix in [q,g]:
            check(np.isfinite(matrix).all(), 'Nonfinite descriptors')
            check(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6), 'Nonunit descriptors')
        scores = (q / np.linalg.norm(q,axis=1,keepdims=True)) @ (g / np.linalg.norm(g,axis=1,keepdims=True)).T
        for i, query_id in enumerate(qi):
            indexes = np.lexsort((np.array(gi), -scores[i]))[:5]
            actual_rows = grouped[query_id]
            check([gi[j] for j in indexes] == [r['image_id'] for r in actual_rows], 'Exact ranking mismatch')
            check(np.allclose(scores[i,indexes], [float(r['similarity']) for r in actual_rows], atol=1e-6), 'Exact score mismatch')
    for name in ['retrieval_examples.png', 'precision_at_5.png', 'case_analysis.png']:
        with Image.open(root / 'figures' / name) as image:
            image.verify()
    outcome = {'queries_checked': len(values), 'ranked_rows_checked': len(rows),
               'mean_precision_at_5': mean, 'exact_descriptor_ranking_checked': args.descriptors,
               'results_csv_sha256': hashlib.sha256(results_file.read_bytes()).hexdigest()}
    (root / 'metrics/verification.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    main()
