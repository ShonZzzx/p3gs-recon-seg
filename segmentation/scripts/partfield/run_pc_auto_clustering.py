import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def load_ply_xyz(path):
    ply = PlyData.read(path)
    vertex = ply['vertex']
    return np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)


def write_labeled_ply(points, labels, path):
    labels = np.asarray(labels).reshape(-1)
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique))
    color_map = {label: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique)}
    colors = np.array([color_map[label] for label in labels], dtype=np.uint8)

    vertices = np.empty(
        points.shape[0],
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
        ],
    )
    vertices['x'], vertices['y'], vertices['z'] = points[:, 0], points[:, 1], points[:, 2]
    vertices['red'], vertices['green'], vertices['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(path)


def sample_rows(x, max_rows, rng):
    if x.shape[0] <= max_rows:
        return x
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


def select_k(feat, k_min, k_max, select_sample, score_sample, seed):
    rng = np.random.default_rng(seed)
    sample = sample_rows(feat, select_sample, rng)
    max_valid_k = min(k_max, sample.shape[0] - 1)
    scores = []

    for k in range(k_min, max_valid_k + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(8192, max(1024, sample.shape[0])),
            max_iter=80,
            n_init=3,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(sample)
        if np.unique(labels).size < 2:
            continue
        score = silhouette_score(
            sample,
            labels,
            metric='euclidean',
            sample_size=min(score_sample, sample.shape[0]),
            random_state=seed,
        )
        scores.append((k, float(score), float(model.inertia_)))
        print(f'  k={k:02d} silhouette={score:.4f} inertia={model.inertia_:.2f}', flush=True)

    if not scores:
        return k_min, []
    best_k, _, _ = max(scores, key=lambda item: item[1])
    return best_k, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=30)
    parser.add_argument('--select-sample', type=int, default=30000)
    parser.add_argument('--score-sample', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--max-iter', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    features_dir = Path(args.features)
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    label_dir = out_dir / 'cluster_out'
    ply_dir = out_dir / 'ply'
    label_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(features_dir.glob('part_feat_*_0.npy'))
    print(f'Number of feature files: {len(feature_files)}')
    rows = []

    for feat_path in feature_files:
        uid = feat_path.stem[len('part_feat_'):-len('_0')]
        ply_path = source_dir / f'{uid}.ply'
        if not ply_path.exists():
            print(f'Missing source PLY for {uid}: {ply_path}')
            continue

        print(f'Processing {uid}', flush=True)
        feat = np.load(feat_path).astype(np.float32)
        norm = np.linalg.norm(feat, axis=-1, keepdims=True)
        feat = feat / np.maximum(norm, 1e-8)
        points = load_ply_xyz(ply_path)

        if points.shape[0] != feat.shape[0]:
            n = min(points.shape[0], feat.shape[0])
            print(f'Warning: {uid} point/feature mismatch {points.shape[0]} vs {feat.shape[0]}, truncating to {n}')
            points = points[:n]
            feat = feat[:n]

        best_k, scores = select_k(
            feat,
            args.k_min,
            args.k_max,
            args.select_sample,
            args.score_sample,
            args.seed,
        )
        print(f'  selected k={best_k}', flush=True)

        model = MiniBatchKMeans(
            n_clusters=best_k,
            random_state=args.seed,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            n_init=5,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(feat).astype(np.int32)
        stem = f'{uid}_0_auto_k{best_k:02d}'
        np.save(label_dir / f'{stem}.npy', labels.reshape(-1, 1))
        write_labeled_ply(points, labels, ply_dir / f'{stem}.ply')

        rows.append({
            'uid': uid,
            'n_points': int(feat.shape[0]),
            'selected_k': int(best_k),
            'best_silhouette': max([s[1] for s in scores], default=float('nan')),
            'scores_json': json.dumps([
                {'k': k, 'silhouette': score, 'inertia': inertia}
                for k, score, inertia in scores
            ]),
        })

    with (out_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['uid', 'n_points', 'selected_k', 'best_silhouette', 'scores_json'])
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote ' + str(out_dir / 'summary.csv'))


if __name__ == '__main__':
    main()
