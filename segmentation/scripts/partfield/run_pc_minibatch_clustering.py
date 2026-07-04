import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans


def load_ply_xyz(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)


def write_labeled_ply(points, labels, path):
    labels = np.asarray(labels).reshape(-1)
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique))
    color_map = {label: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique)}
    colors = np.array([color_map[label] for label in labels], dtype=np.uint8)

    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 20])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    features_dir = Path(args.features)
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    label_dir = out_dir / "cluster_out"
    ply_dir = out_dir / "ply"
    label_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(features_dir.glob("part_feat_*_0.npy"))
    print(f"Number of feature files: {len(feature_files)}")

    for feat_path in feature_files:
        uid = feat_path.stem[len("part_feat_"):-len("_0")]
        ply_path = source_dir / f"{uid}.ply"
        if not ply_path.exists():
            print(f"Missing source PLY for {uid}: {ply_path}")
            continue

        feat = np.load(feat_path).astype(np.float32)
        norm = np.linalg.norm(feat, axis=-1, keepdims=True)
        feat = feat / np.maximum(norm, 1e-8)
        points = load_ply_xyz(ply_path)

        if points.shape[0] != feat.shape[0]:
            n = min(points.shape[0], feat.shape[0])
            print(f"Warning: {uid} point/feature mismatch {points.shape[0]} vs {feat.shape[0]}, truncating to {n}")
            points = points[:n]
            feat = feat[:n]

        for k in args.ks:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=0,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                n_init=3,
                reassignment_ratio=0.01,
            )
            labels = model.fit_predict(feat).astype(np.int32)
            stem = f"{uid}_0_{k:02d}"
            np.save(label_dir / f"{stem}.npy", labels.reshape(-1, 1))
            write_labeled_ply(points, labels, ply_dir / f"{stem}.ply")
            print(f"Saved {stem}")


if __name__ == "__main__":
    main()
