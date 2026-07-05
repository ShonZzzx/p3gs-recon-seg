#!/usr/bin/env python3
"""Compute annotation-free plant reconstruction metrics for PLY files."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = [
    "name",
    "ply_path",
    "render_dir",
    "point_count",
    "outlier_ratio",
    "main_component_ratio",
    "bbox_volume",
    "local_thickness_median",
    "render_edge_strength",
]


def read_ply_xyz(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)


def vertex_count_from_header(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise RuntimeError(f"element vertex not found: {path}")


def sample_points(xyz: np.ndarray, sample_n: int, seed: int) -> np.ndarray:
    if len(xyz) <= sample_n:
        return xyz
    rng = np.random.default_rng(seed)
    return xyz[rng.choice(len(xyz), size=sample_n, replace=False)]


def mad_outlier_ratio(xyz: np.ndarray, k: int, sample_n: int, seed: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return 0.0
    sample = sample_points(xyz, sample_n, seed)
    nbr = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xyz)
    dist, _ = nbr.kneighbors(sample)
    mean_dist = dist[:, 1:].mean(axis=1)
    med = np.median(mean_dist)
    mad = np.median(np.abs(mean_dist - med))
    if mad == 0:
        return 0.0
    threshold = med + 3.0 * mad
    return float(np.mean(mean_dist > threshold))


def main_component_ratio(xyz: np.ndarray, sample_n: int, seed: int) -> float:
    from sklearn.cluster import DBSCAN

    if len(xyz) == 0:
        return math.nan
    sample = sample_points(xyz, sample_n, seed)
    span = np.linalg.norm(sample.max(axis=0) - sample.min(axis=0))
    eps = max(span * 0.02, 1e-6)
    labels = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit_predict(sample)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return 0.0
    counts = np.bincount(valid)
    return float(counts.max() / len(labels))


def bbox_volume(xyz: np.ndarray) -> float:
    if len(xyz) == 0:
        return 0.0
    extent = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 0.0)
    return float(np.prod(extent))


def local_thickness_median(xyz: np.ndarray, k: int, sample_n: int, seed: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return math.nan
    sample = sample_points(xyz, sample_n, seed)
    nbr = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    _, idx = nbr.kneighbors(sample)
    vals = []
    for group in idx:
        pts = xyz[group]
        cov = np.cov((pts - pts.mean(axis=0)).T)
        eigvals = np.linalg.eigvalsh(cov)
        vals.append(max(eigvals[0], 0.0) ** 0.5)
    return float(np.median(vals))


def render_edge_strength(render_dir: Path | None) -> float | None:
    if render_dir is None:
        return None
    from skimage import filters

    vals = []
    for path in sorted(render_dir.glob("*.png")):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        sobel = filters.sobel(img)
        lap = np.abs(filters.laplace(img))
        vals.append(float((sobel.mean() + lap.mean()) * 0.5))
    return float(np.mean(vals)) if vals else math.nan


def parse_item(text: str) -> tuple[str, Path, Path | None]:
    parts = text.split("|", 2)
    if len(parts) < 2:
        raise ValueError("--item format: name|/path/to/model.ply[|/path/to/renders]")
    name = parts[0]
    ply = Path(parts[1])
    render_dir = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    return name, ply, render_dir


def compute_one(name: str, ply: Path, render_dir: Path | None, args: argparse.Namespace) -> dict:
    xyz = read_ply_xyz(ply)
    return {
        "name": name,
        "ply_path": str(ply),
        "render_dir": str(render_dir) if render_dir else "",
        "point_count": vertex_count_from_header(ply),
        "outlier_ratio": mad_outlier_ratio(xyz, args.k, args.sample_n, args.seed),
        "main_component_ratio": main_component_ratio(xyz, args.sample_n, args.seed),
        "bbox_volume": bbox_volume(xyz),
        "local_thickness_median": local_thickness_median(xyz, args.k, args.sample_n, args.seed),
        "render_edge_strength": render_edge_strength(render_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute comparable metrics for one or more PLY reconstructions.")
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Format: name|/path/to/point_cloud.ply or name|/path/to/point_cloud.ply|/path/to/test/renders",
    )
    parser.add_argument("--out-csv", default="ply_metrics.csv")
    parser.add_argument("--out-json", default="ply_metrics.json")
    parser.add_argument("--k", type=int, default=16, help="kNN size for outlier and local thickness metrics.")
    parser.add_argument("--sample-n", type=int, default=50000, help="Sample size for expensive metrics.")
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    rows = []
    for item in args.item:
        rows.append(compute_one(*parse_item(item), args))

    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {args.out_csv}")
    print(f"WROTE {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
