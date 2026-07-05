#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement

from ply_color_utils import sh_dc_to_rgb


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)
    names = set(v.data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([np.asarray(v[c], dtype=np.uint8) for c in ("red", "green", "blue")], axis=1)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        f_dc = np.stack([np.asarray(v[c], dtype=np.float64) for c in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1)
        rgb = sh_dc_to_rgb(f_dc)
    else:
        rgb = None
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None) -> None:
    if rgb is None:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    vertex = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = xyz[:, 0].astype(np.float32)
    vertex["y"] = xyz[:, 1].astype(np.float32)
    vertex["z"] = xyz[:, 2].astype(np.float32)
    vertex["red"] = rgb[:, 0].astype(np.uint8)
    vertex["green"] = rgb[:, 1].astype(np.uint8)
    vertex["blue"] = rgb[:, 2].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def sample_for_fit(xyz: np.ndarray, sample_n: int, seed: int) -> np.ndarray:
    if len(xyz) <= sample_n:
        return xyz
    rng = np.random.default_rng(seed)
    return xyz[rng.choice(len(xyz), size=sample_n, replace=False)]


def sor_filter(xyz: np.ndarray, k: int, std_ratio: float, sample_n: int, seed: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return np.ones(len(xyz), dtype=bool)
    fit_xyz = sample_for_fit(xyz, sample_n, seed)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(fit_xyz)
    dist, _ = nn.kneighbors(xyz)
    mean_dist = dist[:, 1:].mean(axis=1)
    threshold = float(mean_dist.mean() + std_ratio * mean_dist.std())
    return mean_dist <= threshold


def auto_eps(xyz: np.ndarray, k: int, eps_mult: float, sample_n: int, seed: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return 1e-6
    sample = sample_for_fit(xyz, sample_n, seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(sample)), algorithm="auto").fit(sample)
    dist, _ = nn.kneighbors(sample)
    base = float(np.median(dist[:, 1:]))
    if not math.isfinite(base) or base <= 0:
        span = float(np.linalg.norm(sample.max(axis=0) - sample.min(axis=0)))
        base = max(span * 0.002, 1e-6)
    return base * eps_mult


def dbscan_filter(
    xyz: np.ndarray,
    eps: float,
    min_points: int,
    keep_clusters: int,
    min_cluster_ratio: float,
    sample_n: int,
    seed: int,
) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    if len(xyz) <= sample_n:
        labels = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit_predict(xyz)
        return cluster_keep_mask(labels, keep_clusters, min_cluster_ratio)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(xyz), size=sample_n, replace=False)
    sample = xyz[sample_idx]
    sample_labels = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit_predict(sample)
    sample_keep = cluster_keep_mask(sample_labels, keep_clusters, min_cluster_ratio)
    kept_sample = sample[sample_keep]
    if len(kept_sample) == 0:
        return np.ones(len(xyz), dtype=bool)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(kept_sample)
    dist, _ = nn.kneighbors(xyz)
    return dist[:, 0] <= eps


def cluster_keep_mask(labels: np.ndarray, keep_clusters: int, min_cluster_ratio: float) -> np.ndarray:
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return np.ones(len(labels), dtype=bool)
    counts = np.bincount(valid)
    order = np.argsort(counts)[::-1]
    keep = set()
    total = len(labels)
    for label in order[:keep_clusters]:
        if counts[label] / total >= min_cluster_ratio:
            keep.add(int(label))
    if not keep:
        keep.add(int(order[0]))
    return np.array([label in keep for label in labels], dtype=bool)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray | None, voxel_size: float) -> tuple[np.ndarray, np.ndarray | None]:
    if voxel_size <= 0 or len(xyz) == 0:
        return xyz, rgb
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    out_xyz = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    for axis in range(3):
        out_xyz[:, axis] = np.bincount(inverse, weights=xyz[:, axis]) / counts
    if rgb is None:
        return out_xyz, None
    out_rgb = np.zeros((len(out_xyz), 3), dtype=np.uint8)
    for axis in range(3):
        out_rgb[:, axis] = np.clip(np.bincount(inverse, weights=rgb[:, axis]) / counts, 0, 255).astype(np.uint8)
    return out_xyz, out_rgb


def render_view(xyz: np.ndarray, rgb: np.ndarray | None, out: Path, axes: tuple[int, int], title: str, seed: int) -> None:
    if len(xyz) > 160000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), 160000, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
    colors = rgb.astype(np.float32) / 255.0 if rgb is not None else xyz[:, 2]
    fig = plt.figure(figsize=(7, 7), dpi=170)
    ax = fig.add_subplot(111)
    ax.scatter(xyz[:, axes[0]], xyz[:, axes[1]], s=0.18, c=colors, cmap="viridis", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout(pad=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def find_input_ply(root: Path, method: str, plant: str) -> Path | None:
    if method == "fused":
        candidate = root / "fused" / plant / "fused_point_cloud.ply"
        return candidate if candidate.exists() else None
    pcs = list((root / method / plant / "point_cloud").glob("iteration_*/*.ply"))
    if not pcs:
        return None

    def iteration_number(path: Path) -> int:
        name = path.parent.name
        if name.startswith("iteration_"):
            suffix = name.removeprefix("iteration_")
            if suffix.isdigit():
                return int(suffix)
        return -1

    point_clouds = [item for item in pcs if item.name == "point_cloud.ply"]
    candidates = point_clouds or pcs
    return max(candidates, key=lambda item: (iteration_number(item), str(item)))


def process_one(root: Path, out_root: Path, method: str, plant: str, args: argparse.Namespace) -> dict:
    ply = find_input_ply(root, method, plant)
    if ply is None:
        return {"method": method, "plant": plant, "status": "missing_input"}
    xyz, rgb = read_ply(ply)
    original_n = len(xyz)

    sor_mask = sor_filter(xyz, args.sor_k, args.sor_std, args.sample_n, args.seed)
    xyz = xyz[sor_mask]
    rgb = rgb[sor_mask] if rgb is not None else None

    eps = args.dbscan_eps if args.dbscan_eps > 0 else auto_eps(xyz, args.sor_k, args.eps_mult, args.sample_n, args.seed)
    db_mask = dbscan_filter(xyz, eps, args.dbscan_min_points, args.keep_clusters, args.min_cluster_ratio, args.sample_n, args.seed)
    xyz = xyz[db_mask]
    rgb = rgb[db_mask] if rgb is not None else None

    voxel_size = args.voxel_size
    if voxel_size < 0:
        voxel_size = auto_eps(xyz, args.sor_k, args.voxel_mult, args.sample_n, args.seed)
    if voxel_size > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, voxel_size)

    out_dir = out_root / method / plant
    out_ply = out_dir / "denoised_point_cloud.ply"
    write_ply(out_ply, xyz, rgb)
    render_view(xyz, rgb, out_dir / "views" / "front.png", (0, 2), f"{method} {plant} front", args.seed)
    render_view(xyz, rgb, out_dir / "views" / "side.png", (1, 2), f"{method} {plant} side", args.seed)
    render_view(xyz, rgb, out_dir / "views" / "top.png", (0, 1), f"{method} {plant} top", args.seed)

    meta = {
        "method": method,
        "plant": plant,
        "status": "ok",
        "input_ply": str(ply),
        "output_ply": str(out_ply),
        "original_points": int(original_n),
        "after_sor_points": int(sor_mask.sum()),
        "after_dbscan_points": int(db_mask.sum()),
        "final_points": int(len(xyz)),
        "dbscan_eps": float(eps),
        "voxel_size": float(voxel_size),
    }
    (out_dir / "denoise_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["radegs", "gof", "fused"])
    parser.add_argument("--plants", nargs="+", default=["plant_002"])
    parser.add_argument("--sor-k", type=int, default=24)
    parser.add_argument("--sor-std", type=float, default=3.0)
    parser.add_argument("--dbscan-eps", type=float, default=-1.0)
    parser.add_argument("--eps-mult", type=float, default=6.0)
    parser.add_argument("--dbscan-min-points", type=int, default=16)
    parser.add_argument("--keep-clusters", type=int, default=5)
    parser.add_argument("--min-cluster-ratio", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="0 disables voxel; negative uses auto voxel.")
    parser.add_argument("--voxel-mult", type=float, default=1.5)
    parser.add_argument("--sample-n", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    out_root = args.out_root or (args.root / "denoised")
    rows = []
    for plant in args.plants:
        for method in args.methods:
            rows.append(process_one(args.root, out_root, method, plant, args))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "denoise_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
