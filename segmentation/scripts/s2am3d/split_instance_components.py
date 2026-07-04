#!/usr/bin/env python3
"""Split each instance id into spatial connected components."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

from refine_s2am3d_instances import (
    normalize_pc,
    palette,
    read_xyz_rgb,
    write_scalar_ply,
    write_xyzrgb_ply,
)


def estimate_spacing(points: np.ndarray, k: int, sample_size: int, seed: int) -> float:
    if len(points) <= 1:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(points) > sample_size:
        idx = rng.choice(len(points), sample_size, replace=False)
        pts = points[idx]
    else:
        pts = points
    n_neighbors = min(k + 1, len(pts))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(pts)
    dist = nn.kneighbors(pts, return_distance=True)[0][:, -1]
    return float(np.median(dist))


def connected_components_radius(points: np.ndarray, radius: float, knn_cap: int) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [np.array([0], dtype=np.int64)]

    nn = NearestNeighbors(n_neighbors=min(knn_cap + 1, len(points)), algorithm="auto").fit(points)
    distances, neighbors = nn.kneighbors(points, return_distance=True)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    visited = np.zeros(len(points), dtype=bool)
    comps = []
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            valid = neighbors[cur][distances[cur] <= radius]
            for nb in valid:
                nb = int(nb)
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps


def split_labels(
    points_norm: np.ndarray,
    labels: np.ndarray,
    radius: float,
    min_component_points: int,
    knn_cap: int,
) -> tuple[np.ndarray, list[dict]]:
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    summary = []

    for old_id in [int(x) for x in np.unique(labels) if x > 0]:
        old_indices = np.flatnonzero(labels == old_id)
        comps_local = connected_components_radius(points_norm[old_indices], radius, knn_cap)
        comps = [old_indices[c] for c in comps_local]
        comps.sort(key=len, reverse=True)
        large = [c for c in comps if len(c) >= min_component_points]
        small = [c for c in comps if len(c) < min_component_points]

        if not large and comps:
            large = [comps[0]]
            small = comps[1:]

        new_ids = []
        for comp in large:
            out[comp] = next_id
            new_ids.append(next_id)
            summary.append(
                {
                    "old_id": old_id,
                    "new_id": next_id,
                    "point_count": int(len(comp)),
                    "kind": "large",
                }
            )
            next_id += 1

        # Keep tiny nearby fragments with the nearest large component of the same old label.
        if large and small:
            centroids = np.stack([points_norm[c].mean(axis=0) for c in large], axis=0)
            large_ids = np.asarray(new_ids, dtype=np.int32)
            for comp in small:
                center = points_norm[comp].mean(axis=0)
                nearest = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
                out[comp] = int(large_ids[nearest])

    return out, summary


def reset_generated_dir(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def write_gof_instances(out_dir: Path, vertex: np.ndarray, labels: np.ndarray, text: bool):
    inst_dir = out_dir / "gof_instances"
    reset_generated_dir(inst_dir)
    inst_dir.mkdir(exist_ok=True)
    for instance_id in [int(x) for x in np.unique(labels) if x > 0]:
        idx = np.flatnonzero(labels == instance_id)
        subset = vertex[idx]
        PlyData([PlyElement.describe(subset, "vertex")], text=text).write(
            str(inst_dir / f"instance_{instance_id:03d}_gof.ply")
        )


def write_rgb_previews(out_dir: Path, xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray):
    view_dir = out_dir / "view_instances_rgb"
    reset_generated_dir(view_dir)
    view_dir.mkdir(exist_ok=True)
    rows = []
    for instance_id in [int(x) for x in np.unique(labels) if x > 0]:
        idx = np.flatnonzero(labels == instance_id)
        path = view_dir / f"instance_{instance_id:03d}_rgb_preview.ply"
        write_xyzrgb_ply(path, xyz[idx], rgb[idx])
        rows.append([instance_id, int(len(idx)), str(path), str(out_dir / "gof_instances" / f"instance_{instance_id:03d}_gof.ply")])
    rows.sort(key=lambda row: row[1], reverse=True)
    with open(out_dir / "instance_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "full_point_count", "rgb_preview_path", "gof_subset_path"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", required=True, type=Path)
    parser.add_argument("--seg_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--spacing_k", type=int, default=8)
    parser.add_argument("--spacing_sample_size", type=int, default=50000)
    parser.add_argument("--radius_factor", type=float, default=6.0)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--min_component_points", type=int, default=120)
    parser.add_argument("--knn_cap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ply, vertex, xyz, rgb = read_xyz_rgb(args.input_ply)
    labels = np.load(args.seg_dir / "full_instance_ids.npy").astype(np.int32)
    if len(labels) != len(xyz):
        raise ValueError(
            f"Label count mismatch: {args.seg_dir / 'full_instance_ids.npy'} has {len(labels)} labels, "
            f"but {args.input_ply} has {len(xyz)} points"
        )
    points_norm = normalize_pc(xyz)

    spacing = estimate_spacing(points_norm, args.spacing_k, args.spacing_sample_size, args.seed)
    radius = args.radius if args.radius is not None else spacing * args.radius_factor
    split, split_summary = split_labels(points_norm, labels, radius, args.min_component_points, args.knn_cap)

    np.save(args.output_dir / "full_instance_ids.npy", split)
    write_xyzrgb_ply(args.output_dir / "full_instances_colored.ply", xyz, palette(split))
    write_scalar_ply(args.output_dir / "full_instance_id_scalar.ply", xyz, split)
    write_gof_instances(args.output_dir, vertex, split, ply.text)
    write_rgb_previews(args.output_dir, xyz, rgb, split)

    metadata = {
        "input_ply": str(args.input_ply),
        "source_seg_dir": str(args.seg_dir),
        "before_instances": int(len([x for x in np.unique(labels) if x > 0])),
        "after_instances": int(len([x for x in np.unique(split) if x > 0])),
        "spacing": float(spacing),
        "radius": float(radius),
        "radius_factor": args.radius_factor,
        "min_component_points": args.min_component_points,
        "knn_cap": args.knn_cap,
        "split_summary": split_summary,
    }
    with open(args.output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"before instances: {metadata['before_instances']}")
    print(f"after instances: {metadata['after_instances']}")
    print(f"radius: {radius:.6f}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
