#!/usr/bin/env python3
"""Refine S2AM3D instance ids with spatial cleanup and adjacency merging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


def normalize_pc(xyz: np.ndarray) -> np.ndarray:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    center = (mn + mx) / 2.0
    scale = float(np.max((mx - mn) / 2.0)) + 1e-10
    return ((xyz - center) / scale).astype(np.float32)


def read_xyz_rgb(path: Path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    names = vertex.dtype.names or ()
    if all(name in names for name in ("red", "green", "blue")):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
    else:
        rgb = np.ones((len(xyz), 3), dtype=np.float32) * 0.75
    return ply, vertex, xyz, (rgb * 255).astype(np.uint8)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(math.floor(h * 6.0))
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def palette(instance_ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(instance_ids), 3), dtype=np.uint8)
    for idx in np.unique(instance_ids):
        if idx <= 0:
            continue
        hue = (int(idx) * 0.618033988749895) % 1.0
        colors[instance_ids == idx] = np.asarray(hsv_to_rgb(hue, 0.70, 0.95)) * 255
    return colors


def reset_generated_dir(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def write_xyzrgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    data = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def write_scalar_ply(path: Path, xyz: np.ndarray, ids: np.ndarray):
    data = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("instance_id", "f4")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["instance_id"] = ids.astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


class UnionFind:
    def __init__(self, values):
        self.parent = {int(v): int(v) for v in values}

    def find(self, x: int) -> int:
        x = int(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def remove_small_labels(labels: np.ndarray, min_points: int) -> np.ndarray:
    labels = labels.copy()
    for label in np.unique(labels):
        if label <= 0:
            continue
        if int((labels == label).sum()) < min_points:
            labels[labels == label] = 0
    return labels


def merge_adjacent(labels: np.ndarray, neighbors: np.ndarray, min_ratio: float, min_edges: int) -> tuple[np.ndarray, list[dict]]:
    ids = [int(x) for x in np.unique(labels) if x > 0]
    counts = {idx: int((labels == idx).sum()) for idx in ids}
    pair_edges: Counter[tuple[int, int]] = Counter()

    for src, neigh in zip(labels, neighbors):
        if src <= 0:
            continue
        for j in neigh:
            dst = int(labels[j])
            if dst <= 0 or dst == src:
                continue
            a, b = sorted((int(src), dst))
            pair_edges[(a, b)] += 1

    uf = UnionFind(ids)
    merges = []
    for (a, b), edges in pair_edges.items():
        ratio = edges / max(1, min(counts[a], counts[b]))
        if edges >= min_edges and ratio >= min_ratio:
            uf.union(a, b)
            merges.append({"a": a, "b": b, "edges": int(edges), "ratio": float(ratio)})

    merged = labels.copy()
    for idx in ids:
        merged[labels == idx] = uf.find(idx)
    return merged, merges


def smooth_labels(labels: np.ndarray, neighbors: np.ndarray, iterations: int, min_votes: int) -> np.ndarray:
    labels = labels.copy()
    for _ in range(iterations):
        updated = labels.copy()
        for i, neigh in enumerate(neighbors):
            current = int(labels[i])
            votes = [int(labels[j]) for j in neigh if labels[j] > 0]
            if not votes:
                continue
            winner, count = Counter(votes).most_common(1)[0]
            if count >= min_votes and (current == 0 or winner != current):
                updated[i] = winner
        labels = updated
    return labels


def relabel_contiguous(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping = {int(old): new for new, old in enumerate([x for x in np.unique(labels) if x > 0], start=1)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new
    return out, mapping


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
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--min_points", type=int, default=120)
    parser.add_argument("--merge_ratio", type=float, default=0.28)
    parser.add_argument("--merge_min_edges", type=int, default=180)
    parser.add_argument("--smooth_iterations", type=int, default=2)
    parser.add_argument("--smooth_min_votes", type=int, default=7)
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
    pts = normalize_pc(xyz)
    neighbors = NearestNeighbors(n_neighbors=args.knn + 1, algorithm="auto").fit(pts).kneighbors(
        pts, return_distance=False
    )[:, 1:]

    before_ids = [int(x) for x in np.unique(labels) if x > 0]
    labels = remove_small_labels(labels, args.min_points)
    labels, merges = merge_adjacent(labels, neighbors, args.merge_ratio, args.merge_min_edges)
    labels = smooth_labels(labels, neighbors, args.smooth_iterations, args.smooth_min_votes)
    labels = remove_small_labels(labels, args.min_points)
    labels, mapping = relabel_contiguous(labels)

    np.save(args.output_dir / "full_instance_ids.npy", labels)
    write_xyzrgb_ply(args.output_dir / "full_instances_colored.ply", xyz, palette(labels))
    write_scalar_ply(args.output_dir / "full_instance_id_scalar.ply", xyz, labels)
    write_gof_instances(args.output_dir, vertex, labels, ply.text)
    write_rgb_previews(args.output_dir, xyz, rgb, labels)

    metadata = {
        "source_seg_dir": str(args.seg_dir),
        "input_ply": str(args.input_ply),
        "before_instances": len(before_ids),
        "after_instances": int(len([x for x in np.unique(labels) if x > 0])),
        "knn": args.knn,
        "min_points": args.min_points,
        "merge_ratio": args.merge_ratio,
        "merge_min_edges": args.merge_min_edges,
        "smooth_iterations": args.smooth_iterations,
        "smooth_min_votes": args.smooth_min_votes,
        "merges": merges,
    }
    with open(args.output_dir / "refine_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"before instances: {metadata['before_instances']}")
    print(f"after instances: {metadata['after_instances']}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
