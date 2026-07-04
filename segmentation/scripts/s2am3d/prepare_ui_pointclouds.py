#!/usr/bin/env python3
"""Convert GOF/3DGS-style PLY point clouds into S2AM3D UI .npy inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


SH_C0 = 0.28209479177387814


def load_xyz_color(path: Path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"{path} is missing vertex property {name!r}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if all(name in names for name in ("red", "green", "blue")):
        color = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        if color.max() > 1.5:
            color /= 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        color = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
    else:
        color = np.ones((len(xyz), 3), dtype=np.float32)

    return xyz, color.astype(np.float32)


def sample_indices(num_points: int, sample_size: int, seed: int):
    if sample_size <= 0 or sample_size >= num_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, sample_size, replace=False)).astype(np.int64)


def convert_file(path: Path, output_dir: Path, sample_size: int, seed: int, overwrite: bool):
    xyz, color = load_xyz_color(path)
    indices = sample_indices(len(xyz), sample_size, seed)
    out_path = output_dir / f"{path.stem}.npy"
    if out_path.exists() and not overwrite:
        print(f"skip existing: {out_path}")
        return

    data = {
        "coord": xyz[indices],
        "color": color[indices],
        "source_ply": str(path),
        "sample_indices": indices,
        "num_full_points": int(len(xyz)),
    }
    np.save(out_path, data)
    print(f"wrote {out_path} ({len(indices):,}/{len(xyz):,} points)")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare S2AM3D interactive UI point clouds")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} under {args.input_dir}")
    for path in files:
        convert_file(path, args.output_dir, args.sample_size, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
