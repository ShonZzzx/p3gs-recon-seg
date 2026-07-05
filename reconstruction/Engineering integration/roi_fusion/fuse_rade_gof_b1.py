#!/usr/bin/env python3
"""Fuse RaDe-GS and GOF point clouds into a single B1 plant reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def read_vertex_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = list(vertex.data.dtype.names or [])
    xyz = np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)
    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([np.asarray(vertex[c], dtype=np.uint8) for c in ("red", "green", "blue")], axis=1)
    return xyz, rgb, names


def normalize_rgb(rgb: np.ndarray | None, n: int, fallback: np.ndarray | None = None) -> np.ndarray:
    if rgb is not None and len(rgb) == n:
        return rgb
    if fallback is not None and len(fallback) == n:
        return fallback
    return np.full((n, 3), 180, dtype=np.uint8)


def nearest_distances(src: np.ndarray, dst: np.ndarray, block: int = 65536) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if len(dst) == 0:
        return np.full(len(src), np.inf, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(dst)
    out = np.empty(len(src), dtype=np.float64)
    for start in range(0, len(src), block):
        stop = min(len(src), start + block)
        out[start:stop] = nn.kneighbors(src[start:stop], return_distance=True)[0][:, 0]
    return out


def safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return default
    return float(np.quantile(vals, q))


def fuse_points(
    rade_xyz: np.ndarray,
    rade_rgb: np.ndarray | None,
    gof_xyz: np.ndarray,
    gof_rgb: np.ndarray | None,
    keep_quantile: float,
    fill_quantile: float,
    max_fill_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(rade_xyz) == 0 and len(gof_xyz) == 0:
        raise ValueError("Both RaDe and GOF point clouds are empty")

    rade_rgb = normalize_rgb(rade_rgb, len(rade_xyz))
    gof_rgb = normalize_rgb(gof_rgb, len(gof_xyz))

    if len(rade_xyz) == 0:
        fused_xyz = gof_xyz.copy()
        fused_rgb = gof_rgb.copy()
        return fused_xyz, fused_rgb, {
            "rade_points": 0,
            "gof_points": len(gof_xyz),
            "kept_rade_points": 0,
            "added_gof_points": len(gof_xyz),
            "keep_threshold": 0.0,
            "fill_threshold": 0.0,
        }

    if len(gof_xyz) == 0:
        fused_xyz = rade_xyz.copy()
        fused_rgb = rade_rgb.copy()
        return fused_xyz, fused_rgb, {
            "rade_points": len(rade_xyz),
            "gof_points": 0,
            "kept_rade_points": len(rade_xyz),
            "added_gof_points": 0,
            "keep_threshold": 0.0,
            "fill_threshold": 0.0,
        }

    rade_to_gof = nearest_distances(rade_xyz, gof_xyz)
    gof_to_rade = nearest_distances(gof_xyz, rade_xyz)

    keep_threshold = safe_quantile(rade_to_gof, keep_quantile, 0.0)
    fill_threshold = safe_quantile(gof_to_rade, fill_quantile, keep_threshold)
    if fill_threshold < keep_threshold:
        fill_threshold = keep_threshold

    keep_mask = rade_to_gof <= keep_threshold
    kept_xyz = rade_xyz[keep_mask]
    kept_rgb = rade_rgb[keep_mask]

    fill_mask = gof_to_rade > fill_threshold
    fill_xyz = gof_xyz[fill_mask]
    fill_rgb = gof_rgb[fill_mask]
    if len(fill_xyz) > max_fill_points:
        rng = np.random.default_rng(20260625)
        choice = rng.choice(len(fill_xyz), size=max_fill_points, replace=False)
        fill_xyz = fill_xyz[choice]
        fill_rgb = fill_rgb[choice]

    fused_xyz = np.concatenate([kept_xyz, fill_xyz], axis=0)
    fused_rgb = np.concatenate([kept_rgb, fill_rgb], axis=0)

    if len(fused_xyz) == 0:
        raise ValueError("Fusion produced an empty cloud")

    meta = {
        "rade_points": int(len(rade_xyz)),
        "gof_points": int(len(gof_xyz)),
        "kept_rade_points": int(len(kept_xyz)),
        "added_gof_points": int(len(fill_xyz)),
        "keep_threshold": float(keep_threshold),
        "fill_threshold": float(fill_threshold),
    }
    return fused_xyz, fused_rgb, meta


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse RaDe-GS and GOF point clouds for B1.")
    parser.add_argument("--rade-ply", required=True, type=Path)
    parser.add_argument("--gof-ply", required=True, type=Path)
    parser.add_argument("--out-ply", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--keep-quantile", type=float, default=0.75)
    parser.add_argument("--fill-quantile", type=float, default=0.35)
    parser.add_argument("--max-fill-points", type=int, default=200000)
    args = parser.parse_args()

    rade_xyz, rade_rgb, _ = read_vertex_ply(args.rade_ply)
    gof_xyz, gof_rgb, _ = read_vertex_ply(args.gof_ply)
    fused_xyz, fused_rgb, meta = fuse_points(
        rade_xyz,
        rade_rgb,
        gof_xyz,
        gof_rgb,
        args.keep_quantile,
        args.fill_quantile,
        args.max_fill_points,
    )
    write_ply(args.out_ply, fused_xyz, fused_rgb)
    out_json = args.out_json or args.out_ply.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {args.out_ply}")
    print(f"WROTE {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
