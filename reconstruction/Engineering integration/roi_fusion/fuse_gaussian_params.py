#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def vertex_xyz(vertex: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)


def nearest_distances(src: np.ndarray, dst: np.ndarray, block: int = 65536) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if len(dst) == 0:
        return np.full(len(src), np.inf, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(dst)
    out = np.empty(len(src), dtype=np.float64)
    for start in range(0, len(src), block):
        stop = min(start + block, len(src))
        out[start:stop] = nn.kneighbors(src[start:stop], return_distance=True)[0][:, 0]
    return out


def safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return default
    return float(np.quantile(vals, q))


def normalize_rotations(vertex: np.ndarray) -> None:
    names = set(vertex.dtype.names or [])
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if not set(rot_names).issubset(names):
        return
    rot = np.stack([np.asarray(vertex[c], dtype=np.float64) for c in rot_names], axis=1)
    norm = np.linalg.norm(rot, axis=1)
    valid = np.isfinite(norm) & (norm > 1e-12)
    for i, col in enumerate(rot_names):
        arr = vertex[col]
        arr[valid] = (rot[valid, i] / norm[valid]).astype(arr.dtype)
    if np.any(~valid):
        vertex["rot_0"][~valid] = 1.0
        for col in rot_names[1:]:
            vertex[col][~valid] = 0.0


def sanitize_opacity(vertex: np.ndarray, quantile: float = 0.999) -> dict:
    if "opacity" not in set(vertex.dtype.names or []):
        return {"opacity_sanitized": False}
    opacity = vertex["opacity"]
    values = np.asarray(opacity, dtype=np.float64)
    finite = np.isfinite(values)
    nonfinite = ~finite
    if not np.any(nonfinite):
        return {"opacity_sanitized": True, "nonfinite_opacity_replaced": 0}
    if np.any(finite):
        replacement = float(np.quantile(values[finite], quantile))
    else:
        replacement = 0.0
    opacity[nonfinite] = np.asarray(replacement, dtype=opacity.dtype)
    return {
        "opacity_sanitized": True,
        "nonfinite_opacity_replaced": int(np.count_nonzero(nonfinite)),
        "opacity_replacement_quantile": float(quantile),
        "opacity_replacement_value": replacement,
    }


def fuse_gaussian_vertices(
    rade_vertex: np.ndarray,
    gof_vertex: np.ndarray,
    keep_quantile: float,
    fill_quantile: float,
    max_fill_points: int,
    seed: int,
    normalize_rot: bool = True,
    sanitize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if rade_vertex.dtype.names != gof_vertex.dtype.names:
        raise ValueError("RaDe and GOF PLY vertex fields differ; cannot concatenate safely")
    if len(rade_vertex) == 0 and len(gof_vertex) == 0:
        raise ValueError("Both Gaussian point clouds are empty")

    rade_xyz = vertex_xyz(rade_vertex)
    gof_xyz = vertex_xyz(gof_vertex)

    if len(rade_vertex) == 0:
        fused = gof_vertex.copy()
        source = np.full(len(fused), "gof", dtype="<U6")
        source_index = np.arange(len(fused), dtype=np.int64)
        meta = {
            "rade_points": 0,
            "gof_points": int(len(gof_vertex)),
            "kept_rade_points": 0,
            "added_gof_points": int(len(gof_vertex)),
            "keep_threshold": 0.0,
            "fill_threshold": 0.0,
        }
    elif len(gof_vertex) == 0:
        fused = rade_vertex.copy()
        source = np.full(len(fused), "radegs", dtype="<U6")
        source_index = np.arange(len(fused), dtype=np.int64)
        meta = {
            "rade_points": int(len(rade_vertex)),
            "gof_points": 0,
            "kept_rade_points": int(len(rade_vertex)),
            "added_gof_points": 0,
            "keep_threshold": 0.0,
            "fill_threshold": 0.0,
        }
    else:
        rade_to_gof = nearest_distances(rade_xyz, gof_xyz)
        gof_to_rade = nearest_distances(gof_xyz, rade_xyz)

        keep_threshold = safe_quantile(rade_to_gof, keep_quantile, 0.0)
        fill_threshold = safe_quantile(gof_to_rade, fill_quantile, keep_threshold)
        if fill_threshold < keep_threshold:
            fill_threshold = keep_threshold

        keep_mask = rade_to_gof <= keep_threshold
        fill_mask = gof_to_rade > fill_threshold
        kept_indices = np.nonzero(keep_mask)[0]
        fill_indices = np.nonzero(fill_mask)[0]
        if len(fill_indices) > max_fill_points:
            rng = np.random.default_rng(seed)
            fill_indices = np.sort(rng.choice(fill_indices, size=max_fill_points, replace=False))

        fused = np.concatenate([rade_vertex[kept_indices], gof_vertex[fill_indices]]).copy()
        source = np.concatenate(
            [
                np.full(len(kept_indices), "radegs", dtype="<U6"),
                np.full(len(fill_indices), "gof", dtype="<U6"),
            ]
        )
        source_index = np.concatenate([kept_indices.astype(np.int64), fill_indices.astype(np.int64)])
        meta = {
            "rade_points": int(len(rade_vertex)),
            "gof_points": int(len(gof_vertex)),
            "kept_rade_points": int(len(kept_indices)),
            "added_gof_points": int(len(fill_indices)),
            "keep_threshold": float(keep_threshold),
            "fill_threshold": float(fill_threshold),
            "keep_quantile": float(keep_quantile),
            "fill_quantile": float(fill_quantile),
            "max_fill_points": int(max_fill_points),
            "seed": int(seed),
        }

    if normalize_rot:
        normalize_rotations(fused)
        meta["rotations_normalized"] = True
    else:
        meta["rotations_normalized"] = False
    if sanitize:
        meta.update(sanitize_opacity(fused))
    else:
        meta["opacity_sanitized"] = False
    return fused, source, source_index, meta


def read_vertex(path: Path) -> tuple[np.ndarray, PlyData]:
    ply = PlyData.read(str(path))
    return ply["vertex"].data, ply


def write_vertex(path: Path, vertex: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse RaDe and GOF Gaussian PLY files while preserving all Gaussian parameters.")
    parser.add_argument("--rade-ply", required=True, type=Path)
    parser.add_argument("--gof-ply", required=True, type=Path)
    parser.add_argument("--out-ply", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-map", type=Path, default=None)
    parser.add_argument("--keep-quantile", type=float, default=0.75)
    parser.add_argument("--fill-quantile", type=float, default=0.35)
    parser.add_argument("--max-fill-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--no-normalize-rot", action="store_true")
    parser.add_argument("--sanitize-opacity", action="store_true")
    args = parser.parse_args()

    rade_vertex, _ = read_vertex(args.rade_ply)
    gof_vertex, _ = read_vertex(args.gof_ply)
    fused, source, source_index, meta = fuse_gaussian_vertices(
        rade_vertex,
        gof_vertex,
        args.keep_quantile,
        args.fill_quantile,
        args.max_fill_points,
        args.seed,
        normalize_rot=not args.no_normalize_rot,
        sanitize=args.sanitize_opacity,
    )
    meta.update(
        {
            "rade_ply": str(args.rade_ply),
            "gof_ply": str(args.gof_ply),
            "out_ply": str(args.out_ply),
            "property_count": len(fused.dtype.names or []),
            "fused_points": int(len(fused)),
        }
    )
    write_vertex(args.out_ply, fused)
    out_json = args.out_json or args.out_ply.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    out_map = args.out_map or args.out_ply.with_name("source_map.npz")
    np.savez_compressed(out_map, source=source, source_index=source_index)
    print(f"WROTE {args.out_ply}")
    print(f"WROTE {out_json}")
    print(f"WROTE {out_map}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
