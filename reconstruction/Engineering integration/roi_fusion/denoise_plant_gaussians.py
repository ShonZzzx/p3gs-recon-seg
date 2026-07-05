from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from plant_weighting import build_plant_mask, load_plant_mask, save_plant_mask


def read_vertices(path: str | Path) -> np.ndarray:
    return PlyData.read(str(path))["vertex"].data


def xyz_from_vertices(vertices: np.ndarray) -> np.ndarray:
    missing = [name for name in ("x", "y", "z") if name not in vertices.dtype.names]
    if missing:
        raise ValueError(f"PLY vertex data missing coordinates: {', '.join(missing)}")
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64, copy=False)


def sor_keep_mask(xyz: np.ndarray, k: int = 16, mad_multiplier: float = 3.0) -> tuple[np.ndarray, dict[str, float | int]]:
    if len(xyz) <= max(2, k):
        return np.ones(len(xyz), dtype=np.bool_), {
            "sor_k": int(k),
            "sor_threshold": 0.0,
            "sor_removed_count": 0,
        }
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=int(k) + 1, workers=-1)
    scores = dists[:, 1:].mean(axis=1)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = median + float(mad_multiplier) * 1.4826 * mad
    keep = scores <= threshold
    stats: dict[str, float | int] = {
        "sor_k": int(k),
        "sor_mad_multiplier": float(mad_multiplier),
        "sor_score_median": median,
        "sor_score_mad": mad,
        "sor_threshold": float(threshold),
        "sor_removed_count": int((~keep).sum()),
    }
    return keep, stats


def dbscan_keep_mask(
    xyz: np.ndarray,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.bool_), {
            "dbscan_eps": float(eps),
            "dbscan_min_samples": int(min_samples),
            "dbscan_min_cluster_size": int(min_cluster_size),
            "dbscan_removed_count": 0,
            "dbscan_cluster_count": 0,
        }

    tree = cKDTree(xyz)
    neighbor_lists = tree.query_ball_point(xyz, r=float(eps), workers=-1)
    core = np.array([len(neighbors) >= min_samples for neighbors in neighbor_lists], dtype=np.bool_)
    row = []
    col = []
    for idx, neighbors in enumerate(neighbor_lists):
        if not core[idx]:
            continue
        for neighbor in neighbors:
            if core[neighbor]:
                row.append(idx)
                col.append(neighbor)
    if row:
        graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(xyz), len(xyz))).tocsr()
        cluster_id, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        cluster_id = 0
        labels = np.full(len(xyz), -1, dtype=np.int64)

    keep = np.zeros(len(xyz), dtype=np.bool_)
    cluster_sizes = []
    for label in range(cluster_id):
        cluster_mask = np.logical_and(labels == label, core)
        size = int(cluster_mask.sum())
        if size == 0:
            continue
        cluster_sizes.append(size)
        if size >= min_cluster_size:
            keep[cluster_mask] = True
            border_candidates = np.flatnonzero(~core)
            for border_idx in border_candidates:
                if np.any(cluster_mask[neighbor_lists[border_idx]]):
                    keep[border_idx] = True

    stats: dict[str, float | int] = {
        "dbscan_eps": float(eps),
        "dbscan_min_samples": int(min_samples),
        "dbscan_min_cluster_size": int(min_cluster_size),
        "dbscan_cluster_count": int(cluster_id),
        "dbscan_largest_cluster_size": int(max(cluster_sizes) if cluster_sizes else 0),
        "dbscan_removed_count": int((~keep).sum()),
    }
    return keep, stats


def build_plant_keep_mask(
    plant_xyz: np.ndarray,
    sor_k: int = 16,
    sor_mad_multiplier: float = 3.0,
    dbscan_eps: float = 0.01,
    dbscan_min_samples: int = 6,
    dbscan_min_cluster_size: int = 30,
) -> tuple[np.ndarray, dict[str, float | int]]:
    sor_keep, sor_stats = sor_keep_mask(plant_xyz, k=sor_k, mad_multiplier=sor_mad_multiplier)
    dbscan_keep = np.zeros(len(plant_xyz), dtype=np.bool_)
    dbscan_stats: dict[str, float | int]
    if sor_keep.any():
        dbscan_keep_after_sor, dbscan_stats = dbscan_keep_mask(
            plant_xyz[sor_keep],
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            min_cluster_size=dbscan_min_cluster_size,
        )
        dbscan_keep[np.flatnonzero(sor_keep)] = dbscan_keep_after_sor
    else:
        dbscan_stats = {
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_samples": int(dbscan_min_samples),
            "dbscan_min_cluster_size": int(dbscan_min_cluster_size),
            "dbscan_removed_count": 0,
            "dbscan_cluster_count": 0,
            "dbscan_largest_cluster_size": 0,
        }
    stats = {
        "plant_count": int(len(plant_xyz)),
        "kept_plant_count": int(dbscan_keep.sum()),
        "removed_plant_count": int((~dbscan_keep).sum()),
        **sor_stats,
        **dbscan_stats,
    }
    return dbscan_keep, stats


def filter_full_vertices_by_plant_keep(
    full_vertices: np.ndarray,
    plant_mask: np.ndarray,
    plant_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    plant_indices = np.flatnonzero(np.asarray(plant_mask, dtype=np.bool_))
    if len(plant_indices) != len(plant_keep):
        raise ValueError(f"plant_keep length {len(plant_keep)} != plant mask count {len(plant_indices)}")
    full_keep = np.ones(len(full_vertices), dtype=np.bool_)
    rejected_plant_indices = plant_indices[~np.asarray(plant_keep, dtype=np.bool_)]
    full_keep[rejected_plant_indices] = False
    filtered = full_vertices[full_keep]
    stats = {
        "full_count": int(len(full_vertices)),
        "full_kept_count": int(len(filtered)),
        "removed_from_full_count": int((~full_keep).sum()),
    }
    return filtered, full_keep, stats


def write_vertices_like(path: str | Path, vertices: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(output))


def denoise_full_ply(
    full_ply: str | Path,
    plant_ply: str | Path,
    output_ply: str | Path,
    mask_path: str | Path = "",
    tolerance: float = 1e-6,
    sor_k: int = 16,
    sor_mad_multiplier: float = 3.0,
    dbscan_eps: float = 0.01,
    dbscan_min_samples: int = 6,
    dbscan_min_cluster_size: int = 30,
    stats_path: str | Path = "",
) -> dict[str, object]:
    full_vertices = read_vertices(full_ply)
    if mask_path and Path(mask_path).exists():
        plant_mask, mask_meta = load_plant_mask(mask_path)
    else:
        plant_mask, mask_meta = build_plant_mask(full_ply, plant_ply, tolerance=tolerance)
        if mask_path:
            save_plant_mask(plant_mask, mask_path, mask_meta)
    if len(plant_mask) != len(full_vertices):
        raise ValueError(f"plant mask length {len(plant_mask)} != full vertex count {len(full_vertices)}")
    plant_xyz = xyz_from_vertices(full_vertices[plant_mask])
    plant_keep, denoise_stats = build_plant_keep_mask(
        plant_xyz,
        sor_k=sor_k,
        sor_mad_multiplier=sor_mad_multiplier,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_min_cluster_size=dbscan_min_cluster_size,
    )
    filtered, _, full_stats = filter_full_vertices_by_plant_keep(full_vertices, plant_mask, plant_keep)
    write_vertices_like(output_ply, filtered)
    stats: dict[str, object] = {
        "input_full_ply": str(Path(full_ply).resolve()),
        "input_plant_ply": str(Path(plant_ply).resolve()),
        "output_ply": str(Path(output_ply).resolve()),
        "mask_path": str(Path(mask_path).resolve()) if mask_path else "",
        "mask_meta": mask_meta,
        **denoise_stats,
        **full_stats,
    }
    if stats_path:
        stats_output = Path(stats_path)
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove denoised plant outlier Gaussians from a full Gaussian PLY.")
    parser.add_argument("--full_ply", required=True)
    parser.add_argument("--plant_ply", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--mask_path", default="")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--sor_k", type=int, default=16)
    parser.add_argument("--sor_mad_multiplier", type=float, default=3.0)
    parser.add_argument("--dbscan_eps", type=float, default=0.01)
    parser.add_argument("--dbscan_min_samples", type=int, default=6)
    parser.add_argument("--dbscan_min_cluster_size", type=int, default=30)
    parser.add_argument("--stats", default="")
    args = parser.parse_args()
    stats = denoise_full_ply(
        full_ply=args.full_ply,
        plant_ply=args.plant_ply,
        output_ply=args.output_ply,
        mask_path=args.mask_path,
        tolerance=args.tolerance,
        sor_k=args.sor_k,
        sor_mad_multiplier=args.sor_mad_multiplier,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_min_cluster_size=args.dbscan_min_cluster_size,
        stats_path=args.stats,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
