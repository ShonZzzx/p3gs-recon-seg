#!/usr/bin/env python3
"""Automatic S2AM3D segmentation for GOF Gaussian-center PLY files.

Pipeline:
  GOF point_cloud.ply -> sampled S2AM3D points -> automatic prompt masks
  -> nearest-neighbor mask propagation -> original GOF PLY subsets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "decoder"))

from decoder.interactive_demo import FeatureExtractor, S2AM3DInference, normalize_pc  # noqa: E402
from decoder.param import parse_args as parse_decoder_args  # noqa: E402
from decoder.utils.misc import load_config  # noqa: E402


SH_C0 = 0.28209479177387814


def load_gof_ply(path: Path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"{path} is missing vertex property '{required}'")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
        if colors.max() > 1.5:
            colors /= 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        colors = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
    else:
        colors = np.ones((len(xyz), 3), dtype=np.float32)

    return ply, vertex, xyz, colors


def random_sample_indices(num_total: int, num_points: int, seed: int) -> np.ndarray:
    if num_points >= num_total:
        return np.arange(num_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_total, size=num_points, replace=False)).astype(np.int64)


def fps_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Farthest point sampling for prompt placement on the sampled cloud."""
    n = len(points)
    count = min(count, n)
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    min_dist2 = np.full((n,), np.inf, dtype=np.float32)

    for i in range(1, count):
        diff = points - points[selected[i - 1]]
        dist2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def batched(iterable: Iterable[int], batch_size: int):
    batch = []
    for value in iterable:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_enhanced_features(
    model: S2AM3DInference,
    point_feat: np.ndarray,
    point_coords: np.ndarray,
    point_color: np.ndarray,
    continuous_scale: float | None,
) -> torch.Tensor:
    point_feat_t = torch.from_numpy(point_feat).float().unsqueeze(0).to(model.device)
    point_coords_t = torch.from_numpy(point_coords).float().unsqueeze(0).to(model.device)
    point_color_t = torch.from_numpy(point_color).float().unsqueeze(0).to(model.device)

    continuous_scales = None
    if continuous_scale is not None and model.config.get("use_continuous_scale", True):
        continuous_scales = torch.tensor([continuous_scale], device=model.device).float()

    with torch.no_grad():
        return model.PointFeatureEnhancer(point_feat_t, point_coords_t, point_color_t, continuous_scales)


def predict_from_enhanced(
    model: S2AM3DInference,
    enhance_feat: torch.Tensor,
    prompt_idx: int,
    threshold: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    num_points = enhance_feat.shape[1]
    prompt_feat = enhance_feat[:, prompt_idx : prompt_idx + 1, :]
    with torch.no_grad():
        decoder_output = model.decoder(enhance_feat, prompt_feat)
        seg_pred = model.seg_head(decoder_output)[0].detach().cpu().numpy()
    mask = seg_pred > threshold
    if mask.any():
        confidence = float(seg_pred[mask].mean())
    else:
        confidence = float(seg_pred.max())
    return mask, confidence, seg_pred


def estimate_spacing(points: np.ndarray, k: int = 8) -> float:
    if len(points) <= 1:
        return 1.0
    n_neighbors = min(k + 1, len(points))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(points)
    distances = nn.kneighbors(points, return_distance=True)[0][:, -1]
    return float(np.median(distances))


def prompt_connected_component(
    mask: np.ndarray,
    coords: np.ndarray,
    prompt_idx: int,
    radius: float,
) -> np.ndarray:
    """Keep only the spatial component around the prompt point inside a mask."""
    inside = np.flatnonzero(mask)
    if len(inside) <= 1:
        return mask

    coords_inside = coords[inside]
    if mask[prompt_idx]:
        anchor = int(np.flatnonzero(inside == prompt_idx)[0])
    else:
        diff = coords_inside - coords[prompt_idx]
        anchor = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))

    nn = NearestNeighbors(radius=radius, algorithm="auto")
    nn.fit(coords_inside)
    neighbors = nn.radius_neighbors(coords_inside, return_distance=False)

    visited = np.zeros((len(inside),), dtype=bool)
    visited[anchor] = True
    stack = [anchor]
    while stack:
        current = stack.pop()
        for nb in neighbors[current]:
            nb = int(nb)
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)

    component = np.zeros_like(mask, dtype=bool)
    component[inside[visited]] = True
    return component


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def nms_masks(candidates: list[dict], iou_threshold: float) -> list[dict]:
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []
    for candidate in candidates:
        if all(mask_iou(candidate["mask"], kept_item["mask"]) < iou_threshold for kept_item in kept):
            kept.append(candidate)
    return kept


def assign_instances(num_points: int, masks: list[dict], mode: str) -> np.ndarray:
    instance_ids = np.zeros((num_points,), dtype=np.int32)
    if mode == "score":
        score_map = np.full((num_points,), -np.inf, dtype=np.float32)
        for instance_id, item in enumerate(masks, start=1):
            mask = item["mask"]
            scores = item["scores"]
            update = mask & (scores > score_map)
            instance_ids[update] = instance_id
            score_map[update] = scores[update]
    elif mode == "order":
        for instance_id, item in enumerate(masks, start=1):
            update = item["mask"] & (instance_ids == 0)
            instance_ids[update] = instance_id
    else:
        raise ValueError(f"Unknown assignment mode: {mode}")
    return instance_ids


def palette(instance_ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(instance_ids), 3), dtype=np.uint8)
    ids = np.unique(instance_ids)
    ids = ids[ids > 0]
    for idx in ids:
        hue = (int(idx) * 0.618033988749895) % 1.0
        rgb = hsv_to_rgb(hue, 0.70, 0.95)
        colors[instance_ids == idx] = np.asarray(rgb) * 255
    return colors


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


def write_xyzrgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    data = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def reset_generated_dir(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def write_gof_subset(path: Path, vertex: np.ndarray, indices: np.ndarray, as_text: bool):
    subset = vertex[indices]
    PlyData([PlyElement.describe(subset, "vertex")], text=as_text).write(str(path))


def propagate_to_full(
    full_xyz_norm: np.ndarray,
    sample_xyz_norm: np.ndarray,
    sample_instance_ids: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(sample_xyz_norm)
    full_instance_ids = np.zeros((len(full_xyz_norm),), dtype=np.int32)
    for start in range(0, len(full_xyz_norm), chunk_size):
        end = min(start + chunk_size, len(full_xyz_norm))
        nearest = nn.kneighbors(full_xyz_norm[start:end], return_distance=False)[:, 0]
        full_instance_ids[start:end] = sample_instance_ids[nearest]
    return full_instance_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic S2AM3D segmentation for GOF PLY files")
    parser.add_argument("--input_ply", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "decoder" / "configs" / "train.yaml")
    parser.add_argument("--decoder_ckpt", type=Path, default=ROOT / "ckpt" / "S2AM3D_decoder.pt")
    parser.add_argument("--encoder_config", type=Path, default=ROOT / "encoder" / "configs" / "final" / "demo.yaml")
    parser.add_argument("--encoder_ckpt", type=Path, default=ROOT / "ckpt" / "Encoder.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.3])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_mask_points", type=int, default=80)
    parser.add_argument("--max_mask_ratio", type=float, default=0.60)
    parser.add_argument("--nms_iou", type=float, default=0.80)
    parser.add_argument("--prompt_component", action="store_true")
    parser.add_argument("--component_radius", type=float, default=None)
    parser.add_argument("--component_radius_factor", type=float, default=8.0)
    parser.add_argument("--component_spacing_k", type=int, default=8)
    parser.add_argument("--assign_mode", choices=["score", "order"], default="score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--propagate_chunk_size", type=int, default=200000)
    parser.add_argument("--max_export_instances", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = args.output_dir / "gof_instances"
    reset_generated_dir(instance_dir)
    instance_dir.mkdir(exist_ok=True)

    print(f"[1/7] Loading GOF PLY: {args.input_ply}")
    ply, vertex, full_xyz, full_colors = load_gof_ply(args.input_ply)
    print(f"      full points: {len(full_xyz):,}")

    print(f"[2/7] Sampling {min(args.num_points, len(full_xyz)):,} points for S2AM3D")
    full_xyz_norm = normalize_pc(full_xyz).astype(np.float32)
    sample_indices = random_sample_indices(len(full_xyz), args.num_points, args.seed)
    sample_xyz = full_xyz[sample_indices]
    sample_colors = full_colors[sample_indices]
    sample_xyz_norm = full_xyz_norm[sample_indices]
    component_radius = args.component_radius
    if component_radius is None:
        spacing = estimate_spacing(sample_xyz_norm, k=args.component_spacing_k)
        component_radius = spacing * args.component_radius_factor
    print(f"      component radius: {component_radius:.5f}")

    print("[3/7] Loading S2AM3D encoder/decoder")
    decoder_cli_args, decoder_extras = parse_decoder_args([str(args.config)])
    decoder_config = load_config(str(args.config), cli_args=vars(decoder_cli_args), extra_args=decoder_extras)
    segmenter = S2AM3DInference(decoder_config, str(args.decoder_ckpt), device=args.device)
    feature_extractor = FeatureExtractor(str(args.encoder_config), str(args.encoder_ckpt), device=args.device)

    print("[4/7] Extracting point features")
    sample_feats = feature_extractor.extract_features(sample_xyz, sample_colors).astype(np.float32)

    print("[5/7] Creating automatic prompt points")
    prompt_indices = fps_indices(sample_xyz_norm, args.num_prompts, args.seed)
    candidates = []
    min_count = max(1, args.min_mask_points)
    max_count = int(len(sample_xyz_norm) * args.max_mask_ratio)

    for scale in args.scales:
        print(f"      scale={scale}: enhancing features once, then decoding {len(prompt_indices)} prompts")
        enhanced = compute_enhanced_features(segmenter, sample_feats, sample_xyz_norm, sample_colors, scale)
        for batch in batched(prompt_indices.tolist(), 8):
            for prompt_idx in batch:
                mask, confidence, scores = predict_from_enhanced(segmenter, enhanced, prompt_idx, args.threshold)
                raw_count = int(mask.sum())
                if args.prompt_component:
                    mask = prompt_connected_component(mask, sample_xyz_norm, prompt_idx, component_radius)
                    if mask.any():
                        confidence = float(scores[mask].mean())
                count = int(mask.sum())
                if count < min_count or count > max_count:
                    continue
                candidates.append(
                    {
                        "mask": mask,
                        "scores": scores.astype(np.float32),
                        "score": confidence,
                        "prompt_idx": int(prompt_idx),
                        "scale": float(scale),
                        "count": count,
                        "raw_count": raw_count,
                    }
                )
        del enhanced
        torch.cuda.empty_cache()

    print(f"[6/7] NMS on {len(candidates)} candidate masks")
    kept = nms_masks(candidates, args.nms_iou)
    sample_instance_ids = assign_instances(len(sample_xyz_norm), kept, args.assign_mode)
    print(f"      kept instances: {len(kept)}")

    print("[7/7] Propagating sampled masks back to full GOF points")
    full_instance_ids = propagate_to_full(
        full_xyz_norm,
        sample_xyz_norm,
        sample_instance_ids,
        chunk_size=args.propagate_chunk_size,
    )

    np.save(args.output_dir / "sample_indices.npy", sample_indices)
    np.save(args.output_dir / "sample_instance_ids.npy", sample_instance_ids)
    np.save(args.output_dir / "full_instance_ids.npy", full_instance_ids)
    np.save(args.output_dir / "prompt_indices_in_sample.npy", prompt_indices)
    if kept:
        np.save(args.output_dir / "sample_masks.npy", np.stack([item["mask"] for item in kept], axis=0))
    else:
        np.save(args.output_dir / "sample_masks.npy", np.zeros((0, len(sample_xyz_norm)), dtype=bool))

    write_xyzrgb_ply(args.output_dir / "sample_instances_colored.ply", sample_xyz, palette(sample_instance_ids))
    write_xyzrgb_ply(args.output_dir / "full_instances_colored.ply", full_xyz, palette(full_instance_ids))

    exported = 0
    instance_summaries = []
    for instance_id in range(1, len(kept) + 1):
        full_indices = np.flatnonzero(full_instance_ids == instance_id).astype(np.int64)
        if len(full_indices) == 0:
            continue
        if exported < args.max_export_instances:
            write_gof_subset(
                instance_dir / f"instance_{instance_id:03d}_gof.ply",
                vertex,
                full_indices,
                as_text=ply.text,
            )
            exported += 1
        item = kept[instance_id - 1]
        instance_summaries.append(
            {
                "instance_id": instance_id,
                "full_point_count": int(len(full_indices)),
                "sample_point_count": int((sample_instance_ids == instance_id).sum()),
                "prompt_idx_in_sample": int(item["prompt_idx"]),
                "prompt_original_index": int(sample_indices[item["prompt_idx"]]),
                "scale": float(item["scale"]),
                "score": float(item["score"]),
                "raw_sample_point_count": int(item.get("raw_count", item["count"])),
            }
        )

    metadata = {
        "input_ply": str(args.input_ply),
        "num_full_points": int(len(full_xyz)),
        "num_sample_points": int(len(sample_xyz)),
        "num_prompts": int(len(prompt_indices)),
        "scales": args.scales,
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "prompt_component": args.prompt_component,
        "component_radius": float(component_radius),
        "assign_mode": args.assign_mode,
        "num_instances": int(len(kept)),
        "exported_instance_plys": int(exported),
        "instances": instance_summaries,
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output dir: {args.output_dir}")
    print(f"Colored full cloud: {args.output_dir / 'full_instances_colored.ply'}")
    print(f"GOF instance subsets: {instance_dir}")


if __name__ == "__main__":
    main()
