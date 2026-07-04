#!/usr/bin/env python3
"""Merge saved interactive S2AM3D click masks into one instance-labeled result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


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


def palette(ids: np.ndarray) -> np.ndarray:
    rgb = np.zeros((len(ids), 3), dtype=np.uint8)
    rgb[:] = np.array([150, 150, 150], dtype=np.uint8)
    for instance_id in [int(x) for x in np.unique(ids) if x > 0]:
        hue = (instance_id * 0.618033988749895) % 1.0
        rgb[ids == instance_id] = np.asarray(hsv_to_rgb(hue, 0.75, 0.95)) * 255
    return rgb


def read_xyz(path: Path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    return ply, vertex, xyz


def write_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    data = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def write_scalar(path: Path, xyz: np.ndarray, ids: np.ndarray):
    data = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("instance_id", "f4")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["instance_id"] = ids.astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge saved UI click masks")
    parser.add_argument("--click_root", required=True, type=Path, help="Directory containing click_001, click_002, ...")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overlap", choices=["later", "earlier", "background"], default="later")
    return parser.parse_args()


def main():
    args = parse_args()
    click_dirs = sorted([p for p in args.click_root.glob("click_*") if p.is_dir()])
    if not click_dirs:
        raise FileNotFoundError(f"No click_* directories under {args.click_root}")

    output_dir = args.output_dir or (args.click_root / "merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = output_dir / "gof_instances"
    instance_dir.mkdir(exist_ok=True)

    metas = []
    masks = []
    source_ply = None
    for click_dir in click_dirs:
        meta_path = click_dir / "metadata.json"
        mask_path = click_dir / "full_mask.npy"
        if not meta_path.exists() or not mask_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if source_ply is None:
            source_ply = meta["source_ply"]
        elif source_ply != meta["source_ply"]:
            raise ValueError("All clicks must come from the same source_ply")
        mask = np.load(mask_path).astype(bool)
        metas.append((click_dir.name, meta))
        masks.append(mask)

    if not masks:
        raise FileNotFoundError(f"No complete click masks found under {args.click_root}")

    ply, vertex, xyz = read_xyz(Path(source_ply))
    ids = np.zeros(len(xyz), dtype=np.int32)

    for idx, mask in enumerate(masks, start=1):
        if len(mask) != len(ids):
            raise ValueError(f"Mask {idx} length {len(mask)} != point count {len(ids)}")
        if args.overlap == "later":
            ids[mask] = idx
        elif args.overlap == "earlier":
            ids[mask & (ids == 0)] = idx
        else:
            ids[mask & (ids == 0)] = idx
            overlap = mask & (ids != idx)
            ids[overlap] = 0

    write_xyzrgb(output_dir / "merged_instances_colored.ply", xyz, palette(ids))
    write_scalar(output_dir / "merged_instance_id_scalar.ply", xyz, ids)
    np.save(output_dir / "merged_instance_ids.npy", ids)

    rows = []
    for instance_id in [int(x) for x in np.unique(ids) if x > 0]:
        point_idx = np.flatnonzero(ids == instance_id)
        subset = vertex[point_idx]
        gof_path = instance_dir / f"instance_{instance_id:03d}_gof.ply"
        PlyData([PlyElement.describe(subset, "vertex")], text=ply.text).write(str(gof_path))
        click_name, meta = metas[instance_id - 1]
        rows.append([instance_id, click_name, int(len(point_idx)), str(gof_path), meta.get("scale"), meta.get("confidence")])

    with open(output_dir / "instance_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "click", "full_point_count", "gof_subset_path", "scale", "confidence"])
        writer.writerows(rows)

    metadata = {
        "click_root": str(args.click_root),
        "source_ply": source_ply,
        "num_clicks": len(masks),
        "num_instances": int(len([x for x in np.unique(ids) if x > 0])),
        "overlap": args.overlap,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"merged {len(masks)} clicks -> {metadata['num_instances']} instances")
    print(output_dir)


if __name__ == "__main__":
    main()
