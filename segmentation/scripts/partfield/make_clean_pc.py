import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def read_vertices(path):
    return PlyData.read(path)["vertex"].data


def write_xyzrgb(path, xyz, rgb):
    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def clean_one(path, out_dir, opacity_quantile, max_points):
    v = read_vertices(path)
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)
    opacity = np.asarray(v["opacity"], dtype=np.float32)

    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    xyz, rgb, opacity = xyz[finite], rgb[finite], opacity[finite]

    keep = opacity >= np.quantile(opacity, opacity_quantile)
    xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]

    lo = np.quantile(xyz, 0.005, axis=0)
    hi = np.quantile(xyz, 0.995, axis=0)
    crop = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    xyz, rgb, opacity = xyz[crop], rgb[crop], opacity[crop]

    if xyz.shape[0] > max_points:
        extent = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-6)
        volume = float(np.prod(extent))
        voxel = (volume / max_points) ** (1.0 / 3.0)
        origin = xyz.min(axis=0)

        for _ in range(8):
            keys = np.floor((xyz - origin) / voxel).astype(np.int64)
            order = np.lexsort((-opacity, keys[:, 2], keys[:, 1], keys[:, 0]))
            sorted_keys = keys[order]
            first = np.ones(sorted_keys.shape[0], dtype=bool)
            first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
            chosen = order[first]
            if chosen.shape[0] <= max_points * 1.05:
                break
            voxel *= (chosen.shape[0] / max_points) ** (1.0 / 3.0) * 1.03

        if chosen.shape[0] > max_points:
            chosen_order = np.argsort(opacity[chosen])[::-1][:max_points]
            chosen = chosen[chosen_order]
        xyz, rgb, opacity = xyz[chosen], rgb[chosen], opacity[chosen]

    out_path = out_dir / path.name
    write_xyzrgb(out_path, xyz, rgb)
    print(f"{path.name}: kept {xyz.shape[0]} points, opacity sigmoid mean {sigmoid(opacity).mean():.4f}, wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opacity-quantile", type=float, default=0.35)
    parser.add_argument("--max-points", type=int, default=350000)
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.ply")):
        clean_one(path, out_dir, args.opacity_quantile, args.max_points)


if __name__ == "__main__":
    main()
