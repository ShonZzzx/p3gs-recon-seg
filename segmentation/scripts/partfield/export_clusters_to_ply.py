import argparse
import csv
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_uid(path):
    match = re.match(r"(plant_\d+)", Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse uid from {path}")
    return match.group(1)


def find_label_file(cluster_dir, uid):
    files = sorted((Path(cluster_dir) / "cluster_out").glob(f"{uid}_0_auto_k*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one label npy for {uid}, found {len(files)}")
    return files[0]


def write_vertex_subset(out_path, vertex, mask):
    subset = vertex[mask]
    PlyData([PlyElement.describe(subset, "vertex")], text=False).write(out_path)


def export_one(source_path, label_path, out_dir):
    uid = parse_uid(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(source_path)
    vertex = ply["vertex"].data
    labels = np.load(label_path).reshape(-1).astype(np.int64)

    if len(vertex) != len(labels):
        n = min(len(vertex), len(labels))
        print(f"Warning: {uid} point/label mismatch {len(vertex)} vs {len(labels)}, truncating to {n}")
        vertex = vertex[:n]
        labels = labels[:n]

    rows = []
    for cluster_id in sorted(np.unique(labels).tolist()):
        mask = labels == cluster_id
        out_path = out_dir / f"cluster_{cluster_id:02d}.ply"
        write_vertex_subset(out_path, vertex, mask)
        rows.append(
            {
                "uid": uid,
                "cluster_id": int(cluster_id),
                "n_points": int(mask.sum()),
                "ply": str(out_path),
            }
        )
        print(f"{uid} cluster {cluster_id:02d}: {int(mask.sum())} points -> {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Directory containing source PLY files.")
    parser.add_argument("--cluster-dir", required=True, help="Directory containing cluster_out/*.npy.")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-cluster PLY files.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    cluster_dir = Path(args.cluster_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for source_path in sorted(source_dir.glob("*.ply")):
        uid = parse_uid(source_path)
        label_path = find_label_file(cluster_dir, uid)
        all_rows.extend(export_one(source_path, label_path, out_root / uid))

    summary_path = out_root / "cluster_export_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "cluster_id", "n_points", "ply"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
