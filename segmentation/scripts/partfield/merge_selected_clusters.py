import argparse
import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def read_selection(path):
    selections = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"].strip()
            ids = [
                int(x.strip())
                for x in row["cluster_ids"].replace("，", ",").split(",")
                if x.strip()
            ]
            selections[uid] = ids
    return selections


def find_label_file(cluster_dir, uid):
    files = sorted((Path(cluster_dir) / "cluster_out").glob(f"{uid}_0_auto_k*.npy"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one label npy for {uid}, found {len(files)}")
    return files[0]


def find_feature_file(features_dir, uid):
    files = sorted(Path(features_dir).glob(f"part_feat_{uid}_0.npy"))
    if len(files) != 1:
        return None
    return files[0]


def merge_one(uid, cluster_ids, source_dir, cluster_dir, out_data_dir, features_dir, out_features_dir):
    source_path = source_dir / f"{uid}.ply"
    label_path = find_label_file(cluster_dir, uid)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    ply = PlyData.read(source_path)
    vertex = ply["vertex"].data
    labels = np.load(label_path).reshape(-1).astype(np.int64)

    n = min(len(vertex), len(labels))
    if len(vertex) != len(labels):
        print(f"Warning: {uid} point/label mismatch {len(vertex)} vs {len(labels)}, truncating to {n}")
    vertex = vertex[:n]
    labels = labels[:n]

    keep = np.isin(labels, np.asarray(cluster_ids, dtype=np.int64))
    selected_vertex = vertex[keep]
    out_ply = out_data_dir / f"{uid}.ply"
    PlyData([PlyElement.describe(selected_vertex, "vertex")], text=False).write(out_ply)

    feature_points = ""
    feature_path = None
    out_feature = None
    if features_dir is not None and out_features_dir is not None:
        feature_path = find_feature_file(features_dir, uid)
        if feature_path is not None:
            feat = np.load(feature_path)
            n_feat = min(len(feat), len(keep))
            if len(feat) != len(keep):
                print(f"Warning: {uid} feature/mask mismatch {len(feat)} vs {len(keep)}, truncating to {n_feat}")
            feat = feat[:n_feat]
            keep_feat = keep[:n_feat]
            out_feature = out_features_dir / f"part_feat_{uid}_0.npy"
            np.save(out_feature, feat[keep_feat])
            feature_points = int(keep_feat.sum())

    return {
        "uid": uid,
        "cluster_ids": ",".join(str(x) for x in cluster_ids),
        "raw_points": int(n),
        "selected_points": int(keep.sum()),
        "selected_ratio": float(keep.sum() / n) if n else 0.0,
        "out_ply": str(out_ply),
        "feature_points": feature_points,
        "out_feature": "" if out_feature is None else str(out_feature),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cluster-dir", required=True)
    parser.add_argument("--out-data-dir", required=True)
    parser.add_argument("--features-dir")
    parser.add_argument("--out-features-dir")
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    selections = read_selection(args.selection_csv)
    source_dir = Path(args.source_dir)
    cluster_dir = Path(args.cluster_dir)
    out_data_dir = Path(args.out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(args.features_dir) if args.features_dir else None
    out_features_dir = Path(args.out_features_dir) if args.out_features_dir else None
    if out_features_dir is not None:
        out_features_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for uid in sorted(selections):
        row = merge_one(
            uid,
            selections[uid],
            source_dir,
            cluster_dir,
            out_data_dir,
            features_dir,
            out_features_dir,
        )
        rows.append(row)
        print(f"{uid}: kept {row['selected_points']} / {row['raw_points']} points -> {row['out_ply']}")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid",
                "cluster_ids",
                "raw_points",
                "selected_points",
                "selected_ratio",
                "out_ply",
                "feature_points",
                "out_feature",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
