#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = [
    "dataset",
    "method",
    "status",
    "failure_reason",
    "train_time_sec",
    "model_size_mb",
    "point_cloud_path",
    "point_count",
    "outlier_ratio",
    "main_component_ratio",
    "bbox_volume",
    "local_thickness_median",
    "render_edge_strength",
    "PSNR",
    "SSIM",
    "LPIPS",
    "num_test_images",
    "structure_error",
    "render_error",
    "metric_error",
]


def read_ply_xyz(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return np.stack([np.asarray(vertex[c], dtype=np.float64) for c in ("x", "y", "z")], axis=1)


def vertex_count_from_header(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise RuntimeError(f"element vertex not found in {path}")


def find_point_cloud(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        model_dir / "point_cloud" / "iteration_30000" / "apps.ply",
        model_dir / "point_cloud" / "iteration_30000" / "stprs.ply",
        model_dir / "fused_point_cloud.ply",
    ]
    for item in candidates:
        if item.exists():
            return item
    pcs = sorted(model_dir.glob("**/*.ply"))
    return pcs[-1] if pcs else None


def mad_outlier_ratio(xyz: np.ndarray, k: int = 16) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return 0.0
    nbr = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xyz)
    dist, _ = nbr.kneighbors(xyz)
    mean_dist = dist[:, 1:].mean(axis=1)
    med = np.median(mean_dist)
    mad = np.median(np.abs(mean_dist - med))
    if mad == 0:
        return 0.0
    threshold = med + 3.0 * mad
    return float(np.mean(mean_dist > threshold))


def main_component_ratio(xyz: np.ndarray) -> float:
    from sklearn.cluster import DBSCAN

    if len(xyz) == 0:
        return math.nan
    if len(xyz) > 50000:
        rng = np.random.default_rng(20260605)
        xyz = xyz[rng.choice(len(xyz), size=50000, replace=False)]
    span = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
    eps = max(span * 0.02, 1e-6)
    labels = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit_predict(xyz)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return 0.0
    counts = np.bincount(valid)
    return float(counts.max() / len(labels))


def local_thickness_median(xyz: np.ndarray, sample_n: int = 50000, k: int = 16) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(xyz) <= k:
        return math.nan
    if len(xyz) > sample_n:
        rng = np.random.default_rng(20260605)
        sample = xyz[rng.choice(len(xyz), size=sample_n, replace=False)]
    else:
        sample = xyz
    nbr = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    _, idx = nbr.kneighbors(sample)
    vals = []
    for group in idx:
        pts = xyz[group]
        cov = np.cov((pts - pts.mean(axis=0)).T)
        eigvals = np.linalg.eigvalsh(cov)
        vals.append(max(eigvals[0], 0.0) ** 0.5)
    return float(np.median(vals))


def edge_strength(render_dir: Path) -> float:
    from skimage import filters

    vals = []
    for path in sorted(render_dir.glob("*.png")):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        sobel = filters.sobel(img)
        lap = np.abs(filters.laplace(img))
        vals.append(float((sobel.mean() + lap.mean()) * 0.5))
    return float(np.mean(vals)) if vals else math.nan


def image_metrics(render_dir: Path, gt_dir: Path) -> dict:
    import torch
    import torchvision.transforms.functional as tf
    from skimage.metrics import structural_similarity

    try:
        import lpips
        lpips_device = torch.device("cpu")
        lpips_fn = lpips.LPIPS(net="vgg").to(lpips_device).eval()
    except Exception as exc:
        lpips_fn = None
        lpips_error = str(exc)
    else:
        lpips_error = ""

    psnrs, ssims, lpipss = [], [], []
    names = [p.name for p in sorted(render_dir.glob("*.png")) if (gt_dir / p.name).exists()]
    if not names:
        return {"metric_error": "no matched render/gt png files"}

    for name in names:
        r = np.asarray(Image.open(render_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        mse = float(np.mean((r - g) ** 2))
        psnrs.append(100.0 if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse)))
        ssims.append(float(structural_similarity(r, g, channel_axis=2, data_range=1.0)))
        if lpips_fn is not None:
            try:
                with torch.no_grad():
                    rt = tf.to_tensor(Image.open(render_dir / name).convert("RGB")).unsqueeze(0).to(lpips_device) * 2 - 1
                    gt = tf.to_tensor(Image.open(gt_dir / name).convert("RGB")).unsqueeze(0).to(lpips_device) * 2 - 1
                    lpipss.append(float(lpips_fn(rt, gt).detach().cpu().item()))
            except Exception as exc:
                lpips_error = str(exc)
                lpips_fn = None

    out = {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "num_test_images": len(names),
    }
    if lpipss:
        out["LPIPS"] = float(np.mean(lpipss))
    else:
        out["LPIPS"] = None
        out["metric_error"] = f"LPIPS failed: {lpips_error}"
    return out


def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total / 1024 / 1024


def parse_train_time(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    match = re.search(r"TRAIN_TIME_SEC=([0-9.]+)", text)
    return float(match.group(1)) if match else None


def collect_one(root: Path, method: str, plant: str) -> dict:
    run_dir = root / method / plant
    model_dir = run_dir / "model"
    if method == "fused":
        model_dir = run_dir
    render_dir = run_dir / "renders" / "test"
    gt_dir = run_dir / "gt" / "test"
    if method == "fused":
        render_dir = run_dir / "pointcloud_views"
        gt_dir = None
    log_path = run_dir / "logs" / "train.log"
    row = {
        "dataset": plant,
        "method": method,
        "status": "pending",
        "failure_reason": None,
        "train_time_sec": parse_train_time(log_path),
        "model_size_mb": dir_size_mb(model_dir) if model_dir.exists() else None,
    }
    pc = find_point_cloud(model_dir)
    if pc is None:
        row["structure_error"] = "point cloud not found"
    else:
        row["point_cloud_path"] = str(pc)
        row["point_count"] = vertex_count_from_header(pc)
        xyz = read_ply_xyz(pc)
        row["bbox_volume"] = float(np.prod(np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 0.0))) if len(xyz) else 0.0
        row["outlier_ratio"] = mad_outlier_ratio(xyz)
        row["main_component_ratio"] = main_component_ratio(xyz)
        row["local_thickness_median"] = local_thickness_median(xyz)
    try:
        row["render_edge_strength"] = edge_strength(render_dir)
    except Exception as exc:
        row["render_edge_strength"] = None
        row["render_error"] = str(exc)
    if method != "fused":
        if render_dir.exists() and gt_dir is not None and gt_dir.exists():
            row.update(image_metrics(render_dir, gt_dir))
        else:
            row["metric_error"] = f"missing render or gt dir: {render_dir}, {gt_dir}"
    errors = [row.get("structure_error"), row.get("render_error"), row.get("metric_error")]
    errors = [item for item in errors if item]
    if pc is not None and not errors:
        row["status"] = "ok"
    elif model_dir.exists() or log_path.exists():
        row["status"] = "failed"
    row["failure_reason"] = "; ".join(errors) if errors else None
    return row


def write_summary(rows: list[dict], path: Path) -> None:
    fields = ["dataset", "method", "status", "PSNR", "SSIM", "LPIPS", "train_time_sec", "point_count", "failure_reason"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            elif value is None:
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", nargs="+", default=["radegs", "gof", "fused"])
    parser.add_argument("--plants", nargs="+", default=["plant_002", "plant_013", "plant_016", "plant_019"])
    args = parser.parse_args()
    root = Path(args.root)
    rows = [collect_one(root, method, plant) for plant in args.plants for method in args.methods]
    csv_path = root / "metrics_b1.csv"
    json_path = root / "metrics_b1.json"
    summary_path = root / "summary_b1.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(rows, summary_path)
    print(f"WROTE {csv_path}")
    print(f"WROTE {json_path}")
    print(f"WROTE {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
