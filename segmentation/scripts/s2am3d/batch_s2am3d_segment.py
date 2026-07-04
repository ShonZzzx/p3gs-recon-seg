#!/usr/bin/env python3
"""Batch runner for S2AM3D automatic PLY segmentation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Batch segment PLY files with S2AM3D")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_mask_points", type=int, default=80)
    parser.add_argument("--max_mask_ratio", type=float, default=0.6)
    parser.add_argument("--nms_iou", type=float, default=0.8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    plys = sorted(args.input_dir.rglob(args.pattern))
    if not plys:
        raise FileNotFoundError(f"No PLY files found under {args.input_dir} with pattern {args.pattern}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(plys)} PLY files", flush=True)

    for index, ply in enumerate(plys, start=1):
        out_dir = args.output_root / ply.stem
        done_marker = out_dir / "metadata.json"
        if args.skip_existing and done_marker.exists():
            print(f"[{index}/{len(plys)}] Skip existing: {ply}", flush=True)
            continue

        print(f"\n[{index}/{len(plys)}] Segmenting {ply}", flush=True)
        run(
            [
                sys.executable,
                str(ROOT / "gof_s2am3d_auto_segment.py"),
                "--input_ply",
                str(ply),
                "--output_dir",
                str(out_dir),
                "--num_points",
                str(args.num_points),
                "--num_prompts",
                str(args.num_prompts),
                "--scales",
                *[str(scale) for scale in args.scales],
                "--threshold",
                str(args.threshold),
                "--min_mask_points",
                str(args.min_mask_points),
                "--max_mask_ratio",
                str(args.max_mask_ratio),
                "--nms_iou",
                str(args.nms_iou),
                "--device",
                args.device,
            ]
        )

        print(f"[{index}/{len(plys)}] Creating CloudCompare view files", flush=True)
        run(
            [
                sys.executable,
                str(ROOT / "make_s2am3d_view_files.py"),
                "--input_ply",
                str(ply),
                "--seg_dir",
                str(out_dir),
            ]
        )

    print("\nBatch done.", flush=True)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()
