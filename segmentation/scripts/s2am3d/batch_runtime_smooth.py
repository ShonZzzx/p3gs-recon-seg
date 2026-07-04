#!/usr/bin/env python3
"""Batch run the runtime-improved + smooth-refined S2AM3D pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
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
        raise FileNotFoundError(f"No PLY files found under {args.input_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(plys)} PLY files", flush=True)

    for index, ply in enumerate(plys, start=1):
        stem = ply.stem
        base_dir = args.output_root / f"{stem}_runtime_base"
        smooth_dir = args.output_root / f"{stem}_runtime_smooth"

        if args.skip_existing and (smooth_dir / "refine_metadata.json").exists():
            print(f"[{index}/{len(plys)}] Skip existing smooth result: {smooth_dir}", flush=True)
            continue

        print(f"\n[{index}/{len(plys)}] Runtime improved inference: {ply}", flush=True)
        if not (args.skip_existing and (base_dir / "metadata.json").exists()):
            run(
                [
                    sys.executable,
                    str(ROOT / "gof_s2am3d_auto_segment.py"),
                    "--input_ply",
                    str(ply),
                    "--output_dir",
                    str(base_dir),
                    "--num_points",
                    "20000",
                    "--num_prompts",
                    "32",
                    "--scales",
                    "0.2",
                    "0.3",
                    "0.5",
                    "--threshold",
                    "0.5",
                    "--min_mask_points",
                    "160",
                    "--max_mask_ratio",
                    "0.70",
                    "--nms_iou",
                    "0.70",
                    "--prompt_component",
                    "--component_radius_factor",
                    "8",
                    "--assign_mode",
                    "order",
                    "--device",
                    args.device,
                ]
            )
        else:
            print(f"Base exists, skip: {base_dir}", flush=True)

        print(f"[{index}/{len(plys)}] Smooth refinement: {base_dir}", flush=True)
        run(
            [
                sys.executable,
                str(ROOT / "refine_s2am3d_instances.py"),
                "--input_ply",
                str(ply),
                "--seg_dir",
                str(base_dir),
                "--output_dir",
                str(smooth_dir),
                "--knn",
                "16",
                "--min_points",
                "120",
                "--merge_ratio",
                "0.95",
                "--merge_min_edges",
                "1000",
                "--smooth_iterations",
                "2",
                "--smooth_min_votes",
                "10",
            ]
        )

    print("\nBatch done.", flush=True)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()
