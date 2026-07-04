#!/usr/bin/env python3
"""Batch create fine-grained runtime refined results without aggressive merging."""

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
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    plys = sorted(args.input_dir.rglob(args.pattern))
    print(f"Found {len(plys)} PLY files", flush=True)
    for index, ply in enumerate(plys, start=1):
        stem = ply.stem
        base_dir = args.output_root / f"{stem}_runtime_base"
        fine_dir = args.output_root / f"{stem}_runtime_fine"
        if not (base_dir / "metadata.json").exists():
            raise FileNotFoundError(f"Missing runtime_base result: {base_dir}")
        if args.skip_existing and (fine_dir / "refine_metadata.json").exists():
            print(f"[{index}/{len(plys)}] Skip existing: {fine_dir}", flush=True)
            continue
        print(f"[{index}/{len(plys)}] Fine refinement: {stem}", flush=True)
        run([
            sys.executable,
            str(ROOT / "refine_s2am3d_instances.py"),
            "--input_ply", str(ply),
            "--seg_dir", str(base_dir),
            "--output_dir", str(fine_dir),
            "--knn", "10",
            "--min_points", "80",
            "--merge_ratio", "999",
            "--merge_min_edges", "999999",
            "--smooth_iterations", "1",
            "--smooth_min_votes", "7",
        ])
    print("\nBatch fine refinement done.", flush=True)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()
