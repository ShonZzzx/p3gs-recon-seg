#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True, type=Path)
    p.add_argument('--output_root', required=True, type=Path)
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()

def run(cmd):
    print('\n$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    args=parse_args()
    plys=sorted(args.input_dir.rglob('*.ply'))
    print(f'Found {len(plys)} PLY files', flush=True)
    for i, ply in enumerate(plys, 1):
        stem=ply.stem
        seg=args.output_root/f'{stem}_runtime_fine'
        out=args.output_root/f'{stem}_runtime_fine_split'
        if args.skip_existing and (out/'split_metadata.json').exists():
            print(f'[{i}/{len(plys)}] Skip existing: {out}', flush=True)
            continue
        print(f'[{i}/{len(plys)}] Split components: {stem}', flush=True)
        run([sys.executable, str(ROOT/'split_instance_components.py'), '--input_ply', str(ply), '--seg_dir', str(seg), '--output_dir', str(out), '--radius_factor', '5.0', '--min_component_points', '120', '--knn_cap', '16'])
    print('done', flush=True)
if __name__ == '__main__':
    main()
