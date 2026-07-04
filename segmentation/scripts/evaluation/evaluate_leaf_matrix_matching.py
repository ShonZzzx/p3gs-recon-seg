import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_uid(path):
    m = re.match(r'(plant_\d+)', Path(path).name)
    if not m:
        raise ValueError(f'Cannot parse uid from {path}')
    return m.group(1)


def find_pred(cluster_dir, uid):
    files = sorted((Path(cluster_dir) / 'cluster_out').glob(f'{uid}_0_auto_k*.npy'))
    if len(files) != 1:
        return None
    return files[0]


def read_metrics(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[0]


def mean_row(rows):
    out = {'uid': 'MEAN'}
    keys = rows[0].keys()
    for k in keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if vals and len(vals) == len(rows):
            out[k] = float(np.mean(vals))
        elif k not in out:
            out[k] = ''
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--cluster-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iou-leaf-threshold', type=float, default=0.1)
    ap.add_argument('--max-distance', type=float, default=-1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    audit_root = out_dir / 'audit_samples'
    audit_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / 'audit_leaf_matching_one.py'
    rows = []
    for gt_path in sorted(Path(args.gt_dir).glob('*_labeled.ply')):
        uid = parse_uid(gt_path)
        source_path = Path(args.source_dir) / f'{uid}.ply'
        pred_path = find_pred(args.cluster_dir, uid)
        if not source_path.exists():
            print(f'Skip {uid}: missing source {source_path}')
            continue
        if pred_path is None:
            print(f'Skip {uid}: missing unique prediction in {args.cluster_dir}')
            continue

        sample_out = audit_root / uid
        cmd = [
            sys.executable,
            str(script),
            '--uid', uid,
            '--gt', str(gt_path),
            '--source', str(source_path),
            '--pred-npy', str(pred_path),
            '--out-dir', str(sample_out),
            '--iou-leaf-threshold', str(args.iou_leaf_threshold),
            '--max-distance', str(args.max_distance),
        ]
        print('Running', uid)
        subprocess.run(cmd, check=True)
        rows.append(read_metrics(sample_out / 'metrics.csv'))

    if not rows:
        raise RuntimeError('No samples evaluated')
    rows.append(mean_row(rows))

    out_csv = out_dir / 'evaluation_matrix_matching.csv'
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
