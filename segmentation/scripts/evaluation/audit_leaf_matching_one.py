import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score


def safe_div(a, b):
    return float(a / b) if b else 0.0


def read_vertex(path):
    return PlyData.read(path)['vertex'].data


def xyz(v):
    return np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float64)


def load_gt(path):
    v = read_vertex(path)
    pts = xyz(v)
    names = v.dtype.names
    if 'scalar_label' in names:
        raw = np.asarray(v['scalar_label'], dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return pts, raw.astype(np.int64), 'scalar_label'
    rgb = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)
    unique = sorted(map(tuple, np.unique(rgb, axis=0).tolist()))
    mapping = {}
    next_id = 1
    for color in unique:
        if color == (255, 255, 255):
            mapping[color] = 0
        else:
            mapping[color] = next_id
            next_id += 1
    labels = np.array([mapping[tuple(c)] for c in rgb], dtype=np.int64)
    return pts, labels, 'rgb_color_map:' + json.dumps({str(k): v for k, v in mapping.items()}, ensure_ascii=False)


def write_xyz_labels(path, pts, labels, palette='tab20'):
    labels = labels.reshape(-1)
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap(palette, max(len(unique), 1))
    color_map = {}
    for i, lab in enumerate(unique):
        if lab == 0:
            color_map[lab] = np.array([220, 220, 220], dtype=np.uint8)
        else:
            color_map[lab] = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
    colors = np.array([color_map[x] for x in labels], dtype=np.uint8)
    vertex = np.empty(pts.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    vertex['x'], vertex['y'], vertex['z'] = pts[:,0], pts[:,1], pts[:,2]
    vertex['red'], vertex['green'], vertex['blue'] = colors[:,0], colors[:,1], colors[:,2]
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)


def compute_tables(gt_labels, pred_labels, extra_pred_labels=None):
    if extra_pred_labels is None:
        extra_pred_labels = np.array([], dtype=np.int64)
    gt_ids = np.array(sorted([x for x in np.unique(gt_labels) if x > 0]), dtype=np.int64)
    pred_ids = np.array(sorted([x for x in np.unique(np.concatenate([pred_labels, extra_pred_labels])) if x > 0]), dtype=np.int64)
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    pred_counts = np.zeros(len(pred_ids), dtype=np.int64)
    for j, pid in enumerate(pred_ids):
        pred_counts[j] = int(np.sum(pred_labels == pid) + np.sum(extra_pred_labels == pid))
    for i, gid in enumerate(gt_ids):
        g = gt_labels == gid
        for j, pid in enumerate(pred_ids):
            overlap[i, j] = int(np.sum(g & (pred_labels == pid)))
    iou = np.zeros_like(overlap, dtype=np.float64)
    cover_gt = np.zeros_like(overlap, dtype=np.float64)
    cover_pred = np.zeros_like(overlap, dtype=np.float64)
    for i, gid in enumerate(gt_ids):
        g = gt_labels == gid
        for j, pid in enumerate(pred_ids):
            inter = overlap[i, j]
            iou[i, j] = safe_div(inter, np.sum(g) + pred_counts[j] - inter)
            cover_gt[i, j] = safe_div(inter, np.sum(g))
            cover_pred[i, j] = safe_div(inter, pred_counts[j])
    return gt_ids, pred_ids, overlap, iou, cover_gt, cover_pred


def write_matrix_csv(path, row_ids, col_ids, matrix):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([r'GT\Pred'] + [f'P{p}' for p in col_ids])
        for rid, row in zip(row_ids, matrix):
            w.writerow([f'G{rid}'] + list(row))


def plot_heatmap(path, row_ids, col_ids, matrix, title, fmt='.2f'):
    fig_w = max(8, 0.45 * len(col_ids) + 2)
    fig_h = max(5, 0.35 * len(row_ids) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Pred cluster')
    ax.set_ylabel('GT leaf instance')
    ax.set_xticks(np.arange(len(col_ids)))
    ax.set_xticklabels([str(x) for x in col_ids], rotation=90)
    ax.set_yticks(np.arange(len(row_ids)))
    ax.set_yticklabels([str(x) for x in row_ids])
    fig.colorbar(im, ax=ax)
    if matrix.size and matrix.shape[0] <= 30 and matrix.shape[1] <= 35:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text = f'{val:{fmt}}' if fmt != 'd' else str(int(val))
                ax.text(j, i, text, ha='center', va='center', fontsize=6, color='white' if val > matrix.max() * 0.45 else 'black')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def semantic_from_iou(gt_ids, pred_ids, iou, threshold):
    pred_leaf_ids = set()
    pairs = []
    if iou.size:
        rows, cols = np.where(iou >= threshold)
        for r, c in zip(rows, cols):
            pred_leaf_ids.add(int(pred_ids[c]))
            pairs.append((int(gt_ids[r]), int(pred_ids[c]), float(iou[r, c])))
    return pred_leaf_ids, pairs


def one_to_one_matches(gt_ids, pred_ids, iou, threshold):
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    out = []
    for r, c in zip(rows, cols):
        if iou[r, c] >= threshold:
            out.append((int(gt_ids[r]), int(pred_ids[c]), float(iou[r, c])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uid', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--pred-npy', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-distance', type=float, default=-1.0)
    ap.add_argument('--iou-leaf-threshold', type=float, default=0.1, help='Pred cluster is leaf if IoU with any GT leaf >= this threshold.')
    ap.add_argument('--full-output', action='store_true', help='Write intermediate CSV files and PLY visualizations. Default keeps only metrics.csv and check PNGs.')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gt_pts, gt_labels, label_source = load_gt(args.gt)
    src_pts = xyz(read_vertex(args.source))
    pred_src = np.load(args.pred_npy).reshape(-1).astype(np.int64) + 1
    if len(src_pts) != len(pred_src):
        n = min(len(src_pts), len(pred_src))
        src_pts = src_pts[:n]
        pred_src = pred_src[:n]

    tree = cKDTree(src_pts)
    dists, nn = tree.query(gt_pts, k=1, workers=-1)
    max_dist = args.max_distance
    if max_dist < 0:
        max_dist = 0.002 * np.linalg.norm(gt_pts.max(axis=0) - gt_pts.min(axis=0))
    match_mask = dists <= max_dist
    pred_on_gt = np.zeros(len(gt_pts), dtype=np.int64)
    pred_on_gt[match_mask] = pred_src[nn[match_mask]]
    used_src = np.zeros(len(src_pts), dtype=bool)
    used_src[np.unique(nn[match_mask])] = True
    extra_pred = pred_src[~used_src]

    gt_ids, pred_ids, overlap, iou, cover_gt, cover_pred = compute_tables(gt_labels, pred_on_gt, extra_pred)
    leaf_pred_ids, many_to_many_pairs = semantic_from_iou(gt_ids, pred_ids, iou, args.iou_leaf_threshold)
    gt_leaf = gt_labels > 0
    pred_leaf = np.isin(pred_on_gt, list(leaf_pred_ids))
    extra_pred_leaf = np.isin(extra_pred, list(leaf_pred_ids))

    tp = int(np.sum(gt_leaf & pred_leaf))
    fp = int(np.sum(~gt_leaf & pred_leaf) + np.sum(extra_pred_leaf))
    fn = int(np.sum(gt_leaf & ~pred_leaf))
    tn = int(np.sum(~gt_leaf & ~pred_leaf) + np.sum(~extra_pred_leaf))
    precision = safe_div(tp, tp + fp); recall = safe_div(tp, tp + fn)
    metrics = {
        'uid': args.uid,
        'label_source': label_source,
        'n_gt_points': int(len(gt_pts)),
        'n_gt_leaf_points': int(np.sum(gt_leaf)),
        'n_source_points': int(len(src_pts)),
        'n_matched_points': int(np.sum(match_mask)),
        'n_source_only_points': int(len(extra_pred)),
        'n_eval_points': int(len(gt_pts) + len(extra_pred)),
        'match_rate': safe_div(np.sum(match_mask), len(gt_pts)),
        'max_distance': float(max_dist),
        'nn_dist_p50': float(np.quantile(dists, 0.5)),
        'nn_dist_p95': float(np.quantile(dists, 0.95)),
        'nn_dist_p99': float(np.quantile(dists, 0.99)),
        'n_gt_instances': int(len(gt_ids)),
        'n_pred_clusters': int(len(pred_ids)),
        'n_pred_leaf_clusters': int(len(leaf_pred_ids)),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Leaf_IoU': safe_div(tp, tp + fp + fn),
        'Precision': precision,
        'Recall': recall,
        'F1_Dice': safe_div(2 * precision * recall, precision + recall),
        'Accuracy': safe_div(tp + tn, tp + fp + fn + tn),
    }

    inst_iou = iou[:, [np.where(pred_ids == pid)[0][0] for pid in sorted(leaf_pred_ids) if pid in pred_ids]] if leaf_pred_ids else np.zeros((len(gt_ids), 0))
    inst_pred_ids = np.array(sorted([pid for pid in leaf_pred_ids if pid in pred_ids]), dtype=np.int64)
    for th in [0.5, 0.75]:
        matches = one_to_one_matches(gt_ids, inst_pred_ids, inst_iou, th)
        inst_tp = len(matches); inst_fp = len(inst_pred_ids) - inst_tp; inst_fn = len(gt_ids) - inst_tp
        p = safe_div(inst_tp, inst_tp + inst_fp); r = safe_div(inst_tp, inst_tp + inst_fn)
        metrics[f'F1@{th}'] = safe_div(2 * p * r, p + r)
        metrics[f'Inst_TP@{th}'] = inst_tp
        metrics[f'Inst_FP@{th}'] = inst_fp
        metrics[f'Inst_FN@{th}'] = inst_fn
    matches05 = one_to_one_matches(gt_ids, inst_pred_ids, inst_iou, 0.5)
    pq_tp = len(matches05); pq_fp = len(inst_pred_ids) - pq_tp; pq_fn = len(gt_ids) - pq_tp
    sq = safe_div(sum(x[2] for x in matches05), pq_tp)
    rq = safe_div(pq_tp, pq_tp + 0.5 * pq_fp + 0.5 * pq_fn)
    ari_mask_gt = gt_leaf | pred_leaf
    if np.any(ari_mask_gt) or np.any(extra_pred_leaf):
        ari_gt_labels = np.concatenate([gt_labels[ari_mask_gt], np.zeros(int(np.sum(extra_pred_leaf)), dtype=np.int64)])
        ari_pred_labels = np.concatenate([pred_on_gt[ari_mask_gt], extra_pred[extra_pred_leaf]])
        ari = adjusted_rand_score(ari_gt_labels, ari_pred_labels)
    else:
        ari = 0.0
    metrics.update({'SQ': sq, 'RQ': rq, 'PQ': sq * rq, 'ARI': ari})

    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader(); w.writerow(metrics)
    if args.full_output:
        write_matrix_csv(out / 'overlap_counts.csv', gt_ids, pred_ids, overlap)
        write_matrix_csv(out / 'iou_matrix.csv', gt_ids, pred_ids, iou)
        write_matrix_csv(out / 'gt_coverage_matrix.csv', gt_ids, pred_ids, cover_gt)
        write_matrix_csv(out / 'pred_purity_matrix.csv', gt_ids, pred_ids, cover_pred)
        plot_heatmap(out / 'overlap_counts.png', gt_ids, pred_ids, overlap, 'GT instance x Pred cluster overlap counts', fmt='d')

    plot_heatmap(out / 'iou_matrix.png', gt_ids, pred_ids, iou, 'IoU matrix')
    plot_heatmap(out / 'gt_coverage_matrix.png', gt_ids, pred_ids, cover_gt, 'GT coverage: overlap / |GT|')
    plot_heatmap(out / 'pred_purity_matrix.png', gt_ids, pred_ids, cover_pred, 'Pred purity: overlap / |Pred|')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dists, bins=80)
    ax.axvline(max_dist, color='red', linestyle='--', label=f'th={max_dist:.4g}')
    ax.set_title('Nearest-neighbor distance: GT point to prediction source')
    ax.set_xlabel('distance'); ax.set_ylabel('count'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'nn_distance_hist.png', dpi=180); plt.close(fig)

    if args.full_output:
        with open(out / 'many_to_many_iou_pairs.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['gt_id', 'pred_id', 'iou'])
            w.writerows(many_to_many_pairs)
        with open(out / 'matches_iou_05.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['gt_id', 'pred_id', 'iou'])
            w.writerows(matches05)
        with open(out / 'matches_iou_075.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['gt_id', 'pred_id', 'iou'])
            w.writerows(one_to_one_matches(gt_ids, inst_pred_ids, inst_iou, 0.75))

        write_xyz_labels(out / 'gt_instances.ply', gt_pts, gt_labels)
        write_xyz_labels(out / 'pred_clusters_on_gt_points.ply', gt_pts, pred_on_gt)
        write_xyz_labels(out / 'pred_leaf_binary_on_gt_points.ply', gt_pts, pred_leaf.astype(np.int64))
        error = np.zeros(len(gt_pts), dtype=np.int64)
        error[gt_leaf & pred_leaf] = 1
        error[~gt_leaf & pred_leaf] = 2
        error[gt_leaf & ~pred_leaf] = 3
        write_xyz_labels(out / 'semantic_error_tp_fp_fn.ply', gt_pts, error)

    print('Wrote audit to', out)
    print(metrics)


if __name__ == '__main__':
    main()
