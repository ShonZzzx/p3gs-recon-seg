#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ml_course}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataGS}"
ROI_ROOT="${ROI_ROOT:-$ROOT/dataGS_roi}"
OUT_ROOT="${OUT_ROOT:-$ROOT/compare_b1_r2_30k}"
SCRIPT_ROOT="${SCRIPT_ROOT:-$ROOT/scripts/roi_fusion}"
RADE_REPO="${RADE_REPO:-$ROOT/repos/RaDe-GS}"
GOF_REPO="${GOF_REPO:-$ROOT/repos/gaussian-opacity-fields}"

PLANTS="${PLANTS:-plant_002 plant_013 plant_016 plant_019}"
RESOLUTION="${RESOLUTION:-2}"
ITERATIONS="${ITERATIONS:-30000}"
MARGIN="${MARGIN:-0.20}"
PERCENTILE="${PERCENTILE:-1.0}"
ROI_DRIVER="${ROI_DRIVER:-hybrid}"
MIN_MASK_RATIO="${MIN_MASK_RATIO:-0.001}"
MAX_MASK_RATIO="${MAX_MASK_RATIO:-0.60}"
MIN_POINTS="${MIN_POINTS:-30}"
MIN_SIZE="${MIN_SIZE:-256}"

PYTHON_PREP="${PYTHON_PREP:-/root/miniconda3/bin/python}"
RADE_PYTHON="${RADE_PYTHON:-python}"
GOF_PYTHON="${GOF_PYTHON:-python}"

GPU_MODE="${GPU_MODE:-single}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

plant_source_dir() {
  local plant="$1"
  if [[ "$plant" == "plant_002" && -d "$DATA_ROOT/plant_002_undistorted" ]]; then
    printf '%s\n' "$DATA_ROOT/plant_002_undistorted"
  else
    printf '%s\n' "$DATA_ROOT/$plant"
  fi
}

ensure_dataset_ready() {
  local dataset="$1"
  test -d "$dataset/images"
  if [[ -f "$dataset/sparse/0/cameras.bin" && -f "$dataset/sparse/0/images.bin" && -f "$dataset/sparse/0/points3D.bin" ]]; then
    return 0
  fi
  if [[ -f "$dataset/sparse/cameras.bin" && -f "$dataset/sparse/images.bin" && -f "$dataset/sparse/points3D.bin" ]]; then
    return 0
  fi
  echo "Missing COLMAP sparse binary model in $dataset" >&2
  return 1
}

ensure_repo_ready() {
  local repo="$1"
  test -f "$repo/train.py"
  test -f "$repo/render.py"
}

ensure_single_gpu_mode() {
  if [[ "$GPU_MODE" != "single" ]]; then
    echo "This pipeline is configured for single-GPU execution only." >&2
    exit 2
  fi
}

