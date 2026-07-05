#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

ensure_single_gpu_mode
ensure_repo_ready "$RADE_REPO"
mkdir -p "$OUT_ROOT/radegs/logs"

for plant in $PLANTS; do
  data="$(plant_source_dir "$plant")"
  out="$OUT_ROOT/radegs/$plant"
  log_file="$OUT_ROOT/radegs/logs/${plant}.log"
  ensure_dataset_ready "$data"
  mkdir -p "$out" "$(dirname "$log_file")"
  if [[ -f "$out/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
    log "B1 RaDe-GS skip existing $plant -> $out"
    continue
  fi
  log "Installing RaDe-GS CUDA extensions for $plant"
  "$RADE_PYTHON" -m pip install -q --force-reinstall "$RADE_REPO/submodules/diff-gaussian-rasterization"
  "$RADE_PYTHON" -m pip install -q --force-reinstall "$RADE_REPO/submodules/simple-knn"
  "$RADE_PYTHON" -m pip install -q --force-reinstall "$RADE_REPO/submodules/warp-patch-ncc"
  log "B1 RaDe-GS training $plant -> $out"
  (
    cd "$RADE_REPO"
    "$RADE_PYTHON" train.py \
      -s "$data" \
      -m "$out" \
      -r "$RESOLUTION" \
      --eval \
      --iterations "$ITERATIONS" \
      --test_iterations 7000 "$ITERATIONS" \
      --save_iterations 7000 "$ITERATIONS" \
      --checkpoint_iterations 15000 \
      --use_decoupled_appearance 3

    "$RADE_PYTHON" render.py -m "$out" --iteration "$ITERATIONS"
    if [[ -f metric.py ]]; then
      "$RADE_PYTHON" metric.py -m "$out"
    elif [[ -f metrics.py ]]; then
      "$RADE_PYTHON" metrics.py -m "$out"
    fi
    "$RADE_PYTHON" geometry_metric.py -m "$out" --iteration "$ITERATIONS" || true
    "$RADE_PYTHON" mesh_extract.py -m "$out" --iteration "$ITERATIONS" || true
  ) 2>&1 | tee "$log_file"
done

log "B1 RaDe-GS queue finished"
