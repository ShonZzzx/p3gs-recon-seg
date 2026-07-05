#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

ensure_single_gpu_mode
ensure_repo_ready "$GOF_REPO"
mkdir -p "$OUT_ROOT/gof/logs"

for plant in $PLANTS; do
  data="$(plant_source_dir "$plant")"
  out="$OUT_ROOT/gof/$plant"
  log_file="$OUT_ROOT/gof/logs/${plant}.log"
  ensure_dataset_ready "$data"
  mkdir -p "$out" "$(dirname "$log_file")"
  if [[ -f "$out/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
    log "B1 GOF skip existing $plant -> $out"
    continue
  fi
  log "Installing GOF CUDA extensions for $plant"
  "$GOF_PYTHON" -m pip install -q --force-reinstall "$GOF_REPO/submodules/diff-gaussian-rasterization"
  "$GOF_PYTHON" -m pip install -q --force-reinstall "$GOF_REPO/submodules/simple-knn"
  log "B1 GOF training $plant -> $out"
  (
    cd "$GOF_REPO"
    "$GOF_PYTHON" train.py \
      -s "$data" \
      -m "$out" \
      -r "$RESOLUTION" \
      --eval \
      --iterations "$ITERATIONS" \
      --test_iterations 7000 "$ITERATIONS" \
      --save_iterations 7000 "$ITERATIONS" \
      --checkpoint_iterations 15000 \
      --use_decoupled_appearance

    "$GOF_PYTHON" render.py -m "$out" --iteration "$ITERATIONS"
    "$GOF_PYTHON" metrics.py -m "$out" -r "$RESOLUTION"
    "$GOF_PYTHON" extract_mesh.py -m "$out" --iteration "$ITERATIONS" --filter_mesh
  ) 2>&1 | tee "$log_file"
done

log "B1 GOF queue finished"
