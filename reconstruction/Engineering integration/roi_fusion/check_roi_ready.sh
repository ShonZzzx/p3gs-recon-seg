#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

echo "ROOT=$ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "ROI_ROOT=$ROI_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "PLANTS=$PLANTS"
echo "RESOLUTION=$RESOLUTION"
echo "ITERATIONS=$ITERATIONS"
echo "ROI_DRIVER=$ROI_DRIVER"

echo
echo "GPU:"
(nvidia-smi || true) 2>&1 | head -20

echo
echo "CUDA compiler:"
(nvcc --version || true) 2>&1 | head -20

echo
echo "Repos:"
ensure_repo_ready "$RADE_REPO"
ensure_repo_ready "$GOF_REPO"
git -C "$RADE_REPO" rev-parse --short HEAD
git -C "$GOF_REPO" rev-parse --short HEAD

echo
echo "Python syntax:"
"$PYTHON_PREP" -m py_compile "$SCRIPT_ROOT/prepare_roi_dataset.py"
bash -n "$SCRIPT_DIR/roi_common.sh"
bash -n "$SCRIPT_DIR/run_prepare_roi.sh"
bash -n "$SCRIPT_DIR/run_roi_radegs.sh"
bash -n "$SCRIPT_DIR/run_roi_gof.sh"
bash -n "$SCRIPT_DIR/run_roi_rade_gof_queue.sh"

echo
echo "Dataset checks:"
for plant in $PLANTS; do
  src="$(plant_source_dir "$plant")"
  printf '%s -> %s\n' "$plant" "$src"
  ensure_dataset_ready "$src"
done

echo
echo "Entry command checks:"
grep -q -- "use_decoupled_appearance" "$RADE_REPO/arguments/__init__.py"
grep -q -- "use_decoupled_appearance" "$GOF_REPO/arguments/__init__.py"
grep -q -- "--iteration" "$RADE_REPO/render.py"
grep -q -- "--iteration" "$GOF_REPO/render.py"
grep -q -- "--iteration" "$GOF_REPO/extract_mesh.py"

echo
echo "OK: no-GPU readiness checks passed. CUDA extension installation and train smoke test still require a GPU/CUDA image."
