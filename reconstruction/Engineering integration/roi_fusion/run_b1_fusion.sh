#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

ensure_single_gpu_mode

B1_ROOT="${B1_ROOT:-$OUT_ROOT}"
FUSE_PY="${FUSE_PY:-python}"

latest_ply() {
  local base="$1"
  find "$base" -type f -name 'point_cloud.ply' | sort | tail -n 1
}

for plant in $PLANTS; do
  rade_ply=$(latest_ply "$B1_ROOT/radegs/$plant" || true)
  gof_ply=$(latest_ply "$B1_ROOT/gof/$plant" || true)
  if [[ -z "${rade_ply:-}" || -z "${gof_ply:-}" ]]; then
    echo "Missing RaDe or GOF point_cloud.ply for $plant" >&2
    exit 1
  fi
  fused_dir="$B1_ROOT/fused/$plant"
  mkdir -p "$fused_dir"
  "$FUSE_PY" "$SCRIPT_DIR/fuse_rade_gof_b1.py" \
    --rade-ply "$rade_ply" \
    --gof-ply "$gof_ply" \
    --out-ply "$fused_dir/fused_point_cloud.ply" \
    --out-json "$fused_dir/fusion_meta.json"
  "$ROOT/scripts/compare_30k/remote_pointcloud_views.py" --ply "$fused_dir/fused_point_cloud.ply" --out-dir "$fused_dir/pointcloud_views"
done

log "B1 fusion package finished"
