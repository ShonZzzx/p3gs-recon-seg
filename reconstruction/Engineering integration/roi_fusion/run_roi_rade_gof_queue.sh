#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

MODE="${MODE:-sequential}"

ensure_single_gpu_mode

case "$MODE" in
  sequential)
    FUSE_PY="${FUSE_PY:-python}"
    for plant in $PLANTS; do
      log "B1 closed-loop start: $plant"
      PLANTS="$plant" "$SCRIPT_DIR/run_roi_radegs.sh"
      PLANTS="$plant" "$SCRIPT_DIR/run_roi_gof.sh"
      PLANTS="$plant" "$SCRIPT_DIR/run_b1_fusion.sh"
      "$FUSE_PY" "$SCRIPT_DIR/run_b1_metrics.py" --root "${B1_ROOT:-$OUT_ROOT}" --plants "$plant"
      log "B1 closed-loop finished: $plant"
    done
    ;;
  *)
    echo "Unknown MODE=$MODE. Use sequential only on the current single-GPU server." >&2
    exit 2
    ;;
esac

log "RaDe-GS + GOF B1 queue finished"
