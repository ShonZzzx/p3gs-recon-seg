#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

mkdir -p "$ROI_ROOT"

for plant in $PLANTS; do
  src="$(plant_source_dir "$plant")"
  dst="$ROI_ROOT/$plant"
  log "Preparing ROI dataset for $plant"
  ensure_dataset_ready "$src"
  if [[ -d "$dst" && "${OVERWRITE:-0}" != "1" ]]; then
    log "Skip existing ROI dataset: $dst"
    continue
  fi
  "$PYTHON_PREP" "$SCRIPT_ROOT/prepare_roi_dataset.py" \
    --source "$src" \
    --output "$dst" \
    --margin "$MARGIN" \
    --percentile "$PERCENTILE" \
    --driver "$ROI_DRIVER" \
    --min-mask-ratio "$MIN_MASK_RATIO" \
    --max-mask-ratio "$MAX_MASK_RATIO" \
    --min-points "$MIN_POINTS" \
    --min-size "$MIN_SIZE" \
    --preview \
    ${OVERWRITE:+--overwrite}
done

log "ROI preprocessing finished: $ROI_ROOT"
