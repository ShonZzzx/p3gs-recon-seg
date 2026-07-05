#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/roi_common.sh"

source /root/miniconda3/etc/profile.d/conda.sh

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is missing. Use a GPU image with CUDA toolkit before running this installer." >&2
  exit 1
fi

ensure_repo_ready "$RADE_REPO"
ensure_repo_ready "$GOF_REPO"

log "Installing RaDe-GS environment"
if [[ ! -x "$ROOT/envs/radegs/bin/python" ]]; then
  conda create -y -p "$ROOT/envs/radegs" python=3.12
fi
conda run -p "$ROOT/envs/radegs" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda run -p "$ROOT/envs/radegs" pip install -r "$RADE_REPO/requirements.txt"
conda run -p "$ROOT/envs/radegs" pip install "$RADE_REPO/submodules/diff-gaussian-rasterization" --no-build-isolation
conda run -p "$ROOT/envs/radegs" pip install "$RADE_REPO/submodules/warp-patch-ncc" --no-build-isolation
conda run -p "$ROOT/envs/radegs" pip install "$RADE_REPO/submodules/simple-knn" --no-build-isolation
conda run -p "$ROOT/envs/radegs" pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
conda install -y -p "$ROOT/envs/radegs" conda-forge::cgal
conda run -p "$ROOT/envs/radegs" pip install "$RADE_REPO/submodules/tetra_triangulation" --no-build-isolation

log "Installing GOF environment"
if [[ ! -x "$ROOT/envs/gof/bin/python" ]]; then
  conda create -y -p "$ROOT/envs/gof" python=3.8
fi
conda run -p "$ROOT/envs/gof" pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y -p "$ROOT/envs/gof" cudatoolkit-dev=11.3 cmake -c conda-forge
conda install -y -p "$ROOT/envs/gof" conda-forge::gmp conda-forge::cgal
conda run -p "$ROOT/envs/gof" pip install -r "$GOF_REPO/requirements.txt"
conda run -p "$ROOT/envs/gof" pip install "$GOF_REPO/submodules/diff-gaussian-rasterization"
conda run -p "$ROOT/envs/gof" pip install "$GOF_REPO/submodules/simple-knn"
conda run -p "$ROOT/envs/gof" pip install -e "$GOF_REPO/submodules/tetra-triangulation"

log "Environment install finished"
echo "Use:"
echo "  RADE_PYTHON=$ROOT/envs/radegs/bin/python GOF_PYTHON=$ROOT/envs/gof/bin/python bash $SCRIPT_DIR/run_roi_rade_gof_queue.sh"
