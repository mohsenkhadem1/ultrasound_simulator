#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ ! -f data/remind-001/gaussians.npz ]]; then
  .venv-ultrasound/bin/python scripts/prepare_gaussians.py
fi
exec .venv/bin/python -u scripts/run_isaac.py \
  --headless --anatomy data/remind-001/anatomy.npz \
  --probe-motion press --scatterers 1000 \
  --live-gaussians data/remind-001/gaussians.npz \
  --output "outputs/live_$(date +%Y%m%d_%H%M%S)" "$@"
