#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
anatomy_dir="${DEFORM_ANATOMY_DIR:-data/remind-041}"
anatomy_patient="${DEFORM_PATIENT:-ReMIND-041}"
if [[ ! -f "$anatomy_dir/anatomy.npz" ]]; then
  .venv-ultrasound/bin/python scripts/prepare_brain.py --patient "$anatomy_patient" --output "$anatomy_dir"
fi
if [[ ! -f "$anatomy_dir/gaussians.npz" ]]; then
  .venv-ultrasound/bin/python scripts/prepare_gaussians.py --anatomy "$anatomy_dir/anatomy.npz" --output "$anatomy_dir/gaussians.npz" --depth-mm 80
fi
reconstruction_args=()
if [[ -n "${DEFORM_RECONSTRUCTION_DIR:-}" && -f "$DEFORM_RECONSTRUCTION_DIR/manifest.json" ]]; then
  reconstruction_args+=(--ultrasound-reconstruction "$DEFORM_RECONSTRUCTION_DIR")
fi
exec .venv/bin/python -u scripts/run_isaac.py \
  --headless --anatomy "$anatomy_dir/anatomy.npz" \
  --probe-motion press --scatterers 300000 --tumor-backscatter 3 \
  --live-gaussians "$anatomy_dir/gaussians.npz" \
  --live-ultrasound --live-ultrasound-inverse "${reconstruction_args[@]}" \
  --output "outputs/live_$(date +%Y%m%d_%H%M%S)" "$@"
