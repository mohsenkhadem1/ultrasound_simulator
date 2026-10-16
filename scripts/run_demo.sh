#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
episode="${1:-outputs/isaac_demo}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python scripts/run_isaac.py --headless --output "$episode" --seconds 2 --fps 2 --scatterers 20000
.venv-ultrasound/bin/python scripts/validate_episode.py "$episode"
.venv-ultrasound/bin/python -m deform_sim.ultrasound "$episode" --preview
printf 'Scene: %s/scene.usda\nUltrasound: %s/ultrasound/sequence.gif\n' "$episode" "$episode"
