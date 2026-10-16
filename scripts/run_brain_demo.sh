#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
episode_path="${1:-outputs/brain_demo}"
if [[ ! -f data/remind-001/anatomy.npz ]]; then
    .venv-ultrasound/bin/python scripts/prepare_brain.py
fi
.venv/bin/python scripts/run_isaac.py --headless --anatomy data/remind-001/anatomy.npz \
    --probe-motion sweep --seconds 2 --fps 8 --scatterers 40000 --output "$episode_path"
.venv-ultrasound/bin/python scripts/validate_episode.py "$episode_path"
.venv/bin/python scripts/capture_replay.py "$episode_path"
.venv-ultrasound/bin/python scripts/render_acoustic_sequence.py "$episode_path" --workers 4
.venv-ultrasound/bin/python scripts/make_video.py "$episode_path"
