#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Separate environments keep Isaac's pinned NumPy and acoustic dependencies apart.
python3.12 scripts/bootstrap_env.py .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'isaacsim[all,extscache]==6.2.0.0' --extra-index-url https://pypi.nvidia.com
python3.12 scripts/bootstrap_env.py .venv-ultrasound
.venv-ultrasound/bin/python -m pip install --upgrade pip
.venv-ultrasound/bin/python -m pip install -r requirements-ultrasound.lock
.venv-ultrasound/bin/python -m pip install -r requirements-anatomy.lock
