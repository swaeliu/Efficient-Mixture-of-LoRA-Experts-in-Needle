#!/usr/bin/env bash
# Reproducible CPU-only environment setup for the molora project.
#
#   bash scripts/setup_env.sh [NEEDLE_PYTHON_DIR]
#
# Creates ./.venv, installs the project (editable) with dev extras, and links
# the local Needle implementation via a .pth file. NEEDLE_PYTHON_DIR is the
# directory that contains the needle/ package (e.g. ~/projects/dlsys/hw2/python);
# if omitted, ../hw2/python relative to this repository is tried.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/link_needle.py ${1:-}
python -c "import molora; print(molora.needle_info())"
echo "Environment ready. Activate with: source .venv/bin/activate"
