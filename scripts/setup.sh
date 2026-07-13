#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    echo "[setup] .venv already exists — recreating"
    rm -rf .venv
fi

echo "[setup] creating Python virtual environment"
python3 -m venv .venv

echo "[setup] activating and installing dependencies"
source .venv/bin/activate
pip install -e ".[dev]"

mkdir -p results

echo ""
echo "===== Setup complete ====="
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Run tests with:"
echo "  pytest -v"
echo ""
echo "Run experiments with (e.g.):"
echo "  python -m experiments.scenario_1_nominal"
