#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=${PYTHON:-python3}

if [ ! -d .venv ]; then
	echo "Creating virtual environment in .venv ..."
	"$PY" -m venv .venv
fi

# shellcheck source=/dev/null
. .venv/bin/activate

pip install --upgrade pip >/dev/null

echo "Installing base requirements ..."
pip install numpy pygame-ce

if [ "${1:-}" = "--train" ]; then
	echo "Installing training requirements (torch) ..."
	pip install torch
fi

mkdir -p models

echo
echo "Setup complete. Run ./play.sh to start."
