#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
	echo "Run ./setup.sh --train first." >&2
	exit 1
fi

# shellcheck source=/dev/null
. .venv/bin/activate

if ! python -c "import torch" 2>/dev/null; then
	echo "torch is not installed. Run ./setup.sh --train to install it." >&2
	exit 1
fi

exec python -m src.main train "$@"
