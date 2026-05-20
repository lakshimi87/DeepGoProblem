#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
	echo "Run ./setup.sh first." >&2
	exit 1
fi

# shellcheck source=/dev/null
. .venv/bin/activate

exec python -m src.main play "$@"
