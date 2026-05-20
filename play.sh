#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
	echo "Run ./setup.sh first." >&2
	exit 1
fi

# shellcheck source=/dev/null
. .venv/bin/activate

# Default to the pygame-ce graphical UI. Pass --cli to use the text-based loop.
mode="ui"
passthrough=()
for arg in "$@"; do
	case "$arg" in
		--cli|--text) mode="cli" ;;
		--ui|--gui) mode="ui" ;;
		*) passthrough+=("$arg") ;;
	esac
done

if [ "$mode" = "cli" ]; then
	exec python -m src.main play "${passthrough[@]}"
fi

if ! python -c "import pygame" 2>/dev/null; then
	echo "pygame-ce is not installed. Run ./setup.sh to install it, or pass --cli." >&2
	exit 1
fi

exec python -m src.main play-ui "${passthrough[@]}"
