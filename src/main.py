"""CLI entry: play, train, solve."""

import sys


USAGE = "usage: python -m src.main {play|play-ui|train|solve} [args]"


def main():
	args = sys.argv[1:]
	if not args:
		print(USAGE)
		return 1
	cmd, rest = args[0], args[1:]
	if cmd == "play":
		from . import play
		play.main(rest)
		return 0
	if cmd in ("play-ui", "play_ui", "ui"):
		from . import ui_pygame
		ui_pygame.main(rest)
		return 0
	if cmd == "train":
		from . import train
		train.main(rest)
		return 0
	if cmd == "solve":
		from . import solver
		solver.main(rest)
		return 0
	print(f"unknown command: {cmd}")
	print(USAGE)
	return 1


if __name__ == "__main__":
	sys.exit(main())
