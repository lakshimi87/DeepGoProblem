# DeepGoProblem

A Go (baduk) tsumego (life-and-death problem) trainer.

Each tsumego happens in one of nine board regions — top-left, top-center,
top-right, middle-left, center, middle-right, bottom-left, bottom-center,
bottom-right — and is stored as a small json file containing the initial stone
arrangement, the side to play, an explanation, and a **response tree** that
records the program's reply to both the correct (success) and incorrect
(failure) moves. The play loop walks that tree.

## Architecture decision

This repository takes a **hybrid** approach. A deterministic **alpha-beta game
tree** is the source of truth for tactical reading, and a **convolutional policy
network** is used as a move-ordering / pruning prior to speed search up on
harder problems.

Why hybrid and not DL-only or game-tree-only:

- Tsumego is a constrained, local tactical search problem. On a 9x9 sub-region
  the search space is small enough that alpha-beta with a simple liberty/eye
  evaluation solves the bulk of easy and medium problems quickly and with
  provable correctness. Pure deep-learning evaluators cannot offer that
  guarantee.
- A pure-DL approach would also need a sizable supervised dataset to be
  reliable. Bootstrapping from a small problem database is exactly where
  alpha-beta dominates.
- A pure-game-tree approach explodes on harder problems (large regions, many
  candidate moves). A trained policy network supplies a strong move ordering
  that prunes the tree heavily, mirroring the policy head in AlphaGo / KataGo.

During play, neither the solver nor the network is consulted. `play.sh` simply
traverses the pre-stored response tree shipped with each problem. The solver
and the network are used **offline** by problem authors to extend those trees.

## Layout

	/
	|- setup.sh              install python deps into a venv
	|- play.sh               interactive problem player
	|- train.sh              train the policy network
	|- README.md
	|- requirements.txt
	|- src/
	|    |- main.py          cli entry point
	|    |- board.py         19x19 go board with rules (capture, suicide, ko)
	|    |- region.py        9-region geometry, local <-> global coords
	|    |- problem.py       problem dataclass, json load/save
	|    |- database.py      problem listing by difficulty
	|    |- play.py          interactive play loop
	|    |- solver.py        alpha-beta tsumego solver
	|    |- neural.py        policy cnn (pytorch, optional)
	|    |- train.py         supervised training over the problem set
	|- problems/
	|    |- easy/*.json
	|    |- medium/*.json
	|    |- hard/*.json
	|- models/               saved network weights (created by train.sh)

## Running

	./setup.sh              # one time, installs numpy into .venv
	./setup.sh --train      # also installs pytorch
	./play.sh               # choose a problem and play
	./train.sh              # train the policy network (requires --train above)

## Coordinates

All coordinates inside a problem file are written in *local* Go notation
`A1` to `J9` (no `I`), relative to the 9x9 view of the chosen region. Row 1
sits at the bottom of the view, row 9 at the top. Columns run left to right
`A B C D E F G H J`. The region offset is applied internally when the position
is placed on the 19x19 board, so the board edges that fall inside the view
become the actual board edges (top and left for the top-left region, and so
on).

## Problem format

	{
		"id": "easy-001",
		"difficulty": "easy",
		"region": "top-left",
		"player": "B",
		"goal": "capture",
		"target": ["A9", "A8"],
		"description": "Black to play and capture the two white stones.",
		"setup": {
			"B": ["B9", "B8"],
			"W": ["A9", "A8"]
		},
		"tree": {
			"A7": {
				"correct": true,
				"comment": "Correct. White is captured.",
				"reply": null,
				"branches": {}
			},
			"B7": {
				"correct": false,
				"comment": "Inefficient. White extends at A7.",
				"reply": "A7",
				"branches": {}
			}
		}
	}

`tree` keys are the player's candidate moves. Each node has a `correct` flag,
a `comment` shown to the user, an optional `reply` move played by the opponent,
and a `branches` dict keyed by the player's next move. Moves not present in the
tree are treated as out-of-book and end the problem with a "not in database"
message — author additional branches to handle them.

`goal` is `capture` or `live`. `target` lists the local coordinates of the
stones the goal applies to. The solver uses these when verifying a problem.

## Adding problems

Drop a new json file into the right `problems/<difficulty>/` folder. The id
should be unique. `play.sh` will pick it up automatically.

Once a policy network is trained, the solver

	python -m src.main solve <problem-id>

can suggest additional branches when extending a tree.

## Difficulty levels

	easy     immediate or one-step captures, simple atari / fill
	medium   2-4 move sequences, basic vital points
	hard     longer reading, eye-stealing, throw-ins, snapbacks
