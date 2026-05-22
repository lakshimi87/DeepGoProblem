"""One-off: extend hukseon_drop_large.json with a branch for Black F2.

Computes the solver's verdict for F2 from the initial position, records
White's best reply, and recursively populates the tree from there using the
same logic as builder._build_node (single principal-variation extension).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import board as bd
from src import region as reg
from src import database
from src import solver
from src import builder


def main():
	pid = "hukseon_drop_large"
	first_move = "F2"
	problem = database.find_problem(pid)
	if problem is None:
		print(f"problem '{pid}' not found.")
		return 1

	board = problem.initial_board()
	player_color = problem.player  # Black
	opp_color = bd.opponent(player_color)
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]

	gr, gc = reg.parse_global(first_move)
	b_after = board.copy()
	ok, _ = b_after.play(gr, gc, player_color)
	if not ok:
		print(f"illegal first move: {first_move}")
		return 1

	# Evaluate from White's perspective (defender to move). Returned score is
	# from problem.player (Black)'s view: +1 = Black still wins, -1 = Black loses.
	print(f"solving after Black {first_move} (White to move)...", flush=True)
	score, white_reply_xy = builder._iterative_solve(
		b_after, opp_color, problem, target_globals, tcolor,
		builder.DEFAULT_SOLVE_DEPTH, verbose=True, label="post-F2",
	)

	correct = (score == 1)
	comment = builder._terminal_verdict(score)
	node = {
		"correct": correct,
		"comment": comment,
		"reply": None,
		"branches": {},
	}

	if white_reply_xy is not None:
		b_after_reply = b_after.copy()
		ok, _ = b_after_reply.play(white_reply_xy[0], white_reply_xy[1], opp_color)
		if ok:
			node["reply"] = reg.format_global(*white_reply_xy)
			# Only continue building if Black still has a winning continuation.
			post_eval = solver.evaluate(b_after_reply, problem, target_globals, tcolor)
			if score == 1 and post_eval == 0:
				builder._build_node(
					node["branches"], b_after_reply, player_color, opp_color,
					problem, target_globals, tcolor,
					builder.DEFAULT_SOLVE_DEPTH, builder.DEFAULT_MAX_PLIES - 1,
					verbose=True, ply=2,
					save_cb=lambda r: _save(problem, first_move, node),
					root=None,
				)
			elif post_eval == 1:
				node["comment"] = "Captures target." if problem.goal == "capture" else "Group lives."

	_save(problem, first_move, node)
	print(f"saved branch {first_move} -> {node['comment']} (correct={correct}); reply={node['reply']}")
	return 0


def _save(problem, first_move, node):
	tree = problem.data.get("tree") or {}
	tree[first_move] = node
	problem.data["tree"] = tree
	problem.save()


if __name__ == "__main__":
	sys.exit(main())
