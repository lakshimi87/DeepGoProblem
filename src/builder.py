"""Auto-populate a problem's response tree by repeatedly calling the solver.

For each attacker-to-move position, the solver's best move is recorded with a
solver-derived verdict (correct/incorrect). The defender's best reply is then
chosen, and the process recurses until the position is resolved or the per-line
depth budget runs out.

The generated tree is a single principal variation. Off-book deviations during
play remain off-book — the user can extend the tree manually in the editor.
"""

from . import board as bd
from . import region as reg
from . import database
from . import solver


DEFAULT_SOLVE_DEPTH = 14
DEFAULT_MAX_PLIES = 20


def _iterative_solve(board, color, problem, target_globals, tcolor, max_depth, verbose, label):
	"""Iterative deepening from depth 2 up to max_depth, returning the deepest
	(score, move) reached. Sharing one tt across iterations lets the PV from
	the shallower search prime move ordering for the deeper one."""
	tt = {}
	best_score, best_move = 0, None
	for d in range(2, max_depth + 1, 2):
		score, move = solver.alphabeta(
			board, color, problem, target_globals,
			d, -solver.INF, solver.INF, tt=tt, tcolor=tcolor,
		)
		if move is not None:
			best_score, best_move = score, move
		if verbose:
			mv_str = reg.format_global(*move) if move else "(none)"
			print(f"    {label} depth={d}: {mv_str} score={score}", flush=True)
		if score in (1, -1):
			break
	return best_score, best_move


def _terminal_verdict(score):
	if score == 1:
		return "Solver: player achieves goal."
	if score == -1:
		return "Solver: this line fails."
	return "Solver: unresolved within depth — extend the tree manually."


def build_tree(problem, solve_depth=DEFAULT_SOLVE_DEPTH, max_plies=DEFAULT_MAX_PLIES,
		verbose=False, save_cb=None):
	"""Mutates and returns a tree dict built top-down. `save_cb(root)` is called
	after each meaningful update so a long run can be killed without losing
	partial progress. Does NOT mutate problem.data directly."""
	board = problem.initial_board()
	player_color = problem.player
	opp_color = bd.opponent(player_color)
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	root = {}
	noop = lambda r: None
	_build_node(
		root, board, player_color, opp_color,
		problem, target_globals, tcolor,
		solve_depth, max_plies, verbose, ply=1,
		save_cb=save_cb or noop, root=root,
	)
	return root


def _build_node(branches, board, player_color, opp_color, problem, target_globals,
		tcolor, solve_depth, plies_left, verbose, ply, save_cb, root):
	if plies_left <= 0:
		return

	cur_eval = solver.evaluate(board, problem, target_globals, tcolor)
	if cur_eval != 0:
		return

	if verbose:
		print(f"  ply {ply}: solving attacker ({bd.letter_from_color(player_color)})", flush=True)
	score, best_move_xy = _iterative_solve(
		board, player_color, problem, target_globals, tcolor,
		solve_depth, verbose, label=f"ply{ply}-att",
	)
	if best_move_xy is None:
		return

	best_move = reg.format_global(*best_move_xy)
	correct = (score == 1)

	b1 = board.copy()
	ok, _ = b1.play(best_move_xy[0], best_move_xy[1], player_color)
	if not ok:
		return

	if verbose:
		print(f"  ply {ply}: player {bd.letter_from_color(player_color)} -> {best_move} (score={score})", flush=True)

	node = {
		"correct": correct,
		"comment": _terminal_verdict(score),
		"reply": None,
		"branches": {},
	}
	branches[best_move] = node
	save_cb(root)

	# If the solver could not prove a win, record the suggestion but do not
	# fabricate a continuation — the rest of the line would be heuristic.
	if score != 1:
		return

	post_attack_eval = solver.evaluate(b1, problem, target_globals, tcolor)
	if post_attack_eval == 1:
		node["comment"] = "Captures target." if problem.goal == "capture" else "Group lives."
		save_cb(root)
		return
	if post_attack_eval == -1:
		return

	if verbose:
		print(f"  ply {ply}: solving defender ({bd.letter_from_color(opp_color)})", flush=True)
	reply_score, reply_move_xy = _iterative_solve(
		b1, opp_color, problem, target_globals, tcolor,
		solve_depth, verbose, label=f"ply{ply}-def",
	)
	reply_global = None
	b2 = b1.copy()
	if reply_move_xy is not None:
		ok, _ = b2.play(reply_move_xy[0], reply_move_xy[1], opp_color)
		if ok:
			reply_global = reg.format_global(*reply_move_xy)

	if verbose and reply_global:
		print(f"  ply {ply}: opp   {bd.letter_from_color(opp_color)} -> {reply_global} (score={reply_score})", flush=True)

	node["reply"] = reply_global
	save_cb(root)

	post_reply_eval = solver.evaluate(b2, problem, target_globals, tcolor)
	if post_reply_eval != 0 or reply_global is None:
		return

	_build_node(
		node["branches"], b2, player_color, opp_color, problem, target_globals, tcolor,
		solve_depth, plies_left - 1, verbose, ply + 1, save_cb, root,
	)


def main(args):
	if not args or args[0] in ("-h", "--help"):
		print("usage: python -m src.main build-tree <problem-id> [--depth N] [--plies N] [--force] [-v]")
		return
	pid = args[0]
	rest = args[1:]
	depth = DEFAULT_SOLVE_DEPTH
	plies = DEFAULT_MAX_PLIES
	force = False
	verbose = False
	i = 0
	while i < len(rest):
		a = rest[i]
		if a == "--depth":
			depth = int(rest[i + 1]); i += 2; continue
		if a == "--plies":
			plies = int(rest[i + 1]); i += 2; continue
		if a == "--force":
			force = True; i += 1; continue
		if a in ("-v", "--verbose"):
			verbose = True; i += 1; continue
		print(f"unknown argument: {a}")
		return
	problem = database.find_problem(pid)
	if problem is None:
		print(f"problem '{pid}' not found.")
		return
	if problem.tree and not force:
		print(f"{pid}: tree already populated ({len(problem.tree)} entries). use --force to overwrite.")
		return

	print(f"building tree for {pid} (solve_depth={depth}, max_plies={plies})...")

	def save_cb(root):
		problem.data["tree"] = root
		problem.save()

	tree = build_tree(problem, solve_depth=depth, max_plies=plies, verbose=verbose, save_cb=save_cb)
	problem.data["tree"] = tree
	problem.save()
	n = _count_nodes(tree)
	print(f"saved {n} nodes to {problem.source_path}.")


def _count_nodes(tree):
	if not tree:
		return 0
	n = 0
	for entry in tree.values():
		n += 1
		n += _count_nodes(entry.get("branches") or {})
	return n
