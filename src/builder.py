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


DEFAULT_SOLVE_DEPTH = 8
DEFAULT_MAX_PLIES = 12


def _terminal_verdict(score):
	if score == 1:
		return "Solver: player achieves goal."
	if score == -1:
		return "Solver: this line fails."
	return "Solver: unresolved within depth — extend the tree manually."


def build_tree(problem, solve_depth=DEFAULT_SOLVE_DEPTH, max_plies=DEFAULT_MAX_PLIES, verbose=False):
	"""Returns a fresh tree dict. Does NOT mutate problem.data."""
	board = problem.initial_board()
	player_color = problem.player
	opp_color = bd.opponent(player_color)
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	tt = {}
	return _build_node(
		board, player_color, opp_color,
		problem, target_globals, tcolor,
		solve_depth, max_plies, tt, verbose, ply=1,
	)


def _build_node(board, player_color, opp_color, problem, target_globals, tcolor,
		solve_depth, plies_left, tt, verbose, ply):
	if plies_left <= 0:
		return {}

	cur_eval = solver.evaluate(board, problem, target_globals, tcolor)
	if cur_eval != 0:
		return {}

	score, best_move_xy = solver.alphabeta(
		board, player_color, problem, target_globals,
		solve_depth, -solver.INF, solver.INF, tt={}, tcolor=tcolor,
	)
	if best_move_xy is None:
		return {}

	best_move = reg.format_global(*best_move_xy)
	correct = (score == 1)

	b1 = board.copy()
	ok, _ = b1.play(best_move_xy[0], best_move_xy[1], player_color)
	if not ok:
		return {}

	if verbose:
		print(f"  ply {ply}: player {bd.letter_from_color(player_color)} -> {best_move} (score={score})")

	# If the solver could not prove a win, record the suggestion but do not
	# fabricate a continuation — the rest of the line would be heuristic.
	if score != 1:
		return {best_move: {
			"correct": correct,
			"comment": _terminal_verdict(score),
			"reply": None,
			"branches": {},
		}}

	post_attack_eval = solver.evaluate(b1, problem, target_globals, tcolor)
	if post_attack_eval == 1:
		return {best_move: {
			"correct": True,
			"comment": "Captures target." if problem.goal == "capture" else "Group lives.",
			"reply": None,
			"branches": {},
		}}
	if post_attack_eval == -1:
		return {best_move: {
			"correct": False,
			"comment": _terminal_verdict(score),
			"reply": None,
			"branches": {},
		}}

	reply_score, reply_move_xy = solver.alphabeta(
		b1, opp_color, problem, target_globals,
		solve_depth, -solver.INF, solver.INF, tt={}, tcolor=tcolor,
	)
	reply_global = None
	b2 = b1.copy()
	if reply_move_xy is not None:
		ok, _ = b2.play(reply_move_xy[0], reply_move_xy[1], opp_color)
		if ok:
			reply_global = reg.format_global(*reply_move_xy)
		else:
			reply_global = None

	if verbose and reply_global:
		print(f"  ply {ply}: opp   {bd.letter_from_color(opp_color)} -> {reply_global} (score={reply_score})")

	post_reply_eval = solver.evaluate(b2, problem, target_globals, tcolor)
	if post_reply_eval != 0 or reply_global is None:
		return {best_move: {
			"correct": correct,
			"comment": _terminal_verdict(score),
			"reply": reply_global,
			"branches": {},
		}}

	child = _build_node(
		b2, player_color, opp_color, problem, target_globals, tcolor,
		solve_depth, plies_left - 1, tt, verbose, ply + 1,
	)
	return {best_move: {
		"correct": correct,
		"comment": _terminal_verdict(score),
		"reply": reply_global,
		"branches": child,
	}}


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
	tree = build_tree(problem, solve_depth=depth, max_plies=plies, verbose=verbose)
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
