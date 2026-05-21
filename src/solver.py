"""Region-constrained alpha-beta tsumego solver with simple liberty/eye eval.

This is a starter solver. It is sound on easy/medium captures and is useful
for suggesting moves when authoring problems. It is not a competition-grade
engine.

Usage: python -m src.main solve <problem-id> [--depth N]
"""

from . import board as bd
from . import region as reg
from . import database


INF = 10 ** 9


def region_cells(region_name):
	(r_lo, c_lo), (r_hi, c_hi) = reg.region_bounds(region_name)
	for r in range(r_lo, r_hi + 1):
		for c in range(c_lo, c_hi + 1):
			yield r, c


def candidate_moves(board, color, region_name):
	moves = []
	for r, c in region_cells(region_name):
		if board.at(r, c) != bd.EMPTY:
			continue
		b2 = board.copy()
		ok, _ = b2.play(r, c, color)
		if ok:
			moves.append((r, c))
	return moves


def all_target_present(board, target_globals):
	return all(board.at(r, c) != bd.EMPTY for r, c in target_globals)


def count_eyes(board, color, region_name):
	"""Heuristic eye count: small empty regions fully surrounded by `color` inside the region."""
	seen = set()
	eyes = 0
	for r0, c0 in region_cells(region_name):
		if board.at(r0, c0) != bd.EMPTY or (r0, c0) in seen:
			continue
		stack = [(r0, c0)]
		cells = set()
		bordered_by = set()
		while stack:
			sr, sc = stack.pop()
			if (sr, sc) in cells:
				continue
			cells.add((sr, sc))
			seen.add((sr, sc))
			for nr, nc in board.neighbors(sr, sc):
				v = board.at(nr, nc)
				if v == bd.EMPTY:
					stack.append((nr, nc))
				else:
					bordered_by.add(v)
		if bordered_by == {color} and len(cells) <= 2:
			eyes += 1
	return eyes


def evaluate(board, problem, target_globals):
	"""+1 attacker wins, -1 defender wins, 0 unknown within depth."""
	goal = problem.goal
	if goal == "capture":
		if not all_target_present(board, target_globals):
			return 1
		if count_eyes(board, bd.opponent(problem.player), problem.region) >= 2:
			return -1
		return 0
	if goal == "live":
		if count_eyes(board, problem.player, problem.region) >= 2:
			return 1
		if not all_target_present(board, target_globals):
			return -1
		return 0
	return 0


def alphabeta(board, color, problem, target_globals, depth, alpha, beta, maximizing):
	val = evaluate(board, problem, target_globals)
	if depth == 0 or val in (1, -1):
		return val, None
	moves = candidate_moves(board, color, problem.region)
	if not moves:
		return val, None
	best_move = None
	if maximizing:
		best = -INF
		for r, c in moves:
			b2 = board.copy()
			b2.play(r, c, color)
			score, _ = alphabeta(b2, bd.opponent(color), problem, target_globals,
				depth - 1, alpha, beta, False)
			if score > best:
				best, best_move = score, (r, c)
			alpha = max(alpha, best)
			if beta <= alpha:
				break
		return best, best_move
	else:
		best = INF
		for r, c in moves:
			b2 = board.copy()
			b2.play(r, c, color)
			score, _ = alphabeta(b2, bd.opponent(color), problem, target_globals,
				depth - 1, alpha, beta, True)
			if score < best:
				best, best_move = score, (r, c)
			beta = min(beta, best)
			if beta <= alpha:
				break
		return best, best_move


def solve(problem, depth=6):
	board = problem.initial_board()
	target_globals = [reg.parse_global(t) for t in problem.target]
	score, move = alphabeta(board, problem.player, problem, target_globals,
		depth, -INF, INF, True)
	if move is None:
		return score, None
	(off_r, off_c), _ = reg.region_bounds(problem.region)
	return score, reg.format_local(move[0] - off_r, move[1] - off_c)


def main(args):
	if not args or args[0] in ("-h", "--help"):
		print("usage: python -m src.main solve <problem-id> [--depth N]")
		return
	pid = args[0]
	depth = 6
	if "--depth" in args:
		depth = int(args[args.index("--depth") + 1])
	problem = database.find_problem(pid)
	if problem is None:
		print(f"problem '{pid}' not found.")
		return
	score, move = solve(problem, depth=depth)
	verdict = {
		1: "attacker wins (goal achieved)",
		-1: "defender wins (goal denied)",
		0: "uncertain within depth",
	}[score]
	print(f"{pid}: best move = {move}   -> {verdict}")
