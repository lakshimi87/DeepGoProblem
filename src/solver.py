"""Region-constrained alpha-beta tsumego solver.

The solver only considers moves inside the target group's *eye-space* — the
empty cells reachable from any target stone through empty cells and other
target stones, stopping at attacker stones and the board edge. That keeps the
branching factor bounded by the local shape rather than the whole 11x11 region.

A transposition table memoizes (board, side-to-move, depth) → score.

Usage: python -m src.main solve <problem-id> [--depth N]
"""

from collections import deque

from . import board as bd
from . import region as reg
from . import database


INF = 10 ** 9
DEFAULT_DEPTH = 12
DEFAULT_EYE_DIST = 3


def target_color(problem):
	player = problem.player
	return player if problem.goal == "live" else bd.opponent(player)


def all_target_captured(board, target_globals, tcolor):
	return all(board.at(r, c) != tcolor for r, c in target_globals)


def target_seed_stones(board, target_globals, tcolor):
	"""All tcolor stones connected to any surviving target stone (the target
	group extended through later-played defender stones)."""
	seeds = set()
	for r, c in target_globals:
		if (r, c) in seeds or board.at(r, c) != tcolor:
			continue
		stones, _ = board.group(r, c)
		seeds |= stones
	return seeds


def eye_space(board, target_globals, tcolor, region_name, max_empty_dist=DEFAULT_EYE_DIST):
	"""Empty cells reachable from the target group via empty cells (each empty
	step counts) or through tcolor stones (free), bounded by the region and
	the attacker wall. `max_empty_dist` caps how many empty steps we'll take —
	this keeps the candidate set tractable when the wall has holes."""
	(r_lo, c_lo), (r_hi, c_hi) = reg.region_bounds(region_name)
	seeds = target_seed_stones(board, target_globals, tcolor)
	dist = {s: 0 for s in seeds}
	dq = deque(seeds)
	empties = set()
	while dq:
		r, c = dq.popleft()
		d = dist[(r, c)]
		if not (r_lo <= r <= r_hi and c_lo <= c <= c_hi):
			continue
		v = board.at(r, c)
		if v == bd.EMPTY:
			empties.add((r, c))
		if v == bd.EMPTY or v == tcolor:
			for nr, nc in board.neighbors(r, c):
				v2 = board.at(nr, nc)
				if v2 == bd.EMPTY:
					nd = d + 1
				elif v2 == tcolor:
					nd = d
				else:
					continue
				if nd > max_empty_dist:
					continue
				if (nr, nc) in dist and dist[(nr, nc)] <= nd:
					continue
				dist[(nr, nc)] = nd
				if v2 == tcolor:
					dq.appendleft((nr, nc))
				else:
					dq.append((nr, nc))
	return empties


def candidate_moves(board, color, target_globals, tcolor, region_name):
	cells = eye_space(board, target_globals, tcolor, region_name)
	moves = []
	for r, c in cells:
		b2 = board.copy()
		ok, _ = b2.play(r, c, color)
		if ok:
			moves.append((r, c))
	return moves


def group_eye_count(board, stones, color):
	"""Number of definite single-point eyes belonging to this group."""
	if not stones:
		return 0
	libs = set()
	for r, c in stones:
		for nr, nc in board.neighbors(r, c):
			if board.at(nr, nc) == bd.EMPTY:
				libs.add((nr, nc))
	eye_points = set()
	for r, c in libs:
		if all(board.at(nr, nc) == color for nr, nc in board.neighbors(r, c)):
			eye_points.add((r, c))
	# Connected components among eye points
	eyes = 0
	seen = set()
	for p in eye_points:
		if p in seen:
			continue
		comp_size = 0
		stk = [p]
		while stk:
			q = stk.pop()
			if q in seen or q not in eye_points:
				continue
			seen.add(q)
			comp_size += 1
			r, c = q
			for nr, nc in board.neighbors(r, c):
				if (nr, nc) in eye_points and (nr, nc) not in seen:
					stk.append((nr, nc))
		if comp_size == 1:
			eyes += 1
	return eyes


def best_target_eyes(board, target_globals, tcolor):
	"""Max eye count among the connected target sub-groups still on the board."""
	seen = set()
	best = 0
	for r, c in target_globals:
		if (r, c) in seen or board.at(r, c) != tcolor:
			continue
		stones, _ = board.group(r, c)
		seen |= stones
		eyes = group_eye_count(board, stones, tcolor)
		if eyes > best:
			best = eyes
	return best


def evaluate(board, problem, target_globals, tcolor):
	"""+1 player achieved goal, -1 player failed, 0 unresolved."""
	captured = all_target_captured(board, target_globals, tcolor)
	eyes = best_target_eyes(board, target_globals, tcolor)
	if problem.goal == "capture":
		if captured:
			return 1
		if eyes >= 2:
			return -1
		return 0
	# live: player == tcolor, wants target to live
	if eyes >= 2:
		return 1
	if captured:
		return -1
	return 0


def target_total_libs(board, target_globals, tcolor):
	"""Combined liberty count across all surviving target groups."""
	seen = set()
	libs = set()
	for r, c in target_globals:
		if (r, c) in seen or board.at(r, c) != tcolor:
			continue
		stones, gl = board.group(r, c)
		seen |= stones
		libs |= gl
	return len(libs)


def order_moves(board, moves, color, target_globals, tcolor, pv_first=None):
	"""Move ordering: PV move first (if given), then attacker prefers
	liberty-reducing & capturing moves; defender prefers liberty-extending."""
	attacker = (color != tcolor)
	scored = []
	for r, c in moves:
		b2 = board.copy()
		_, captured = b2.play(r, c, color)
		libs = target_total_libs(b2, target_globals, tcolor)
		if attacker:
			score = -libs * 10 + captured * 5
		else:
			score = libs * 10 + captured * 5
		if pv_first is not None and (r, c) == pv_first:
			score += 10 ** 6
		scored.append((score, (r, c)))
	scored.sort(reverse=True, key=lambda x: x[0])
	return [m for _, m in scored]


def board_key(board, region_name):
	(r_lo, c_lo), (r_hi, c_hi) = reg.region_bounds(region_name)
	return tuple(
		tuple(board.grid[r][c_lo:c_hi + 1])
		for r in range(r_lo, r_hi + 1)
	)


def alphabeta(board, color, problem, target_globals, depth, alpha, beta, tt=None, tcolor=None):
	"""Returns (score, best_move) with score from `problem.player`'s view (+1 = goal achieved)."""
	if tt is None:
		tt = {}
	if tcolor is None:
		tcolor = target_color(problem)

	maximizing = (color == problem.player)

	key = (board_key(board, problem.region), color, depth)
	if key in tt:
		return tt[key]

	val = evaluate(board, problem, target_globals, tcolor)
	if depth == 0 or val in (1, -1):
		tt[key] = (val, None)
		return val, None

	moves = candidate_moves(board, color, target_globals, tcolor, problem.region)
	if not moves:
		tt[key] = (val, None)
		return val, None

	# Use any previously-cached best move at this position as PV hint
	pv_hint = None
	for cached_depth in range(depth - 1, 0, -1):
		pv_entry = tt.get((key[0], color, cached_depth))
		if pv_entry and pv_entry[1] is not None:
			pv_hint = pv_entry[1]
			break
	moves = order_moves(board, moves, color, target_globals, tcolor, pv_first=pv_hint)
	best_move = None
	if maximizing:
		best = -INF
		for r, c in moves:
			b2 = board.copy()
			b2.play(r, c, color)
			score, _ = alphabeta(b2, bd.opponent(color), problem, target_globals,
				depth - 1, alpha, beta, tt, tcolor)
			if score > best:
				best, best_move = score, (r, c)
			alpha = max(alpha, best)
			if beta <= alpha:
				break
		tt[key] = (best, best_move)
		return best, best_move
	else:
		best = INF
		for r, c in moves:
			b2 = board.copy()
			b2.play(r, c, color)
			score, _ = alphabeta(b2, bd.opponent(color), problem, target_globals,
				depth - 1, alpha, beta, tt, tcolor)
			if score < best:
				best, best_move = score, (r, c)
			beta = min(beta, best)
			if beta <= alpha:
				break
		tt[key] = (best, best_move)
		return best, best_move


def solve(problem, depth=DEFAULT_DEPTH, iterative=True):
	board = problem.initial_board()
	target_globals = [reg.parse_global(t) for t in problem.target]
	tcolor = target_color(problem)
	tt = {}
	score, move = -INF, None
	if iterative:
		for d in range(2, depth + 1, 2):
			score, move = alphabeta(board, problem.player, problem, target_globals,
				d, -INF, INF, tt=tt, tcolor=tcolor)
			if score in (1, -1):
				break
	else:
		score, move = alphabeta(board, problem.player, problem, target_globals,
			depth, -INF, INF, tt=tt, tcolor=tcolor)
	if move is None:
		return score, None
	return score, reg.format_global(*move)


def main(args):
	if not args or args[0] in ("-h", "--help"):
		print("usage: python -m src.main solve <problem-id> [--depth N]")
		return
	pid = args[0]
	depth = DEFAULT_DEPTH
	if "--depth" in args:
		depth = int(args[args.index("--depth") + 1])
	problem = database.find_problem(pid)
	if problem is None:
		print(f"problem '{pid}' not found.")
		return
	score, move = solve(problem, depth=depth)
	verdict = {
		1: "player achieves goal",
		-1: "player fails (opponent wins)",
		0: "uncertain within depth",
	}[score]
	print(f"{pid}: best move = {move}   -> {verdict}")
