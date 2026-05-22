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
DEFAULT_DEPTH = 18
DEFAULT_EYE_DIST = 3

# Module-level cache for the trained policy network. `_POLICY` is one of:
#   * None       -- not yet attempted (lazy)
#   * False      -- attempted, unavailable (no torch / no checkpoint / load failed)
#   * model      -- a loaded torch.nn.Module in eval mode
_POLICY = None

# Cache of {(board_key, color, region_name): {(r,c): prob}} so each unique
# position runs the CNN once instead of once per visit during alpha-beta. The
# CNN forward dominates wall time without this; capped to keep memory bounded.
_POLICY_CACHE = {}
_POLICY_CACHE_MAX = 200_000

# Soft cap on alpha-beta transposition-table size. Deep searches were observed
# to grow the TT past 10 GB on hard problems; when we exceed this we drop the
# table and keep searching. Losing memoization is preferable to OOM.
_TT_MAX = 2_000_000


def _load_policy():
	"""Load and cache models/policy.pt. Returns the model or None if unavailable."""
	global _POLICY
	if _POLICY is False:
		return None
	if _POLICY is not None:
		return _POLICY
	try:
		import torch
	except ImportError:
		_POLICY = False
		return None
	from pathlib import Path
	ckpt = Path(__file__).resolve().parent.parent / "models" / "policy.pt"
	if not ckpt.exists():
		_POLICY = False
		return None
	try:
		from . import neural
		model = neural.build_model()
		state = torch.load(ckpt, map_location="cpu", weights_only=True)
		model.load_state_dict(state)
		model.eval()
	except Exception:
		_POLICY = False
		return None
	_POLICY = model
	return model


def policy_score_map(board, color, region):
	"""Return {(r, c): prob} over the 11x11 view, or {} if the policy is unavailable.

	The CNN was trained from the problem's player perspective ('own' vs 'opp').
	When evaluating moves for a different side-to-move (the defender), we flip
	the perspective by passing `color` as the player_color — that way 'own'
	always means the side about to play.

	Results are memoized per (position, color, region); the cache is dropped
	wholesale when it exceeds _POLICY_CACHE_MAX entries.
	"""
	model = _load_policy()
	if model is None:
		return {}
	cache_key = (board_key(board, region), color, region)
	cached = _POLICY_CACHE.get(cache_key)
	if cached is not None:
		return cached
	try:
		import torch
		from . import neural
	except ImportError:
		return {}
	with torch.no_grad():
		t = neural.board_to_tensor(board, region, color).unsqueeze(0)
		logits = model(t)[0]
		probs = torch.softmax(logits, dim=-1).tolist()
	out = {}
	for idx, p in enumerate(probs):
		coord = neural.index_to_move(idx, region)
		gr, gc = reg.parse_global(coord)
		out[(gr, gc)] = p
	if len(_POLICY_CACHE) >= _POLICY_CACHE_MAX:
		_POLICY_CACHE.clear()
	_POLICY_CACHE[cache_key] = out
	return out


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


def candidate_moves(board, color, target_globals, tcolor, region_name, allowed_globals=None):
	"""When `allowed_globals` is provided (frozenset of (r,c)), it REPLACES the
	eye-space heuristic — only those positions are considered (modulo legality).
	This applies to both sides, so the user must list every point that matters
	in any variation, or the solver may declare an unreachable win/loss."""
	if allowed_globals is not None:
		cells = allowed_globals
	else:
		cells = eye_space(board, target_globals, tcolor, region_name)
	moves = []
	for r, c in cells:
		b2 = board.copy()
		ok, _ = b2.play(r, c, color)
		if ok:
			moves.append((r, c))
	return moves


_ALLOWED_UNSET = object()


def _derive_allowed(problem):
	moves = getattr(problem, "allowed_moves", None) or []
	if not moves:
		return None
	return frozenset(reg.parse_global(m) for m in moves)


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


def order_moves(board, moves, color, target_globals, tcolor, region_name, pv_first=None, policy=None):
	"""Move ordering: PV move first (if given), policy-net prior next, then
	attacker prefers liberty-reducing & capturing moves; defender prefers
	liberty-extending. `policy` is an optional {(r,c): prob} prior; if None, we
	look one up from the trained checkpoint."""
	attacker = (color != tcolor)
	if policy is None:
		policy = policy_score_map(board, color, region_name)
	scored = []
	for r, c in moves:
		b2 = board.copy()
		_, captured = b2.play(r, c, color)
		libs = target_total_libs(b2, target_globals, tcolor)
		if attacker:
			score = -libs * 10 + captured * 5
		else:
			score = libs * 10 + captured * 5
		# Policy prior is a probability in [0, 1]; scale so it nudges ordering
		# among tactically-similar moves but cannot override a clear liberty
		# differential.
		prior = policy.get((r, c), 0.0)
		score += prior * 30
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


def alphabeta(board, color, problem, target_globals, depth, alpha, beta, tt=None, tcolor=None,
		allowed_globals=_ALLOWED_UNSET):
	"""Returns (score, best_move) with score from `problem.player`'s view (+1 = goal achieved).

	`allowed_globals` (frozenset of (r,c) | None): if a frozenset, both sides may
	only play there. None means no restriction. The default sentinel derives the
	value from `problem.allowed_moves` once at the top of the call chain, then
	passes through recursion so we don't re-parse on every node."""
	if tt is None:
		tt = {}
	if tcolor is None:
		tcolor = target_color(problem)
	if allowed_globals is _ALLOWED_UNSET:
		allowed_globals = _derive_allowed(problem)

	maximizing = (color == problem.player)

	# Soft cap: if the TT is enormous, drop it. We'd rather lose memoization
	# than swap to disk or get OOM-killed on a long solve.
	if len(tt) >= _TT_MAX:
		tt.clear()

	key = (board_key(board, problem.region), color, depth)
	if key in tt:
		return tt[key]

	val = evaluate(board, problem, target_globals, tcolor)
	if depth == 0 or val in (1, -1):
		tt[key] = (val, None)
		return val, None

	moves = candidate_moves(board, color, target_globals, tcolor, problem.region, allowed_globals)
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
	moves = order_moves(board, moves, color, target_globals, tcolor, problem.region, pv_first=pv_hint)
	best_move = None
	if maximizing:
		best = -INF
		for r, c in moves:
			b2 = board.copy()
			b2.play(r, c, color)
			score, _ = alphabeta(b2, bd.opponent(color), problem, target_globals,
				depth - 1, alpha, beta, tt, tcolor, allowed_globals)
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
				depth - 1, alpha, beta, tt, tcolor, allowed_globals)
			if score < best:
				best, best_move = score, (r, c)
			beta = min(beta, best)
			if beta <= alpha:
				break
		tt[key] = (best, best_move)
		return best, best_move


def best_move(board, color, problem, time_budget=2.0, tt=None, max_depth=20):
	"""Live-play wrapper: iterative-deepening alphabeta capped by wall-clock budget.

	Returns (score, move_xy) — score from problem.player's perspective:
	+1 = problem.player achieves goal, -1 = fails, 0 = unresolved at depth reached.

	Iterates depth 2, 4, 6, ... and stops when:
	  - the just-finished iteration produced a proven win/loss (score ±1),
	  - elapsed time exceeds `time_budget`, OR
	  - the next iteration is predicted to overshoot the remaining budget
	    (using last iteration's time × 4 as the growth estimate — alpha-beta
	    branching is roughly that bad in the worst case for this problem set).

	`tt` is a session-shared dict; reuse it across calls for cumulative speedup."""
	import time as _time
	if tt is None:
		tt = {}
	tcolor = target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	start = _time.monotonic()
	best_s, best_m = 0, None
	last_iter = 0.0
	for d in range(2, max_depth + 1, 2):
		elapsed = _time.monotonic() - start
		remaining = time_budget - elapsed
		if remaining <= 0:
			break
		if last_iter > 0 and last_iter * 4 > remaining:
			break
		t0 = _time.monotonic()
		s, m = alphabeta(board, color, problem, target_globals,
			d, -INF, INF, tt=tt, tcolor=tcolor)
		last_iter = _time.monotonic() - t0
		if m is not None:
			best_s, best_m = s, m
		if s in (1, -1):
			break
	return best_s, best_m


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
