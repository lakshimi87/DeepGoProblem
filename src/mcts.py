"""Monte Carlo Tree Search with PUCT selection.

Public entry points:

    mcts.best_move(board, color, problem, time_budget=2.0, tt=None, ...)
        -> (score_from_player_perspective, move_xy)
        Live-play wrapper. Uses the neural evaluator if a trained checkpoint is
        available, otherwise falls back to a tactical evaluator (rank-based
        priors + shallow alpha-beta leaf eval) that stays sound without a model.

    mcts.run_simulations(board, color, problem, evaluator, n_sims, ...)
        -> (move_xy, pi_vector, root_value)
        AlphaZero-style self-play call. Always runs a fixed number of
        simulations, applies Dirichlet noise at root, samples the move with the
        given temperature, and returns the visit-count target `pi` (a length
        VIEW*VIEW vector) for policy training plus the root's value estimate.

Every node stores values from its own side-to-move perspective; we negate on
backup so the alternation is implicit. Proven ±1 values (from terminal eval or
leaf alpha-beta resolution) propagate up the tree exactly: if any child is a
proven loss for the child's side-to-move, the parent has a proven win; if
every child is a proven win, the parent is a proven loss.
"""

import math
import random
import time

from . import board as bd
from . import region as reg
from . import solver


C_PUCT = 1.4
LEAF_DEPTH = 4   # alpha-beta plies used to value an unexpanded leaf (tactical evaluator)
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


class Node:
	__slots__ = ("board", "color", "moves", "P", "N", "W", "children",
		"expanded", "terminal", "proven", "leaf_value")

	def __init__(self, board, color):
		self.board = board
		self.color = color          # side to move at this node
		self.moves = []             # legal candidate moves
		self.P = {}                 # move -> prior probability
		self.N = {}                 # move -> visit count
		self.W = {}                 # move -> cumulative value (from this color's view)
		self.children = {}          # move -> child Node (lazy)
		self.expanded = False
		self.terminal = False
		self.proven = None          # ±1 from this color's view, or None
		self.leaf_value = None      # cached evaluator value when first expanded


def _flip(v, node_color, player):
	"""Convert a value between problem.player view and node_color view.
	Involution — same formula either direction."""
	return v if node_color == player else -v


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
# An evaluator is a callable
#     evaluator(board, color, moves, problem) -> (priors_dict, value)
# where `priors_dict` is {(r,c): float} summing to ~1 over `moves`, and `value`
# is from `color`'s view in [-1, +1]. It is only called for non-terminal nodes;
# terminal nodes set `proven` directly from `solver.evaluate`.


def tactical_evaluator(leaf_depth=LEAF_DEPTH, ab_tt=None):
	"""Returns an evaluator that uses rank-based priors and a shallow alpha-beta
	at the leaf — the legacy behavior, sound without any trained network."""
	def _ev(board, color, moves, problem, target_globals, tcolor, allowed_globals):
		ordered = solver.order_moves(
			board, moves, color, target_globals, tcolor, problem.region,
		)
		weights = [1.0 / (i + 1) for i in range(len(ordered))]
		total = sum(weights)
		priors = {m: weights[i] / total for i, m in enumerate(ordered)}
		score_p, _ = solver.alphabeta(
			board, color, problem, target_globals,
			leaf_depth, -solver.INF, solver.INF,
			tt=ab_tt if ab_tt is not None else {}, tcolor=tcolor, allowed_globals=allowed_globals,
		)
		value = float(_flip(score_p, color, problem.player))
		# Pass through whether alpha-beta resolved this exactly.
		proven = int(value) if score_p in (1, -1) else None
		return priors, value, proven
	return _ev


def neural_evaluator(model):
	"""Returns an evaluator that calls the policy+value network. Priors are the
	softmaxed policy logits masked to legal moves; the value head provides v."""
	import torch
	from . import neural

	model.eval()

	def _ev(board, color, moves, problem, target_globals, tcolor, allowed_globals):
		with torch.no_grad():
			t = neural.board_to_tensor(board, problem.region, color).unsqueeze(0)
			policy_logits, value = model(t)
		# Mask logits to legal moves and renormalize.
		idx_for_move = {}
		for m in moves:
			idx = neural.global_to_index(m[0], m[1], problem.region)
			if idx is not None:
				idx_for_move[m] = idx
		if not idx_for_move:
			# No mapped move — uniform fallback.
			n = max(1, len(moves))
			priors = {m: 1.0 / n for m in moves}
		else:
			legal_logits = torch.tensor([policy_logits[0, idx_for_move[m]].item() for m in moves])
			probs = torch.softmax(legal_logits, dim=-1).tolist()
			priors = {m: probs[i] for i, m in enumerate(moves)}
		v = float(value.item())
		return priors, v, None  # NN never proves a position by itself
	return _ev


# ---------------------------------------------------------------------------
# Tree ops
# ---------------------------------------------------------------------------


def _expand_node(node, problem, target_globals, tcolor, allowed_globals, evaluator):
	"""Mark terminal, or compute candidates + priors + leaf value.

	Sets node.expanded. Returns the value from node.color's view, suitable for
	immediate backup; for terminal nodes this is the proven ±1, for non-terminal
	leaves it's the evaluator's value estimate.
	"""
	val_p = solver.evaluate(node.board, problem, target_globals, tcolor)
	if val_p in (1, -1):
		node.terminal = True
		node.proven = _flip(val_p, node.color, problem.player)
		node.expanded = True
		node.leaf_value = float(node.proven)
		return node.leaf_value
	moves = solver.candidate_moves(
		node.board, node.color, target_globals, tcolor, problem.region, allowed_globals,
	)
	if not moves:
		# No legal candidates: side-to-move can't continue. Treat as loss for it.
		node.terminal = True
		node.proven = -1
		node.expanded = True
		node.leaf_value = -1.0
		return node.leaf_value
	node.moves = moves
	priors, value, proven = evaluator(
		node.board, node.color, moves, problem, target_globals, tcolor, allowed_globals,
	)
	for m in moves:
		node.P[m] = priors.get(m, 0.0)
		node.N[m] = 0
		node.W[m] = 0.0
	if proven is not None:
		node.proven = proven
	node.expanded = True
	node.leaf_value = value
	return value


def _add_dirichlet_noise(node, alpha=DIRICHLET_ALPHA, epsilon=DIRICHLET_EPSILON, rng=None):
	"""Mix Dirichlet(alpha) noise into root priors so self-play explores beyond
	what the current policy already favours. AlphaZero default: alpha=0.3 in Go,
	epsilon=0.25 of mixing weight."""
	if not node.moves:
		return
	rng = rng or random
	# Sample Dirichlet via independent Gammas (avoids needing numpy here).
	noise = [rng.gammavariate(alpha, 1.0) for _ in node.moves]
	s = sum(noise) or 1.0
	noise = [x / s for x in noise]
	for i, m in enumerate(node.moves):
		node.P[m] = (1 - epsilon) * node.P[m] + epsilon * noise[i]


def _select_move(node):
	"""PUCT: argmax over moves of Q + c * P * sqrt(parent_N) / (1 + child_N).

	Q is averaged from this node's color's perspective, so higher is better
	for the side about to move — no sign flip needed here."""
	total_N = sum(node.N.values()) or 1
	sqrt_N = math.sqrt(total_N)
	best_m, best_s = None, -1e18
	for m in node.moves:
		n = node.N[m]
		q = (node.W[m] / n) if n > 0 else 0.0
		# Proven moves get extreme priority/avoidance based on outcome.
		child = node.children.get(m)
		if child is not None and child.proven is not None:
			# Child's proven is from child's view = opposite of node's view.
			# A proven loss at the child = proven win for node — take it.
			# A proven win at the child = proven loss for node — avoid.
			q = -float(child.proven)
		u = C_PUCT * node.P[m] * sqrt_N / (1 + n)
		s = q + u
		if s > best_s:
			best_s = s
			best_m = m
	return best_m


def _update_proven(node):
	"""Propagate proven values from children up to `node`."""
	if node.proven is not None:
		return
	saw_proven_loss_child = False
	all_children_present_and_proven_win = True
	for m in node.moves:
		child = node.children.get(m)
		if child is None or child.proven is None:
			all_children_present_and_proven_win = False
			continue
		if child.proven == -1:
			# Child is proven loss for child's side-to-move = proven win for me.
			saw_proven_loss_child = True
			break
		elif child.proven != 1:
			all_children_present_and_proven_win = False
	if saw_proven_loss_child:
		node.proven = 1
	elif all_children_present_and_proven_win:
		node.proven = -1


def _iterate(root, problem, target_globals, tcolor, allowed_globals, evaluator):
	"""One MCTS iteration: select -> expand -> evaluate -> backup."""
	path = []
	node = root
	while node.expanded and not node.terminal and node.proven is None:
		m = _select_move(node)
		path.append((node, m))
		child = node.children.get(m)
		if child is None:
			b2 = node.board.copy()
			ok, _ = b2.play(m[0], m[1], node.color)
			if not ok:
				# Shouldn't happen if candidate_moves is correct; penalize and stop.
				node.N[m] += 1
				node.W[m] -= 1.0
				return
			child = Node(b2, bd.opponent(node.color))
			node.children[m] = child
		node = child

	if not node.expanded:
		v = _expand_node(node, problem, target_globals, tcolor, allowed_globals, evaluator)
	else:
		# Re-visited terminal/proven node.
		v = float(node.proven) if node.proven is not None else (node.leaf_value or 0.0)

	# Backup: each step up flips perspective.
	for parent_node, parent_move in reversed(path):
		v = -v
		parent_node.N[parent_move] += 1
		parent_node.W[parent_move] += v
		_update_proven(parent_node)


# ---------------------------------------------------------------------------
# Search drivers
# ---------------------------------------------------------------------------


def _build_root(board, color, problem, target_globals, tcolor, allowed_globals, evaluator,
		add_noise=False, rng=None):
	root = Node(board, color)
	_expand_node(root, problem, target_globals, tcolor, allowed_globals, evaluator)
	if add_noise and not root.terminal and root.moves:
		_add_dirichlet_noise(root, rng=rng)
	return root


def _search(root, problem, target_globals, tcolor, allowed_globals, evaluator,
		n_sims=None, time_budget=None):
	"""Run MCTS until `n_sims` simulations or `time_budget` seconds expire, or
	the root becomes proven. At least one of `n_sims` / `time_budget` must be
	provided."""
	assert n_sims is not None or time_budget is not None
	deadline = time.monotonic() + time_budget if time_budget is not None else None
	iters = 0
	while True:
		if n_sims is not None and iters >= n_sims:
			break
		if deadline is not None and time.monotonic() >= deadline:
			break
		if root.proven is not None:
			break
		if root.terminal or not root.moves:
			break
		_iterate(root, problem, target_globals, tcolor, allowed_globals, evaluator)
		iters += 1
	return iters


def _pi_vector(root, region):
	"""Normalized visit-count distribution as a length VIEW*VIEW vector. Mass
	lives only on legal moves; illegal cells are zero (the policy net learns
	this implicitly via the CE target)."""
	from . import neural
	pi = [0.0] * (reg.VIEW * reg.VIEW)
	total = sum(root.N.values())
	if total <= 0:
		return pi
	for m, n in root.N.items():
		idx = neural.global_to_index(m[0], m[1], region)
		if idx is not None:
			pi[idx] = n / total
	return pi


def _sample_move(root, temperature, rng=None):
	"""Pick a move from root visit counts. Temperature 0 = argmax; otherwise
	sample with weights proportional to N^(1/T)."""
	rng = rng or random
	moves = root.moves
	if not moves:
		return None
	if temperature <= 1e-6:
		return max(moves, key=lambda m: root.N[m])
	weights = [root.N[m] ** (1.0 / temperature) for m in moves]
	s = sum(weights)
	if s <= 0:
		return rng.choice(moves)
	r = rng.uniform(0, s)
	cum = 0.0
	for m, w in zip(moves, weights):
		cum += w
		if r <= cum:
			return m
	return moves[-1]


def run_simulations(board, color, problem, evaluator, n_sims=200,
		temperature=1.0, add_dirichlet_noise=True, rng=None):
	"""AlphaZero-style search step.

	Returns (move_xy, pi_vector, root_value):
	  * move_xy       picked move (or None if the position is terminal / no moves)
	  * pi_vector     length VIEW*VIEW float list (visit-count distribution)
	  * root_value    Q at the root (from `color`'s view); for terminal/proven
	                  roots, ±1; otherwise mean of N-weighted W over the root.
	"""
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	allowed_globals = solver._derive_allowed(problem)

	root = _build_root(board, color, problem, target_globals, tcolor, allowed_globals,
		evaluator, add_noise=add_dirichlet_noise, rng=rng)

	if root.terminal or not root.moves:
		pi = [0.0] * (reg.VIEW * reg.VIEW)
		v = float(root.proven if root.proven is not None else 0)
		return None, pi, v

	_search(root, problem, target_globals, tcolor, allowed_globals, evaluator, n_sims=n_sims)

	pi = _pi_vector(root, problem.region)
	move = _sample_move(root, temperature, rng=rng)

	if root.proven is not None:
		root_value = float(root.proven)
	else:
		total_N = sum(root.N.values()) or 1
		total_W = sum(root.W.values())
		root_value = total_W / total_N

	return move, pi, root_value


def best_move(board, color, problem, time_budget=2.0, tt=None, max_depth=None,
		leaf_depth=LEAF_DEPTH, verbose=False, evaluator=None):
	"""Live-play wrapper.

	Returns (score_from_player_perspective, move_xy). Score is quantized to
	+1 / 0 / -1 for verdict purposes (root-proven → exact; otherwise ±1 if
	root's robust-child Q exceeds 0.95 in magnitude, else 0).

	`tt` is reused as the alpha-beta transposition table across leaf evals
	when the tactical evaluator is in use. `max_depth` is accepted for
	interface compatibility with solver.best_move but unused (MCTS is anytime).

	If `evaluator` is None, prefer the neural evaluator when a trained
	checkpoint is loadable, else fall back to the tactical one."""
	del max_depth  # unused
	if tt is None:
		tt = {}
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	allowed_globals = solver._derive_allowed(problem)

	if evaluator is None:
		model = solver._load_policy()
		if model is not None:
			evaluator = neural_evaluator(model)
		else:
			evaluator = tactical_evaluator(leaf_depth=leaf_depth, ab_tt=tt)

	root = _build_root(board, color, problem, target_globals, tcolor, allowed_globals,
		evaluator, add_noise=False)
	if root.terminal or not root.moves:
		score_p = _flip(root.proven or 0, root.color, problem.player) if root.proven is not None else 0
		return int(score_p), None

	iters = _search(root, problem, target_globals, tcolor, allowed_globals, evaluator,
		time_budget=time_budget)

	# Robust child: most visits, tie-broken by Q.
	def _rank(m):
		n = root.N[m]
		q = (root.W[m] / n) if n > 0 else -1.0
		# Prefer proven-win children, avoid proven-loss children.
		child = root.children.get(m)
		bonus = 0
		if child is not None and child.proven is not None:
			bonus = -100 if child.proven == 1 else 100  # child win = our loss; avoid
		return (bonus, n, q)
	best_m = max(root.moves, key=_rank)

	if root.proven is not None:
		score_p = int(_flip(root.proven, root.color, problem.player))
	else:
		n = root.N[best_m]
		q_root = (root.W[best_m] / n) if n > 0 else 0.0
		score_color = q_root
		score_p_raw = _flip(score_color, root.color, problem.player)
		if score_p_raw > 0.95:
			score_p = 1
		elif score_p_raw < -0.95:
			score_p = -1
		else:
			score_p = 0

	if verbose:
		print(f"  mcts: {iters} iters, best={reg.format_global(*best_m)} "
			f"N={root.N[best_m]} Q_color={(root.W[best_m]/root.N[best_m]):+.3f} "
			f"proven={root.proven}", flush=True)

	return score_p, best_m
