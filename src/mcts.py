"""Monte Carlo Tree Search with PUCT selection and shallow alpha-beta leaf eval.

Replaces direct alpha-beta as the live-play engine. Public entry point:

    mcts.best_move(board, color, problem, time_budget=2.0, tt=None, ...)
        -> (score_from_player_perspective, move_xy)

Internally every node stores values from its own side-to-move perspective; we
negate on backup so the alternation is implicit. Proven ±1 values (from a leaf
alpha-beta that resolved within `leaf_depth`) propagate up the tree exactly:
if any child is a proven loss for the child's side-to-move, the parent has a
proven win; if every child is a proven win, the parent is a proven loss.

The leaf evaluator is the existing `solver.alphabeta` at fixed shallow depth.
That keeps the engine sound for short tactical resolutions while leaving room
to swap in a value network later.
"""

import math
import time

from . import board as bd
from . import region as reg
from . import solver


C_PUCT = 1.4
LEAF_DEPTH = 4   # alpha-beta plies used to value an unexpanded leaf


class Node:
	__slots__ = ("board", "color", "moves", "P", "N", "W", "children",
		"expanded", "terminal", "proven")

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


def _flip(v, node_color, player):
	"""Convert a value between problem.player view and node_color view.
	Involution — same formula either direction."""
	return v if node_color == player else -v


def _expand_node(node, problem, target_globals, tcolor, allowed_globals):
	"""Mark terminal or compute candidates+uniform priors. Sets node.expanded."""
	val_p = solver.evaluate(node.board, problem, target_globals, tcolor)
	if val_p in (1, -1):
		node.terminal = True
		node.proven = _flip(val_p, node.color, problem.player)
		node.expanded = True
		return
	moves = solver.candidate_moves(
		node.board, node.color, target_globals, tcolor, problem.region, allowed_globals,
	)
	if not moves:
		# No legal candidates: side-to-move can't continue. Treat as loss for it.
		node.terminal = True
		node.proven = -1
		node.expanded = True
		return
	node.moves = moves
	# Rank-based prior derived from solver.order_moves (libs/captures heuristic).
	# Uniform priors leave MCTS exploring tactically irrelevant moves; this nudges
	# visits toward known-good shapes without overriding exploration entirely.
	ordered = solver.order_moves(
		node.board, moves, node.color, target_globals, tcolor, problem.region,
	)
	weights = [1.0 / (i + 1) for i in range(len(ordered))]
	total = sum(weights)
	for i, m in enumerate(ordered):
		node.P[m] = weights[i] / total
		node.N[m] = 0
		node.W[m] = 0.0
	node.expanded = True


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


def _iterate(root, problem, target_globals, tcolor, allowed_globals, leaf_depth, ab_tt):
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
		_expand_node(node, problem, target_globals, tcolor, allowed_globals)

	# Evaluate.
	if node.proven is not None:
		v = float(node.proven)
	else:
		score_p, _ = solver.alphabeta(
			node.board, node.color, problem, target_globals,
			leaf_depth, -solver.INF, solver.INF,
			tt=ab_tt, tcolor=tcolor, allowed_globals=allowed_globals,
		)
		v = float(_flip(score_p, node.color, problem.player))
		if score_p in (1, -1):
			node.proven = int(v)

	# Backup: each step up flips perspective.
	for parent_node, parent_move in reversed(path):
		v = -v
		parent_node.N[parent_move] += 1
		parent_node.W[parent_move] += v
		_update_proven(parent_node)


def best_move(board, color, problem, time_budget=2.0, tt=None, max_depth=None,
		leaf_depth=LEAF_DEPTH, verbose=False):
	"""Live-play wrapper.

	Returns (score_from_player_perspective, move_xy). Score is quantized to
	+1 / 0 / -1 for verdict purposes (root-proven → exact; otherwise ±1 if
	root's robust-child Q exceeds 0.95 in magnitude, else 0).

	`tt` is reused as the alpha-beta transposition table across leaf evals.
	`max_depth` is accepted for interface compatibility with solver.best_move
	but unused (MCTS is anytime — bounded by `time_budget`)."""
	del max_depth  # unused
	if tt is None:
		tt = {}
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	allowed_globals = solver._derive_allowed(problem)

	root = Node(board, color)
	_expand_node(root, problem, target_globals, tcolor, allowed_globals)
	if root.terminal or not root.moves:
		score_p = _flip(root.proven or 0, root.color, problem.player) if root.proven is not None else 0
		return int(score_p), None

	deadline = time.monotonic() + time_budget
	iters = 0
	while time.monotonic() < deadline:
		_iterate(root, problem, target_globals, tcolor, allowed_globals, leaf_depth, tt)
		iters += 1
		if root.proven is not None:
			break

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
