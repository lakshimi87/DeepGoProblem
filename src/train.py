"""AlphaZero-style self-play training for the tsumego policy+value network.

Outer loop:
    for iteration in 1..N:
        self-play games using MCTS with the current network -> (state, pi, z)
        append to replay buffer (bounded FIFO)
        train: policy CE against pi + value MSE against z
        checkpoint to models/policy.pt

Game outcome `z`:
  * if terminal eval (solver.evaluate) returns ±1, that's the truth.
  * if the game reaches `--max-moves` without resolving, fall back to
    `solver.solve` at moderate depth so we still get a ground-truth value for
    every recorded position (better than bootstrapping from an untrained net).

Usage:
    python -m src.main train [--iterations N] [--games-per-problem G]
        [--sims S] [--epochs-per-iter E] [--batch-size B] [--lr L]
        [--max-moves M] [--buffer K] [--reset]
"""

import random
import time
from collections import deque
from pathlib import Path

from . import database
from . import region as reg
from . import board as bd
from . import solver
from . import mcts


def _parse_args(args):
	"""Pull recognised flags out of args; ignore unknown so we don't break callers."""
	def _flag(name, default, cast):
		if name in args:
			return cast(args[args.index(name) + 1])
		return default
	return dict(
		iterations=_flag("--iterations", 20, int),
		games_per_problem=_flag("--games-per-problem", 4, int),
		sims=_flag("--sims", 80, int),
		epochs_per_iter=_flag("--epochs-per-iter", 8, int),
		batch_size=_flag("--batch-size", 64, int),
		lr=_flag("--lr", 1e-3, float),
		max_moves=_flag("--max-moves", 30, int),
		buffer=_flag("--buffer", 4096, int),
		# When the game truncates without terminal, ask the solver for the truth at
		# this depth. Set to 0 to skip the truncated game entirely (no value target).
		truncate_solve_depth=_flag("--truncate-solve-depth", 12, int),
		temperature_moves=_flag("--temperature-moves", 8, int),
		seed=_flag("--seed", None, int),
		reset="--reset" in args,
	)


def _play_one_game(problem, evaluator, sims, max_moves, temperature_moves,
		truncate_solve_depth, rng, verbose=False):
	"""Run a single self-play game on `problem`. Returns a list of
	(state_tensor, pi_list, z_float) tuples, one per move played. `z` is from
	that move's side-to-move view (so the loss target matches the encoded board's
	'own' channel)."""
	from . import neural
	board = problem.initial_board()
	color = problem.player
	target_globals = [reg.parse_global(t) for t in problem.target]
	tcolor = solver.target_color(problem)

	# (state_tensor, pi, color_to_move_at_state)
	trajectory = []
	outcome_p = None  # from problem.player view
	for move_num in range(max_moves):
		# Terminal check before searching: if already resolved, no point in MCTS.
		val_p = solver.evaluate(board, problem, target_globals, tcolor)
		if val_p in (1, -1):
			outcome_p = val_p
			break

		temperature = 1.0 if move_num < temperature_moves else 1e-9
		move, pi, _ = mcts.run_simulations(
			board, color, problem, evaluator,
			n_sims=sims, temperature=temperature,
			add_dirichlet_noise=True, rng=rng,
		)
		if move is None:
			break

		state = neural.board_to_tensor(board, problem.region, color)
		trajectory.append((state, pi, color))

		ok, _ = board.play(move[0], move[1], color)
		if not ok:
			break
		color = bd.opponent(color)

	if outcome_p is None:
		# Either truncated or no legal move from a non-terminal position. Ask the
		# solver to resolve the position we ended up in, from problem.player view.
		if truncate_solve_depth > 0:
			# Build a synthetic Problem snapshot: the search uses `problem.player`
			# as the maximizing side, so we plug the current board into solve.
			score_p, _ = solver.alphabeta(
				board, problem.player, problem, target_globals,
				truncate_solve_depth, -solver.INF, solver.INF,
				tt={}, tcolor=tcolor,
			)
			outcome_p = score_p if score_p in (1, -1) else 0
		else:
			outcome_p = 0

	# Convert per-state outcome from problem.player view to side-to-move view.
	data = []
	for state, pi, c in trajectory:
		z = outcome_p if c == problem.player else -outcome_p
		data.append((state, pi, float(z)))
	if verbose:
		print(f"    {problem.id}: {len(trajectory)} moves, outcome_p={outcome_p:+d}",
			flush=True)
	return data


def _train_step(model, optimizer, batch, torch):
	states = torch.stack([s for s, _, _ in batch])
	pis = torch.tensor([p for _, p, _ in batch], dtype=torch.float32)
	zs = torch.tensor([z for _, _, z in batch], dtype=torch.float32)

	policy_logits, values = model(states)
	log_probs = torch.log_softmax(policy_logits, dim=-1)
	# Soft cross-entropy against the visit distribution.
	policy_loss = -(pis * log_probs).sum(dim=-1).mean()
	value_loss = ((values - zs) ** 2).mean()
	loss = policy_loss + value_loss

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def main(args):
	try:
		import torch
		import torch.optim as optim
	except ImportError:
		print("torch is required. run ./setup.sh --train")
		return

	from . import neural

	cfg = _parse_args(args)
	if cfg["seed"] is not None:
		random.seed(cfg["seed"])
		torch.manual_seed(cfg["seed"])
	rng = random.Random(cfg["seed"])

	problems = database.list_problems()
	if not problems:
		print("no problems found. nothing to train.")
		return

	out = Path(__file__).resolve().parent.parent / "models" / "policy.pt"
	out.parent.mkdir(exist_ok=True)

	model = neural.build_model()
	if not cfg["reset"] and out.exists():
		try:
			state = torch.load(out, map_location="cpu", weights_only=True)
			model.load_state_dict(state, strict=False)
			print(f"resuming from {out} (strict=False — value-head weights kept "
				"random if checkpoint predates them)")
		except Exception as e:
			print(f"warn: could not resume from {out} ({e}); starting fresh")
	else:
		if cfg["reset"]:
			print("starting from a freshly initialised model (--reset)")
		else:
			print("starting from a freshly initialised model")

	optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
	replay = deque(maxlen=cfg["buffer"])

	print(f"alphazero self-play training: {len(problems)} problems, "
		f"{cfg['iterations']} iterations, {cfg['games_per_problem']} games/problem, "
		f"{cfg['sims']} sims/move", flush=True)

	for it in range(1, cfg["iterations"] + 1):
		t0 = time.monotonic()
		# Reset solver caches so a stale model's policy ordering doesn't bleed
		# across iterations (each iter trains the policy that order_moves reads).
		solver._POLICY = None
		solver._POLICY_CACHE.clear()
		evaluator = mcts.neural_evaluator(model)

		# --- self-play ---
		new_samples = 0
		model.eval()
		for p in problems:
			for _ in range(cfg["games_per_problem"]):
				game_data = _play_one_game(
					p, evaluator,
					sims=cfg["sims"],
					max_moves=cfg["max_moves"],
					temperature_moves=cfg["temperature_moves"],
					truncate_solve_depth=cfg["truncate_solve_depth"],
					rng=rng,
				)
				replay.extend(game_data)
				new_samples += len(game_data)
		t_sp = time.monotonic() - t0

		# --- training ---
		model.train()
		buffer = list(replay)
		if not buffer:
			print(f"iter {it}: no samples collected; skipping training")
			continue
		random.shuffle(buffer)
		bs = cfg["batch_size"]
		losses = []
		ploss = []
		vloss = []
		t1 = time.monotonic()
		for epoch in range(cfg["epochs_per_iter"]):
			random.shuffle(buffer)
			for i in range(0, len(buffer), bs):
				batch = buffer[i:i + bs]
				if not batch:
					continue
				l, pl, vl = _train_step(model, optimizer, batch, torch)
				losses.append(l)
				ploss.append(pl)
				vloss.append(vl)
		t_tr = time.monotonic() - t1

		torch.save(model.state_dict(), out)

		avg = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
		print(f"iter {it:3d}: +{new_samples} samples "
			f"(buffer={len(replay)})  sp={t_sp:5.1f}s  tr={t_tr:5.1f}s  "
			f"loss={avg(losses):.4f}  p_ce={avg(ploss):.4f}  v_mse={avg(vloss):.4f}",
			flush=True)

	print(f"saved {out}")
