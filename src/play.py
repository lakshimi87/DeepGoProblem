"""Interactive tsumego play loop with on-the-fly solver responses.

No pre-stored response tree: after each player move, the solver picks the
opponent's best reply and a verdict (still winning / uncertain / refuted)."""

from . import board as bd
from . import region as reg
from . import database
from . import solver
from . import mcts


LIVE_BUDGET = 2.0  # seconds per opponent reply


def render_view(board, region):
	(off_r, off_c), _ = reg.region_bounds(region)
	es = reg.edges(region)
	top, bot = "top" in es, "bottom" in es
	left, right = "left" in es, "right" in es

	width = reg.VIEW
	lines = []
	col_header = "    " + " ".join(reg.LETTERS)
	lines.append(col_header)
	if top:
		lines.append("    " + "- " * width)
	for r in range(width):
		row_num = width - r
		cells = []
		for c in range(width):
			v = board.at(off_r + r, off_c + c)
			cells.append("X" if v == bd.BLACK else ("O" if v == bd.WHITE else "."))
		lcap = "|" if left else " "
		rcap = "|" if right else " "
		lines.append(f"{row_num:>2} {lcap} " + " ".join(cells) + f" {rcap}")
	if bot:
		lines.append("    " + "- " * width)
	lines.append(col_header)
	tag = ", ".join(sorted(es)) if es else "none (open position)"
	lines.append(f"   board edges in view: {tag}")
	return "\n".join(lines)


def choose_problem():
	groups = {d: database.list_problems(d) for d in database.DIFFICULTIES}
	print()
	print("Available problems:")
	flat = []
	for d in database.DIFFICULTIES:
		for p in groups[d]:
			flat.append(p)
			print(f"  [{len(flat):>2}] {d:<6s}  {p.id:<14s}  {p.region:<13s}  {p.description}")
	if not flat:
		print("  (none - add json files under problems/<difficulty>/)")
		return None
	while True:
		raw = input("\nPick a number (q to quit): ").strip().lower()
		if raw in ("q", "quit", "exit", ""):
			return None
		try:
			idx = int(raw)
			if 1 <= idx <= len(flat):
				return flat[idx - 1]
		except ValueError:
			pass
		print("not a valid choice.")


def _global_to_view_str(coord_global, region):
	"""Display a global Go coord using the region's local view labels when possible."""
	vw = reg.global_to_view(coord_global, region)
	if vw is None:
		return coord_global
	return reg.format_local(*vw)


def play_problem(problem, time_budget=LIVE_BUDGET):
	board = problem.initial_board()
	region = problem.region
	player_color = problem.player
	opp_color = bd.opponent(player_color)
	tcolor = solver.target_color(problem)
	target_globals = [reg.parse_global(t) for t in problem.target]
	tt = {}
	you = "black/X" if player_color == bd.BLACK else "white/O"

	print()
	print(f"Problem: {problem.id}   difficulty={problem.difficulty}   region={problem.region}")
	print(f"You play: {bd.letter_from_color(player_color)} ({you})")
	if problem.goal:
		print(f"Goal:     {problem.goal}  target={problem.target}")
	if problem.description:
		print(f"Note:     {problem.description}")
	print()
	print(render_view(board, region))

	while True:
		raw = input("\nYour move (e.g. C5; 'show', '?', 'q'): ").strip().lower()
		if raw in ("q", "quit", "exit"):
			print("aborted.")
			return
		if raw == "show":
			print(render_view(board, region))
			continue
		if raw == "?":
			print(f"  {problem.description}")
			continue
		if raw in ("pass", ""):
			print("  pick a coordinate inside the 11x11 view or 'q' to quit.")
			continue

		try:
			move_local = raw.upper()
			gr, gc = reg.local_to_global(move_local, region)
		except Exception as e:
			print(f"  could not parse '{raw}': {e}")
			continue

		ok, _ = board.play(gr, gc, player_color)
		if not ok:
			print("  illegal move (occupied, suicide, or ko). try again.")
			continue

		print(render_view(board, region))

		eval_now = solver.evaluate(board, problem, target_globals, tcolor)
		if eval_now == 1:
			print("\n  OK Problem solved.")
			return
		if eval_now == -1:
			print("\n  X Goal already failed.")
			return

		print("\n  (engine thinking...)")
		score, opp_xy = mcts.best_move(board, opp_color, problem, time_budget=time_budget, tt=tt)
		mark = {1: "OK", 0: "? ", -1: "X "}[score]
		verdict = {
			1: "still winning.",
			0: "uncertain within solver depth.",
			-1: "this move loses; opponent refutes.",
		}[score]
		print(f"  {mark} {verdict}")

		if opp_xy is None:
			eval_after = solver.evaluate(board, problem, target_globals, tcolor)
			if eval_after == 1:
				print("\n  OK Problem solved (opponent has no legal reply).")
			elif eval_after == -1:
				print("\n  X Lost.")
			else:
				print("\n  Opponent has no candidate move — position stable, stopping.")
			return

		ok2, _ = board.play(opp_xy[0], opp_xy[1], opp_color)
		if not ok2:
			print(f"  warning: solver suggested {reg.format_global(*opp_xy)}, illegal — stopping.")
			return
		opp_global = reg.format_global(*opp_xy)
		print(f"  Opponent plays {_global_to_view_str(opp_global, region)}.")
		print(render_view(board, region))

		eval_after = solver.evaluate(board, problem, target_globals, tcolor)
		if eval_after == 1:
			print("\n  OK Problem solved.")
			return
		if eval_after == -1:
			print("\n  X Lost.")
			return


def main(args):
	problem = None
	if args and args[0] not in ("-h", "--help"):
		problem = database.find_problem(args[0])
		if problem is None:
			print(f"problem id '{args[0]}' not found.")
			return
	if problem is None:
		problem = choose_problem()
		if problem is None:
			return
	play_problem(problem)
