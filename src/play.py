"""Interactive tsumego play loop driven by the pre-stored response tree."""

from . import board as bd
from . import region as reg
from . import database


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


def apply_move(board, coord, color, region):
	gr, gc = reg.local_to_global(coord, region)
	return board.play(gr, gc, color)


def play_problem(problem):
	board = problem.initial_board()
	region = problem.region
	player_color = problem.player
	opp_color = bd.opponent(player_color)
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

	node = {"branches": problem.tree}
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
			print("  pick a coordinate inside the 9x9 view or 'q' to quit.")
			continue

		try:
			move = raw.upper()
			reg.parse_local(move)
		except Exception as e:
			print(f"  could not parse '{raw}': {e}")
			continue

		ok, _ = apply_move(board, move, player_color, region)
		if not ok:
			print("  illegal move (occupied, suicide, or ko). try again.")
			continue

		branches = node.get("branches", {}) or {}
		print(render_view(board, region))
		if move not in branches:
			print("\n  This move is not in the problem database - treated as off-book.")
			print("  Likely incorrect or unexplored. Add a branch in the json to handle it.")
			return

		entry = branches[move]
		mark = "OK" if entry.get("correct") else "X "
		print(f"\n  {mark} {entry.get('comment', '')}")

		reply = entry.get("reply")
		if reply:
			ok2, _ = apply_move(board, reply, opp_color, region)
			if not ok2:
				print(f"  warning: stored reply {reply} is illegal - stopping.")
				return
			print(f"\n  Opponent plays {reply}.")
			print(render_view(board, region))

		next_branches = entry.get("branches") or {}
		if not next_branches:
			if entry.get("correct"):
				print("\n  Problem solved.")
			else:
				print("\n  End of variation - try a different first move.")
			return
		node = entry


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
