"""Pygame-ce interactive UI for tsumego play.

Renders the 9x9 region view, accepts mouse clicks for moves, and walks the
problem's pre-stored response tree (same logic as src/play.py).
"""

import pygame

from . import board as bd
from . import region as reg
from . import database


CELL = 60
BOARD_MARGIN = 48
BOARD_PX = (reg.VIEW - 1) * CELL
SIDEBAR_W = 420
WINDOW_W = BOARD_MARGIN * 2 + BOARD_PX + SIDEBAR_W
WINDOW_H = BOARD_MARGIN * 2 + BOARD_PX + 80  # extra for buttons

ORIGIN = (BOARD_MARGIN, BOARD_MARGIN)
SIDEBAR_X = BOARD_MARGIN * 2 + BOARD_PX

BG = (240, 217, 181)
SIDEBAR_BG = (250, 245, 230)
LINE = (40, 25, 10)
EDGE_LINE = (10, 5, 0)
BLACK = (15, 15, 15)
WHITE = (240, 240, 240)
STONE_OUTLINE = (10, 10, 10)
TEXT = (20, 20, 20)
MUTED = (110, 100, 80)
GOOD = (30, 130, 50)
BAD = (180, 40, 40)
HIGHLIGHT = (210, 170, 40)
HOVER_BG = (250, 235, 200)
BTN_BG = (210, 190, 150)
BTN_HOVER = (225, 205, 165)


def font(size=16, bold=False):
	return pygame.font.SysFont("arial,helvetica,sans-serif", size, bold=bold)


def render_text(surface, text, pos, size=16, color=TEXT, bold=False):
	surf = font(size, bold).render(text, True, color)
	surface.blit(surf, pos)
	return surf.get_rect(topleft=pos)


def wrap_text(text, width_px, size=14):
	f = font(size)
	out = []
	for paragraph in text.split("\n"):
		words = paragraph.split()
		cur = ""
		for w in words:
			cand = w if not cur else cur + " " + w
			if f.size(cand)[0] <= width_px:
				cur = cand
			else:
				if cur:
					out.append(cur)
				cur = w
		out.append(cur)
	return out


def view_to_screen(view_r, view_c):
	ox, oy = ORIGIN
	return ox + view_c * CELL, oy + view_r * CELL


def screen_to_view(x, y):
	"""Mouse position -> (view_r, view_c) on the 9x9 grid, or None."""
	ox, oy = ORIGIN
	col = round((x - ox) / CELL)
	row = round((y - oy) / CELL)
	if not (0 <= col < reg.VIEW and 0 <= row < reg.VIEW):
		return None
	sx, sy = view_to_screen(row, col)
	if (sx - x) ** 2 + (sy - y) ** 2 > (CELL * 0.45) ** 2:
		return None
	return row, col


class Button:
	def __init__(self, label, rect, on_click):
		self.label = label
		self.rect = pygame.Rect(rect)
		self.on_click = on_click
		self.hover = False

	def draw(self, screen):
		pygame.draw.rect(screen, BTN_HOVER if self.hover else BTN_BG, self.rect, border_radius=6)
		pygame.draw.rect(screen, LINE, self.rect, 1, border_radius=6)
		surf = font(15, bold=True).render(self.label, True, TEXT)
		screen.blit(surf, surf.get_rect(center=self.rect.center))

	def handle(self, event):
		if event.type == pygame.MOUSEMOTION:
			self.hover = self.rect.collidepoint(event.pos)
		elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
			if self.rect.collidepoint(event.pos):
				self.on_click()


def draw_board(screen, board, region, last_move=None, candidates=None, hover_xy=None, hover_color=None):
	es = reg.edges(region)
	has = es.__contains__
	ox, oy = ORIGIN
	w = BOARD_PX

	pad = CELL // 2 + 4
	panel = pygame.Rect(ox - pad, oy - pad, w + pad * 2, w + pad * 2)
	pygame.draw.rect(screen, BG, panel)

	for i in range(reg.VIEW):
		sy = oy + i * CELL
		x1 = panel.left if has("left") else ox
		x2 = panel.right if has("right") else ox + w
		pygame.draw.line(screen, LINE, (x1, sy), (x2, sy), 1)
		sx = ox + i * CELL
		y1 = panel.top if has("top") else oy
		y2 = panel.bottom if has("bottom") else oy + w
		pygame.draw.line(screen, LINE, (sx, y1), (sx, y2), 1)

	# Thicker stroke on real board edges
	ew = 3
	if has("top"):
		pygame.draw.line(screen, EDGE_LINE,
			(panel.left if has("left") else ox, oy),
			(panel.right if has("right") else ox + w, oy), ew)
	if has("bottom"):
		pygame.draw.line(screen, EDGE_LINE,
			(panel.left if has("left") else ox, oy + w),
			(panel.right if has("right") else ox + w, oy + w), ew)
	if has("left"):
		pygame.draw.line(screen, EDGE_LINE,
			(ox, panel.top if has("top") else oy),
			(ox, panel.bottom if has("bottom") else oy + w), ew)
	if has("right"):
		pygame.draw.line(screen, EDGE_LINE,
			(ox + w, panel.top if has("top") else oy),
			(ox + w, panel.bottom if has("bottom") else oy + w), ew)

	# Star points (4-4 hoshi) that fall in view
	(off_r, off_c), _ = reg.region_bounds(region)
	for sr in (3, 9, 15):
		for sc in (3, 9, 15):
			vr, vc = sr - off_r, sc - off_c
			if 0 <= vr < reg.VIEW and 0 <= vc < reg.VIEW:
				cx, cy = view_to_screen(vr, vc)
				pygame.draw.circle(screen, LINE, (cx, cy), 3)

	# Coordinate labels
	coord_f = font(11, bold=True)
	for c in range(reg.VIEW):
		letter = reg.LETTERS[c]
		cx, _ = view_to_screen(0, c)
		lbl = coord_f.render(letter, True, MUTED)
		screen.blit(lbl, lbl.get_rect(midbottom=(cx, oy - pad - 2)))
	for r in range(reg.VIEW):
		row_num = reg.VIEW - r
		_, cy = view_to_screen(r, 0)
		lbl = coord_f.render(str(row_num), True, MUTED)
		screen.blit(lbl, lbl.get_rect(midright=(ox - pad - 4, cy)))

	# Candidate book-move markers
	if candidates:
		for coord in candidates:
			try:
				vr, vc = reg.parse_local(coord)
			except Exception:
				continue
			cx, cy = view_to_screen(vr, vc)
			pygame.draw.circle(screen, HIGHLIGHT, (cx, cy), 5, 1)

	# Stones
	radius = int(CELL * 0.45)
	for r in range(reg.VIEW):
		for c in range(reg.VIEW):
			v = board.at(off_r + r, off_c + c)
			if v == bd.EMPTY:
				continue
			cx, cy = view_to_screen(r, c)
			color = BLACK if v == bd.BLACK else WHITE
			pygame.draw.circle(screen, color, (cx, cy), radius)
			pygame.draw.circle(screen, STONE_OUTLINE, (cx, cy), radius, 1)

	# Last-move marker
	if last_move:
		try:
			vr, vc = reg.parse_local(last_move)
			cx, cy = view_to_screen(vr, vc)
			marker = (220, 60, 60)
			pygame.draw.circle(screen, marker, (cx, cy), 5, 2)
		except Exception:
			pass

	# Ghost stone on hover
	if hover_xy is not None and hover_color is not None:
		vr, vc = hover_xy
		v = board.at(off_r + vr, off_c + vc)
		if v == bd.EMPTY:
			cx, cy = view_to_screen(vr, vc)
			ghost = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
			rgb = BLACK if hover_color == bd.BLACK else WHITE
			pygame.draw.circle(ghost, (*rgb, 120), (radius + 1, radius + 1), radius)
			screen.blit(ghost, (cx - radius - 1, cy - radius - 1))


def picker_screen(screen, clock):
	problems = []
	for d in database.DIFFICULTIES:
		problems.extend(database.list_problems(d))

	chosen = {"v": None}
	row_h = 32
	scroll = {"y": 0}

	def quit_clicked():
		chosen["v"] = "__quit__"

	quit_btn = Button("Quit", (WINDOW_W - 100, 18, 80, 32), quit_clicked)

	header_h = 64
	list_top = header_h
	list_bottom = WINDOW_H - 20

	while chosen["v"] is None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return None
			if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
				return None
			quit_btn.handle(event)
			if event.type == pygame.MOUSEWHEEL:
				max_scroll = max(0, len(problems) * row_h - (list_bottom - list_top))
				scroll["y"] = max(0, min(max_scroll, scroll["y"] - event.y * row_h))
			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				mx, my = event.pos
				if list_top <= my < list_bottom and 20 <= mx <= WINDOW_W - 20:
					idx = (my - list_top + scroll["y"]) // row_h
					if 0 <= idx < len(problems):
						chosen["v"] = idx

		screen.fill(SIDEBAR_BG)
		render_text(screen, "DeepGoProblem — pick a tsumego", (20, 18), size=22, bold=True)
		quit_btn.draw(screen)
		if not problems:
			render_text(screen, "No problems found. Add JSON files under problems/<difficulty>/.",
				(20, header_h + 10), size=14, color=BAD)
		else:
			clip = pygame.Rect(0, list_top, WINDOW_W, list_bottom - list_top)
			screen.set_clip(clip)
			mx, my = pygame.mouse.get_pos()
			for i, p in enumerate(problems):
				y = list_top + i * row_h - scroll["y"]
				if y + row_h < list_top or y > list_bottom:
					continue
				rect = pygame.Rect(20, y + 2, WINDOW_W - 40, row_h - 6)
				if rect.collidepoint(mx, my):
					pygame.draw.rect(screen, HOVER_BG, rect, border_radius=4)
				pygame.draw.rect(screen, LINE, rect, 1, border_radius=4)
				diff_color = {"easy": GOOD, "medium": (200, 130, 30), "hard": BAD}.get(p.difficulty, TEXT)
				render_text(screen, p.difficulty, (rect.x + 10, rect.y + 6), size=13, color=diff_color, bold=True)
				render_text(screen, p.id, (rect.x + 90, rect.y + 6), size=13, bold=True)
				render_text(screen, p.region, (rect.x + 200, rect.y + 6), size=13, color=MUTED)
				desc = p.description if len(p.description) <= 70 else p.description[:67] + "..."
				render_text(screen, desc, (rect.x + 320, rect.y + 6), size=13)
			screen.set_clip(None)
		pygame.display.flip()
		clock.tick(60)

	if chosen["v"] == "__quit__":
		return None
	return problems[chosen["v"]]


def play_screen(screen, clock, problem):
	region = problem.region
	(off_r, off_c), _ = reg.region_bounds(region)
	player_color = problem.player
	opp_color = bd.opponent(player_color)

	state = {
		"board": problem.initial_board(),
		"node": {"branches": problem.tree},
		"last_move": None,
		"finished": False,
		"messages": [],
	}

	def status(text, color=TEXT):
		state["messages"].append((text, color))
		if len(state["messages"]) > 14:
			del state["messages"][:len(state["messages"]) - 14]

	def reset():
		state["board"] = problem.initial_board()
		state["node"] = {"branches": problem.tree}
		state["last_move"] = None
		state["finished"] = False
		state["messages"] = []
		status("Position reset.", MUTED)

	action = {"choice": None}

	def back():
		action["choice"] = "back"

	def quit_game():
		action["choice"] = "quit"

	btn_y = WINDOW_H - 50
	reset_btn = Button("Reset (R)", (SIDEBAR_X + 14, btn_y, 110, 36), reset)
	back_btn = Button("Pick another", (SIDEBAR_X + 134, btn_y, 150, 36), back)
	quit_btn = Button("Quit", (SIDEBAR_X + 294, btn_y, 90, 36), quit_game)

	status(f"You play {bd.letter_from_color(player_color)}. Click an intersection.", MUTED)

	while action["choice"] is None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return None
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return "back"
				if event.key == pygame.K_r:
					reset()
			reset_btn.handle(event)
			back_btn.handle(event)
			quit_btn.handle(event)

			if not state["finished"] and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				hit = screen_to_view(*event.pos)
				if hit is None:
					continue
				vr, vc = hit
				coord = reg.format_local(vr, vc)
				ok, _ = state["board"].play(off_r + vr, off_c + vc, player_color)
				if not ok:
					status(f"{coord}: illegal (occupied, suicide, or ko).", BAD)
					continue
				state["last_move"] = coord
				branches = (state["node"].get("branches") or {})
				if coord not in branches:
					status(f"{coord}: off-book — add a branch in the JSON to handle this.", BAD)
					state["finished"] = True
					continue
				entry = branches[coord]
				ok_flag = bool(entry.get("correct"))
				mark = "OK" if ok_flag else "X "
				status(f"{coord}: {mark} {entry.get('comment','')}", GOOD if ok_flag else BAD)
				reply = entry.get("reply")
				if reply:
					rr, rc = reg.local_to_global(reply, region)
					ok2, _ = state["board"].play(rr, rc, opp_color)
					if ok2:
						state["last_move"] = reply
						status(f"Opponent plays {reply}.", MUTED)
					else:
						status(f"Stored reply {reply} is illegal — stopping.", BAD)
						state["finished"] = True
						continue
				next_branches = entry.get("branches") or {}
				if not next_branches:
					if ok_flag:
						status("Problem solved.", GOOD)
					else:
						status("End of variation. Reset and try a different move.", BAD)
					state["finished"] = True
				else:
					state["node"] = entry

		# Render
		screen.fill(SIDEBAR_BG)
		pygame.draw.rect(screen, BG, (0, 0, SIDEBAR_X, WINDOW_H))

		hover_xy = None
		if not state["finished"]:
			mp = pygame.mouse.get_pos()
			hover_xy = screen_to_view(*mp)

		candidates = list((state["node"].get("branches") or {}).keys()) if not state["finished"] else None
		draw_board(screen, state["board"], region,
			last_move=state["last_move"],
			candidates=candidates,
			hover_xy=hover_xy,
			hover_color=player_color if not state["finished"] else None,
		)

		# Sidebar
		sx = SIDEBAR_X + 16
		y = 20
		render_text(screen, problem.id, (sx, y), size=20, bold=True)
		y += 28
		render_text(screen, f"{problem.difficulty}   |   {problem.region}", (sx, y), size=13, color=MUTED)
		y += 22
		you = "Black (X)" if problem.player == bd.BLACK else "White (O)"
		render_text(screen, f"You play: {you}", (sx, y), size=14)
		y += 20
		if problem.goal:
			tgt = ", ".join(problem.target) if problem.target else "-"
			render_text(screen, f"Goal: {problem.goal}   target: {tgt}", (sx, y), size=14)
			y += 20
		if problem.description:
			for line in wrap_text(problem.description, WINDOW_W - sx - 16, 14):
				render_text(screen, line, (sx, y), size=14)
				y += 18
		y += 12

		render_text(screen, "Log", (sx, y), size=14, bold=True)
		y += 22
		log_w = WINDOW_W - sx - 16
		log_bottom = btn_y - 10
		for text, color in state["messages"]:
			if y >= log_bottom:
				break
			for line in wrap_text(text, log_w, 13):
				if y >= log_bottom:
					break
				render_text(screen, line, (sx, y), size=13, color=color)
				y += 16

		reset_btn.draw(screen)
		back_btn.draw(screen)
		quit_btn.draw(screen)
		pygame.display.flip()
		clock.tick(60)

	return action["choice"]


def main(args):
	initial = None
	if args and args[0] not in ("-h", "--help"):
		initial = database.find_problem(args[0])
		if initial is None:
			print(f"problem id '{args[0]}' not found.")
			return

	pygame.init()
	screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
	pygame.display.set_caption("DeepGoProblem")
	clock = pygame.time.Clock()
	try:
		problem = initial
		while True:
			if problem is None:
				problem = picker_screen(screen, clock)
				if problem is None:
					return
			result = play_screen(screen, clock, problem)
			if result in (None, "quit"):
				return
			problem = None  # "back" -> picker
	finally:
		pygame.quit()
