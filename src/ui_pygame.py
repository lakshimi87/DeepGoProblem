"""Pygame-ce interactive UI for tsumego play.

Renders the 11x11 region view, accepts mouse clicks for moves, and walks the
problem's pre-stored response tree (same logic as src/play.py).
"""

import json

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
	"""Mouse position -> (view_r, view_c) on the 11x11 grid, or None."""
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


class TextInput:
	"""Minimal focus-aware text input. Single- or multi-line."""

	def __init__(self, rect, value="", placeholder="", multiline=False):
		self.rect = pygame.Rect(rect)
		self.value = value
		self.placeholder = placeholder
		self.focused = False
		self.multiline = multiline

	def draw(self, screen):
		bg = (255, 250, 230) if self.focused else (255, 255, 255)
		pygame.draw.rect(screen, bg, self.rect, border_radius=3)
		pygame.draw.rect(screen, LINE, self.rect, 1, border_radius=3)
		f = font(13)
		shown = self.value
		color = TEXT
		if not shown and not self.focused and self.placeholder:
			shown = self.placeholder
			color = MUTED

		if self.multiline:
			y = self.rect.y + 4
			for raw in shown.split("\n") if shown else [""]:
				for line in wrap_text(raw, self.rect.width - 8, 13) or [""]:
					if y + 16 > self.rect.bottom - 2:
						break
					screen.blit(f.render(line, True, color), (self.rect.x + 4, y))
					y += 16
		else:
			clip = screen.get_clip()
			screen.set_clip(self.rect.inflate(-4, -4))
			screen.blit(f.render(shown, True, color),
				(self.rect.x + 4, self.rect.y + (self.rect.height - f.get_height()) // 2))
			screen.set_clip(clip)

		if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
			if self.multiline:
				lines = self.value.split("\n") if self.value else [""]
				last = lines[-1]
				cx = self.rect.x + 4 + f.size(last)[0]
				cy = self.rect.y + 4 + (len(lines) - 1) * 16
			else:
				cx = self.rect.x + 4 + f.size(self.value)[0]
				cy = self.rect.y + (self.rect.height - 16) // 2
			pygame.draw.line(screen, TEXT, (cx, cy), (cx, cy + 16), 1)

	def handle(self, event):
		if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
			self.focused = self.rect.collidepoint(event.pos)
			return self.focused
		if event.type == pygame.KEYDOWN and self.focused:
			if event.key == pygame.K_BACKSPACE:
				self.value = self.value[:-1]
				return True
			if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
				if self.multiline:
					self.value += "\n"
				else:
					self.focused = False
				return True
			if event.key in (pygame.K_TAB, pygame.K_ESCAPE):
				self.focused = False
				return True
			ch = event.unicode
			if ch and ch.isprintable():
				self.value += ch
				return True
		return False


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
			vw = reg.global_to_view(coord, region)
			if vw is None:
				continue
			vr, vc = vw
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
		vw = reg.global_to_view(last_move, region)
		if vw is not None:
			vr, vc = vw
			cx, cy = view_to_screen(vr, vc)
			marker = (220, 60, 60)
			pygame.draw.circle(screen, marker, (cx, cy), 5, 2)

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
	edit_mode = {"v": False}
	row_h = 32
	scroll = {"y": 0}

	def quit_clicked():
		chosen["v"] = "__quit__"

	def new_clicked():
		chosen["v"] = ("new", None)

	def toggle_edit():
		edit_mode["v"] = not edit_mode["v"]

	new_btn = Button("+ New", (WINDOW_W - 290, 18, 80, 32), new_clicked)
	edit_btn = Button("", (WINDOW_W - 200, 18, 130, 32), toggle_edit)
	quit_btn = Button("Quit", (WINDOW_W - 60, 18, 60, 32), quit_clicked)
	header_buttons = [new_btn, edit_btn, quit_btn]

	header_h = 64
	list_top = header_h
	list_bottom = WINDOW_H - 20

	while chosen["v"] is None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return None
			if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
				return None
			for btn in header_buttons:
				btn.handle(event)
			if event.type == pygame.MOUSEWHEEL:
				max_scroll = max(0, len(problems) * row_h - (list_bottom - list_top))
				scroll["y"] = max(0, min(max_scroll, scroll["y"] - event.y * row_h))
			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				mx, my = event.pos
				if list_top <= my < list_bottom and 20 <= mx <= WINDOW_W - 20:
					idx = (my - list_top + scroll["y"]) // row_h
					if 0 <= idx < len(problems):
						mode = "edit" if edit_mode["v"] else "play"
						chosen["v"] = (mode, problems[idx])

		edit_btn.label = "Edit Mode: ON" if edit_mode["v"] else "Edit Mode: off"

		screen.fill(SIDEBAR_BG)
		render_text(screen, "DeepGoProblem — pick a tsumego", (20, 18), size=22, bold=True)
		hint = "click a row to edit it, or '+ New' to add a problem" if edit_mode["v"] \
			else "click a row to play, or toggle Edit Mode to edit"
		render_text(screen, hint, (20, 44), size=12, color=MUTED)
		for btn in header_buttons:
			btn.draw(screen)
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
	return chosen["v"]


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

	def edit_current():
		action["choice"] = "edit"

	def quit_game():
		action["choice"] = "quit"

	btn_y = WINDOW_H - 50
	reset_btn = Button("Reset (R)", (SIDEBAR_X + 14, btn_y, 100, 36), reset)
	edit_btn = Button("Edit (E)", (SIDEBAR_X + 122, btn_y, 80, 36), edit_current)
	back_btn = Button("Pick another", (SIDEBAR_X + 210, btn_y, 130, 36), back)
	quit_btn = Button("Quit", (SIDEBAR_X + 348, btn_y, 60, 36), quit_game)

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
				if event.key == pygame.K_e:
					edit_current()
			reset_btn.handle(event)
			edit_btn.handle(event)
			back_btn.handle(event)
			quit_btn.handle(event)

			if not state["finished"] and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				hit = screen_to_view(*event.pos)
				if hit is None:
					continue
				vr, vc = hit
				local_label = reg.format_local(vr, vc)
				gr, gc = off_r + vr, off_c + vc
				coord_global = reg.format_global(gr, gc)
				ok, _ = state["board"].play(gr, gc, player_color)
				if not ok:
					status(f"{local_label}: illegal (occupied, suicide, or ko).", BAD)
					continue
				state["last_move"] = coord_global
				branches = (state["node"].get("branches") or {})
				if coord_global not in branches:
					status(f"{local_label}: off-book — add a branch in the JSON to handle this.", BAD)
					state["finished"] = True
					continue
				entry = branches[coord_global]
				ok_flag = bool(entry.get("correct"))
				mark = "OK" if ok_flag else "X "
				status(f"{local_label}: {mark} {entry.get('comment','')}", GOOD if ok_flag else BAD)
				reply = entry.get("reply")
				if reply:
					rr, rc = reg.parse_global(reply)
					ok2, _ = state["board"].play(rr, rc, opp_color)
					if ok2:
						state["last_move"] = reply
						vw = reg.global_to_view(reply, region)
						reply_label = reg.format_local(*vw) if vw is not None else reply
						status(f"Opponent plays {reply_label}.", MUTED)
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
		edit_btn.draw(screen)
		back_btn.draw(screen)
		quit_btn.draw(screen)
		pygame.display.flip()
		clock.tick(60)

	return action["choice"]


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


def _new_problem_data():
	return {
		"id": "",
		"difficulty": "easy",
		"region": "top-left",
		"player": "B",
		"goal": "capture",
		"target": [],
		"allowed_moves": [],
		"description": "",
		"setup": {"B": [], "W": []},
		"tree": {},
	}


def _board_from_setup(data):
	b = bd.Board()
	setup = data.get("setup", {}) or {}
	for letter in ("B", "W"):
		for coord in setup.get(letter, []) or []:
			try:
				gr, gc = reg.parse_global(coord)
				b.place(gr, gc, bd.color_from_letter(letter))
			except Exception:
				pass
	return b


class _ProblemView:
	__slots__ = ("player", "region", "goal", "target", "allowed_moves")


def _problem_view(data):
	p = _ProblemView()
	p.player = bd.color_from_letter(data["player"])
	p.region = data["region"]
	p.goal = data.get("goal", "capture")
	p.target = list(data.get("target", []))
	p.allowed_moves = list(data.get("allowed_moves", []))
	return p


def _rebuild_explore(data, state):
	"""Replay the explore path from setup, dropping any path entries that no
	longer line up with the tree (e.g. after region change)."""
	state["explore_board"] = _board_from_setup(data)
	state["explore_node"] = {"branches": data.get("tree", {})}
	state["explore_last_move"] = None
	player_color = bd.color_from_letter(data["player"])
	opp = bd.opponent(player_color)
	valid = []
	for move in list(state["explore_path"]):
		branches = state["explore_node"].get("branches") or {}
		if move not in branches:
			break
		try:
			r, c = reg.parse_global(move)
		except Exception:
			break
		ok, _ = state["explore_board"].play(r, c, player_color)
		if not ok:
			break
		state["explore_last_move"] = move
		entry = branches[move]
		reply = entry.get("reply")
		if reply:
			try:
				rr, rc = reg.parse_global(reply)
				ok2, _ = state["explore_board"].play(rr, rc, opp)
				if ok2:
					state["explore_last_move"] = reply
			except Exception:
				pass
		state["explore_node"] = entry
		valid.append(move)
	state["explore_path"] = valid


def _explore_step(data, state, coord_global, log):
	from . import solver

	player_color = bd.color_from_letter(data["player"])
	opp = bd.opponent(player_color)
	branches = state["explore_node"].setdefault("branches", {})

	if coord_global not in branches:
		test = state["explore_board"].copy()
		try:
			gr, gc = reg.parse_global(coord_global)
		except Exception:
			log(f"{coord_global}: bad coord.", BAD)
			return
		ok, _ = test.play(gr, gc, player_color)
		if not ok:
			log(f"{coord_global}: illegal move.", BAD)
			return

		P = _problem_view(data)
		target_globals = [reg.parse_global(t) for t in data.get("target", [])]
		score, reply_move = solver.alphabeta(
			test, opp, P, target_globals, 6,
			-solver.INF, solver.INF,
		)
		correct = (score == 1)
		reply_global = None
		if reply_move is not None:
			cand = reg.format_global(*reply_move)
			tcheck = test.copy()
			try:
				rr, rc = reg.parse_global(cand)
				ok2, _ = tcheck.play(rr, rc, opp)
				if ok2:
					reply_global = cand
			except Exception:
				reply_global = None

		branches[coord_global] = {
			"correct": bool(correct),
			"comment": "Auto-judged by solver (edit to refine).",
			"reply": reply_global,
			"branches": {},
		}
		verdict = "correct" if correct else "incorrect"
		log(f"{coord_global}: {verdict} (solver score={score}).",
			GOOD if correct else BAD)
		if reply_global:
			log(f"Opponent reply: {reply_global}.", MUTED)

	state["explore_path"].append(coord_global)
	_rebuild_explore(data, state)


def _safe_filename(name):
	out = []
	for ch in name:
		if ch.isalnum() or ch in "-_.":
			out.append(ch)
		else:
			out.append("_")
	return "".join(out) or "untitled"


def editor_screen(screen, clock, problem=None):
	if problem is None:
		data = _new_problem_data()
		source_path = None
	else:
		data = json.loads(json.dumps(problem.data))
		data.setdefault("setup", {"B": [], "W": []})
		data["setup"].setdefault("B", [])
		data["setup"].setdefault("W", [])
		data.setdefault("target", [])
		data.setdefault("allowed_moves", [])
		data.setdefault("tree", {})
		data.setdefault("description", "")
		data.setdefault("goal", "capture")
		source_path = problem.source_path

	state = {
		"tool": "setup_B",
		"explore_board": None,
		"explore_node": None,
		"explore_path": [],
		"explore_last_move": None,
		"messages": [],
		"exit": None,
		"source_path": source_path,
	}

	def log(text, color=TEXT):
		state["messages"].append((text, color))
		if len(state["messages"]) > 14:
			del state["messages"][:len(state["messages"]) - 14]

	sx = SIDEBAR_X + 16
	sidebar_w = WINDOW_W - sx - 16

	id_input = TextInput((sx + 60, 18, 220, 26), value=data.get("id", ""),
		placeholder="problem-id (e.g. easy-006)")
	desc_input = TextInput((sx, 200, sidebar_w, 70), value=data.get("description", ""),
		placeholder="description shown to solver", multiline=True)
	inputs = [id_input, desc_input]

	def enter_explore():
		state["explore_board"] = _board_from_setup(data)
		state["explore_node"] = {"branches": data.get("tree", {})}
		state["explore_path"] = []
		state["explore_last_move"] = None

	def cycle_difficulty():
		opts = list(database.DIFFICULTIES)
		i = opts.index(data["difficulty"]) if data["difficulty"] in opts else 0
		data["difficulty"] = opts[(i + 1) % len(opts)]

	def cycle_region():
		opts = list(reg.REGIONS.keys())
		i = opts.index(data["region"]) if data["region"] in opts else 0
		data["region"] = opts[(i + 1) % len(opts)]
		if state["tool"] == "explore":
			enter_explore()

	def cycle_player():
		data["player"] = "W" if data["player"] == "B" else "B"
		if state["tool"] == "explore":
			enter_explore()

	def cycle_goal():
		data["goal"] = "live" if data.get("goal") == "capture" else "capture"
		if state["tool"] == "explore":
			enter_explore()

	def select_tool(name):
		state["tool"] = name
		if name == "explore":
			enter_explore()
			log("Explore mode: clicks ask solver to judge and reply.", MUTED)
		else:
			log(f"Tool: {name}.", MUTED)

	def clear_tree_action():
		data["tree"] = {}
		if state["tool"] == "explore":
			enter_explore()
		log("Tree cleared.", MUTED)

	def reset_explore_action():
		if state["tool"] != "explore":
			select_tool("explore")
		else:
			enter_explore()
			log("Explore reset to setup.", MUTED)

	def undo_action():
		if state["tool"] != "explore" or not state["explore_path"]:
			log("Nothing to undo.", MUTED)
			return
		state["explore_path"].pop()
		_rebuild_explore(data, state)
		log("Undid last move.", MUTED)

	def save_action():
		data["id"] = id_input.value.strip()
		data["description"] = desc_input.value
		if not data["id"]:
			log("Need a non-empty id.", BAD)
			return
		target_dir = database.problems_dir() / data["difficulty"]
		target_dir.mkdir(parents=True, exist_ok=True)
		out_path = target_dir / f"{_safe_filename(data['id'])}.json"
		try:
			with out_path.open("w") as f:
				json.dump(data, f, indent="\t", ensure_ascii=False)
				f.write("\n")
		except Exception as e:
			log(f"Save failed: {e}", BAD)
			return
		state["source_path"] = out_path
		try:
			rel = out_path.relative_to(database.root_dir())
		except ValueError:
			rel = out_path
		log(f"Saved {rel}.", GOOD)

	def back_action():
		state["exit"] = "back"

	def quit_action():
		state["exit"] = "quit"

	meta_y = 56
	diff_btn = Button("", (sx + 60, meta_y, 100, 24), cycle_difficulty)
	region_btn = Button("", (sx + 215, meta_y, 135, 24), cycle_region)
	player_btn = Button("", (sx + 60, meta_y + 32, 60, 24), cycle_player)
	goal_btn = Button("", (sx + 215, meta_y + 32, 100, 24), cycle_goal)
	meta_buttons = [diff_btn, region_btn, player_btn, goal_btn]

	tools_list = [
		("B", "setup_B", 38),
		("W", "setup_W", 38),
		("Erase", "erase", 58),
		("Target", "target", 60),
		("Allowed", "allowed", 70),
		("Explore", "explore", 70),
	]
	tool_y = 282
	tool_buttons = []
	bx = sx
	for label, name, w in tools_list:
		btn = Button(label, (bx, tool_y, w, 28), lambda n=name: select_tool(n))
		tool_buttons.append((btn, name))
		bx += w + 4

	bottom_y = WINDOW_H - 50
	bx = sx
	save_btn = Button("Save", (bx, bottom_y, 70, 36), save_action); bx += 74
	clear_btn = Button("Clear Tree", (bx, bottom_y, 95, 36), clear_tree_action); bx += 99
	reset_btn = Button("Reset", (bx, bottom_y, 60, 36), reset_explore_action); bx += 64
	undo_btn = Button("Undo", (bx, bottom_y, 60, 36), undo_action); bx += 64
	back_btn = Button("Back", (bx, bottom_y, 60, 36), back_action)
	bottom_buttons = [save_btn, clear_btn, reset_btn, undo_btn, back_btn]

	log("Click the board to place setup stones, then switch tools.", MUTED)

	while state["exit"] is None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return None
			any_input_focused = any(i.focused for i in inputs)
			if event.type == pygame.KEYDOWN and not any_input_focused:
				if event.key == pygame.K_ESCAPE:
					return "back"
				key_map = {
					pygame.K_1: "setup_B",
					pygame.K_2: "setup_W",
					pygame.K_3: "erase",
					pygame.K_4: "target",
					pygame.K_5: "allowed",
					pygame.K_6: "explore",
				}
				if event.key in key_map:
					select_tool(key_map[event.key])
				elif event.key == pygame.K_u:
					undo_action()
				elif event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_META or pygame.key.get_mods() & pygame.KMOD_CTRL):
					save_action()

			for inp in inputs:
				inp.handle(event)
			for btn in meta_buttons + [b for b, _ in tool_buttons] + bottom_buttons:
				btn.handle(event)

			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not any_input_focused:
				hit = screen_to_view(*event.pos)
				if hit is None:
					continue
				vr, vc = hit
				(off_r, off_c), _ = reg.region_bounds(data["region"])
				gr, gc = off_r + vr, off_c + vc
				coord_global = reg.format_global(gr, gc)
				_handle_editor_click(data, state, coord_global, log)

		diff_btn.label = data["difficulty"]
		region_btn.label = data["region"]
		player_btn.label = data["player"]
		goal_btn.label = data.get("goal", "capture")
		for btn, name in tool_buttons:
			marker = "* " if state["tool"] == name else "  "
			btn.label = marker + {n: l for l, n, _ in tools_list}[name]

		screen.fill(SIDEBAR_BG)
		pygame.draw.rect(screen, BG, (0, 0, SIDEBAR_X, WINDOW_H))

		if state["tool"] == "explore":
			board_to_show = state["explore_board"] or _board_from_setup(data)
		else:
			board_to_show = _board_from_setup(data)

		hover_xy = None
		hover_color = None
		candidates = None
		last_move = None
		mouse_pos = pygame.mouse.get_pos()
		if state["tool"] == "setup_B":
			hover_xy = screen_to_view(*mouse_pos)
			hover_color = bd.BLACK
		elif state["tool"] == "setup_W":
			hover_xy = screen_to_view(*mouse_pos)
			hover_color = bd.WHITE
		elif state["tool"] == "explore":
			hover_xy = screen_to_view(*mouse_pos)
			hover_color = bd.color_from_letter(data["player"])
			node = state["explore_node"]
			candidates = list((node.get("branches") or {}).keys()) if node else None
			last_move = state["explore_last_move"]
		elif state["tool"] == "target":
			candidates = list(data.get("target", []))
		elif state["tool"] == "allowed":
			candidates = list(data.get("allowed_moves", []))

		draw_board(screen, board_to_show, data["region"],
			last_move=last_move,
			candidates=candidates,
			hover_xy=hover_xy,
			hover_color=hover_color,
		)

		render_text(screen, "ID:", (sx, 22), size=14, bold=True)
		id_input.draw(screen)
		render_text(screen, "Diff:", (sx, meta_y + 4), size=13, bold=True)
		diff_btn.draw(screen)
		render_text(screen, "Region:", (sx + 165, meta_y + 4), size=13, bold=True)
		region_btn.draw(screen)
		render_text(screen, "Player:", (sx, meta_y + 36), size=13, bold=True)
		player_btn.draw(screen)
		render_text(screen, "Goal:", (sx + 165, meta_y + 36), size=13, bold=True)
		goal_btn.draw(screen)

		render_text(screen, "Description:", (sx, 178), size=13, bold=True)
		desc_input.draw(screen)

		render_text(screen, "Tools (1-6):", (sx, tool_y - 18), size=13, bold=True)
		for btn, _ in tool_buttons:
			btn.draw(screen)

		info_y = tool_y + 40
		render_text(screen, f"Active: {state['tool']}", (sx, info_y), size=12, color=MUTED)
		info_y += 18
		if state["tool"] == "target":
			tgt = ", ".join(data.get("target", [])) or "(none)"
			for line in wrap_text(f"Target: {tgt}", sidebar_w, 12):
				render_text(screen, line, (sx, info_y), size=12)
				info_y += 16
		if state["tool"] == "allowed":
			al = ", ".join(data.get("allowed_moves", [])) or "(none — full eye-space)"
			for line in wrap_text(f"Allowed: {al}", sidebar_w, 12):
				render_text(screen, line, (sx, info_y), size=12)
				info_y += 16
		if state["tool"] == "explore":
			path_str = " ".join(state["explore_path"]) or "(at root)"
			for line in wrap_text(f"Path: {path_str}", sidebar_w, 12):
				render_text(screen, line, (sx, info_y), size=12)
				info_y += 16
			node = state["explore_node"]
			if node and "correct" in node:
				verdict = "correct" if node.get("correct") else "incorrect"
				color = GOOD if node.get("correct") else BAD
				render_text(screen, f"Node: {verdict}", (sx, info_y), size=12, color=color)
				info_y += 16

		render_text(screen, "Log:", (sx, info_y + 4), size=13, bold=True)
		info_y += 26
		log_bottom = bottom_y - 6
		for text, color in state["messages"]:
			if info_y >= log_bottom:
				break
			for line in wrap_text(text, sidebar_w, 12):
				if info_y >= log_bottom:
					break
				render_text(screen, line, (sx, info_y), size=12, color=color)
				info_y += 16

		for btn in bottom_buttons:
			btn.draw(screen)
		pygame.display.flip()
		clock.tick(60)

	return state["exit"]


def _handle_editor_click(data, state, coord_global, log):
	tool = state["tool"]
	if tool in ("setup_B", "setup_W", "erase"):
		setup = data.setdefault("setup", {"B": [], "W": []})
		b_list = setup.setdefault("B", [])
		w_list = setup.setdefault("W", [])
		in_b = coord_global in b_list
		in_w = coord_global in w_list
		if tool == "setup_B":
			if in_b:
				b_list.remove(coord_global)
			else:
				if in_w:
					w_list.remove(coord_global)
				b_list.append(coord_global)
		elif tool == "setup_W":
			if in_w:
				w_list.remove(coord_global)
			else:
				if in_b:
					b_list.remove(coord_global)
				w_list.append(coord_global)
		else:
			if in_b:
				b_list.remove(coord_global)
			if in_w:
				w_list.remove(coord_global)
		return
	if tool == "target":
		tgt = data.setdefault("target", [])
		if coord_global in tgt:
			tgt.remove(coord_global)
		else:
			tgt.append(coord_global)
		return
	if tool == "allowed":
		al = data.setdefault("allowed_moves", [])
		if coord_global in al:
			al.remove(coord_global)
		else:
			al.append(coord_global)
		return
	if tool == "explore":
		_explore_step(data, state, coord_global, log)


def main(args):
	start_mode = "play"
	start_problem = None
	if args:
		if args[0] in ("-h", "--help"):
			print("usage: python -m src.main play-ui [--new | --edit <id> | <id>]")
			return
		if args[0] == "--new":
			start_mode = "edit"
			start_problem = None
		elif args[0] == "--edit" and len(args) >= 2:
			start_problem = database.find_problem(args[1])
			if start_problem is None:
				print(f"problem id '{args[1]}' not found.")
				return
			start_mode = "edit"
		else:
			start_problem = database.find_problem(args[0])
			if start_problem is None:
				print(f"problem id '{args[0]}' not found.")
				return
			start_mode = "play"

	pygame.init()
	screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
	pygame.display.set_caption("DeepGoProblem")
	clock = pygame.time.Clock()
	try:
		pending = (start_mode, start_problem) if (start_problem is not None or start_mode == "edit") else None
		while True:
			if pending is None:
				pending = picker_screen(screen, clock)
				if pending is None:
					return
			mode, problem = pending
			pending = None
			if mode == "play":
				result = play_screen(screen, clock, problem)
				if result == "edit":
					pending = ("edit", problem)
					continue
				if result in (None, "quit"):
					return
			elif mode == "edit":
				result = editor_screen(screen, clock, problem)
				if result in (None, "quit"):
					return
			elif mode == "new":
				result = editor_screen(screen, clock, None)
				if result in (None, "quit"):
					return
			# anything else (e.g. "back") falls through to picker
	finally:
		pygame.quit()
