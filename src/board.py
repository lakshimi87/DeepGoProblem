"""19x19 go board with capture, suicide, and simple ko enforcement."""

EMPTY = 0
BLACK = 1
WHITE = 2

SIZE = 19


def opponent(color):
	return WHITE if color == BLACK else BLACK


def color_from_letter(letter):
	letter = letter.upper()
	if letter == "B":
		return BLACK
	if letter == "W":
		return WHITE
	raise ValueError(f"bad color letter: {letter}")


def letter_from_color(color):
	return "B" if color == BLACK else "W"


class Board:
	def __init__(self, size=SIZE):
		self.size = size
		self.grid = [[EMPTY] * size for _ in range(size)]
		self.ko = None
		self.captures = {BLACK: 0, WHITE: 0}

	def copy(self):
		b = Board(self.size)
		b.grid = [row[:] for row in self.grid]
		b.ko = self.ko
		b.captures = dict(self.captures)
		return b

	def at(self, r, c):
		return self.grid[r][c]

	def place(self, r, c, color):
		self.grid[r][c] = color

	def neighbors(self, r, c):
		for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
			nr, nc = r + dr, c + dc
			if 0 <= nr < self.size and 0 <= nc < self.size:
				yield nr, nc

	def group(self, r, c):
		"""Return (stones, liberties) for the group at (r, c). Empty -> (set(), set())."""
		color = self.grid[r][c]
		if color == EMPTY:
			return set(), set()
		stones = set()
		libs = set()
		stack = [(r, c)]
		while stack:
			sr, sc = stack.pop()
			if (sr, sc) in stones:
				continue
			stones.add((sr, sc))
			for nr, nc in self.neighbors(sr, sc):
				v = self.grid[nr][nc]
				if v == EMPTY:
					libs.add((nr, nc))
				elif v == color and (nr, nc) not in stones:
					stack.append((nr, nc))
		return stones, libs

	def play(self, r, c, color):
		"""Attempt to play a stone. Returns (ok, captured_count)."""
		if self.grid[r][c] != EMPTY:
			return False, 0
		if self.ko == (r, c):
			return False, 0
		self.grid[r][c] = color
		opp = opponent(color)
		captured = []
		for nr, nc in self.neighbors(r, c):
			if self.grid[nr][nc] == opp:
				stones, libs = self.group(nr, nc)
				if not libs:
					for sr, sc in stones:
						self.grid[sr][sc] = EMPTY
					captured.extend(stones)
		own_stones, own_libs = self.group(r, c)
		if not own_libs:
			self.grid[r][c] = EMPTY
			for sr, sc in captured:
				self.grid[sr][sc] = opp
			return False, 0
		if len(captured) == 1 and len(own_stones) == 1:
			self.ko = captured[0]
		else:
			self.ko = None
		self.captures[color] += len(captured)
		return True, len(captured)
