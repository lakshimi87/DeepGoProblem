"""Nine-region geometry for a 19x19 board, and local <-> global coordinates.

Local coordinates use Go notation A1..J9 (no I). Row 1 is the bottom of the
9x9 view, row 9 is the top, A is leftmost, J is rightmost.
"""

LETTERS = "ABCDEFGHJ"
VIEW = 9
BOARD = 19

REGIONS = {
	"top-left":      (0, 0),
	"top-center":    (0, 5),
	"top-right":     (0, 10),
	"middle-left":   (5, 0),
	"center":        (5, 5),
	"middle-right":  (5, 10),
	"bottom-left":   (10, 0),
	"bottom-center": (10, 5),
	"bottom-right":  (10, 10),
}


def parse_local(coord):
	"""'C5' -> matrix (row, col) inside the 9x9 view."""
	coord = coord.strip().upper()
	if len(coord) < 2:
		raise ValueError(f"bad coord: {coord!r}")
	col_letter = coord[0]
	row_num = int(coord[1:])
	if col_letter not in LETTERS:
		raise ValueError(f"bad column letter: {col_letter}")
	if not (1 <= row_num <= VIEW):
		raise ValueError(f"row out of range: {row_num}")
	return VIEW - row_num, LETTERS.index(col_letter)


def format_local(row, col):
	return f"{LETTERS[col]}{VIEW - row}"


def local_to_global(coord, region):
	r, c = parse_local(coord)
	off_r, off_c = REGIONS[region]
	return off_r + r, off_c + c


def region_bounds(region):
	"""Inclusive ((row_lo, col_lo), (row_hi, col_hi)) on the 19x19 board."""
	off_r, off_c = REGIONS[region]
	return (off_r, off_c), (off_r + VIEW - 1, off_c + VIEW - 1)


def edges(region):
	"""Which board edges (top/bottom/left/right) fall inside this region's view."""
	(r_lo, c_lo), (r_hi, c_hi) = region_bounds(region)
	out = set()
	if r_lo == 0:
		out.add("top")
	if r_hi == BOARD - 1:
		out.add("bottom")
	if c_lo == 0:
		out.add("left")
	if c_hi == BOARD - 1:
		out.add("right")
	return out
