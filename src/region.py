"""Nine-region geometry for a 19x19 board, and local <-> global coordinates.

Local coordinates use Go notation A1..L11 (no I). Row 1 is the bottom of the
11x11 view, row 11 is the top, A is leftmost, L is rightmost.
"""

LETTERS = "ABCDEFGHJKL"
BOARD_LETTERS = "ABCDEFGHJKLMNOPQRST"
VIEW = 11
BOARD = 19

REGIONS = {
	"top-left":      (0, 0),
	"top-center":    (0, 4),
	"top-right":     (0, 8),
	"middle-left":   (4, 0),
	"center":        (4, 4),
	"middle-right":  (4, 8),
	"bottom-left":   (8, 0),
	"bottom-center": (8, 4),
	"bottom-right":  (8, 8),
}


def parse_local(coord):
	"""'C5' -> matrix (row, col) inside the 11x11 view."""
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


def parse_global(coord):
	"""Standard 19x19 Go coord ('A1'..'T19', I skipped) -> matrix (row, col)."""
	coord = coord.strip().upper()
	if len(coord) < 2:
		raise ValueError(f"bad coord: {coord!r}")
	col_letter = coord[0]
	row_num = int(coord[1:])
	if col_letter not in BOARD_LETTERS:
		raise ValueError(f"bad column letter: {col_letter}")
	if not (1 <= row_num <= BOARD):
		raise ValueError(f"row out of range: {row_num}")
	return BOARD - row_num, BOARD_LETTERS.index(col_letter)


def format_global(row, col):
	return f"{BOARD_LETTERS[col]}{BOARD - row}"


def global_to_view(coord, region):
	"""Global Go coord -> (view_r, view_c) inside the region's view, or None if outside."""
	gr, gc = parse_global(coord)
	off_r, off_c = REGIONS[region]
	vr, vc = gr - off_r, gc - off_c
	if 0 <= vr < VIEW and 0 <= vc < VIEW:
		return vr, vc
	return None


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
