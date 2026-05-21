"""One-shot follow-up: rewrite coord-like substrings in comments and descriptions
from old per-region local notation (A1..J9) to absolute 19x19 Go coords
(A1..T19, I skipped). Run once after migrate_to_global_coords.py.

Only matches `\\b[A-HJ][1-9]\\b` (single letter + single digit), so the
already-migrated multi-digit tree keys ('A17', 'B19', ...) are not touched.
"""

import json
import re
from pathlib import Path


OLD_VIEW = 9
OLD_LETTERS = "ABCDEFGHJ"
OLD_REGIONS = {
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

BOARD = 19
BOARD_LETTERS = "ABCDEFGHJKLMNOPQRST"

COORD_RE = re.compile(r"\b([A-HJ])([1-9])\b")


def to_global_str(coord, region):
	col_letter = coord[0].upper()
	row_num = int(coord[1:])
	off_r, off_c = OLD_REGIONS[region]
	gr = off_r + (OLD_VIEW - row_num)
	gc = off_c + OLD_LETTERS.index(col_letter)
	return f"{BOARD_LETTERS[gc]}{BOARD - gr}"


def convert_text(s, region):
	return COORD_RE.sub(lambda m: to_global_str(m.group(0), region), s)


def walk_tree(tree, region):
	for v in tree.values():
		if "comment" in v and isinstance(v["comment"], str):
			v["comment"] = convert_text(v["comment"], region)
		sub = v.get("branches") or {}
		if sub:
			walk_tree(sub, region)


def migrate_file(path):
	data = json.loads(path.read_text())
	region = data["region"]
	if "description" in data:
		data["description"] = convert_text(data["description"], region)
	if "tree" in data:
		walk_tree(data["tree"], region)
	path.write_text(json.dumps(data, indent="\t", ensure_ascii=False) + "\n")
	print(f"migrated: {path}")


def main():
	root = Path(__file__).resolve().parent.parent / "problems"
	for p in sorted(root.rglob("*.json")):
		migrate_file(p)


if __name__ == "__main__":
	main()
