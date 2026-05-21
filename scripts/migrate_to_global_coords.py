"""One-shot migration: rewrite problem JSONs from per-region local coords
to absolute 19x19 Go coords (A1..T19, I skipped). Run once, then delete.

Converts: setup values, target list, tree keys, and tree 'reply' values.
Leaves 'comment' / 'description' text untouched.
"""

import json
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


def old_local_to_global(coord, region):
	col_letter = coord[0].upper()
	row_num = int(coord[1:])
	off_r, off_c = OLD_REGIONS[region]
	gr = off_r + (OLD_VIEW - row_num)
	gc = off_c + OLD_LETTERS.index(col_letter)
	return gr, gc


def to_global_str(coord, region):
	gr, gc = old_local_to_global(coord, region)
	return f"{BOARD_LETTERS[gc]}{BOARD - gr}"


def migrate_tree(tree, region):
	out = {}
	for k, v in tree.items():
		new_k = to_global_str(k, region)
		new_v = dict(v)
		reply = v.get("reply")
		if reply:
			new_v["reply"] = to_global_str(reply, region)
		sub = v.get("branches") or {}
		if sub:
			new_v["branches"] = migrate_tree(sub, region)
		out[new_k] = new_v
	return out


def migrate_file(path):
	data = json.loads(path.read_text())
	region = data["region"]
	if "setup" in data:
		data["setup"] = {
			letter: [to_global_str(c, region) for c in coords]
			for letter, coords in data["setup"].items()
		}
	if "target" in data:
		data["target"] = [to_global_str(c, region) for c in data["target"]]
	if "tree" in data:
		data["tree"] = migrate_tree(data["tree"], region)
	path.write_text(json.dumps(data, indent="\t", ensure_ascii=False) + "\n")
	print(f"migrated: {path}")


def main():
	root = Path(__file__).resolve().parent.parent / "problems"
	for p in sorted(root.rglob("*.json")):
		migrate_file(p)


if __name__ == "__main__":
	main()
