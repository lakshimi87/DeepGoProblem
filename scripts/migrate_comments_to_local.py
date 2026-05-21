"""One-shot: rewrite coord-like substrings in comments/descriptions from absolute
19x19 Go coords back to the region's current local view labels. Run after the
data is in global form. Coords that fall outside the region's view are left
as-is.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import region as reg  # noqa: E402


COORD_RE = re.compile(r"\b([A-HJ-T])(\d{1,2})\b")


def convert_text(s, region):
	def repl(m):
		coord = m.group(0)
		row = int(m.group(2))
		if not (1 <= row <= reg.BOARD):
			return coord
		vw = reg.global_to_view(coord, region)
		if vw is None:
			return coord
		return reg.format_local(*vw)
	return COORD_RE.sub(repl, s)


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
