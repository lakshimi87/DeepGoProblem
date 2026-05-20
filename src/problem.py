"""Problem dataclass: json load/save and materialization onto a Board."""

import json
from pathlib import Path

from . import region as reg
from . import board as bd


class Problem:
	def __init__(self, data, source_path=None):
		self.data = data
		self.source_path = source_path

	@property
	def id(self):
		return self.data["id"]

	@property
	def difficulty(self):
		return self.data["difficulty"]

	@property
	def region(self):
		return self.data["region"]

	@property
	def player(self):
		return bd.color_from_letter(self.data["player"])

	@property
	def description(self):
		return self.data.get("description", "")

	@property
	def goal(self):
		return self.data.get("goal", "")

	@property
	def target(self):
		return self.data.get("target", [])

	@property
	def tree(self):
		return self.data.get("tree", {})

	def initial_board(self):
		b = bd.Board()
		setup = self.data.get("setup", {})
		for letter, coords in setup.items():
			color = bd.color_from_letter(letter)
			for coord in coords:
				gr, gc = reg.local_to_global(coord, self.region)
				b.place(gr, gc, color)
		return b

	@classmethod
	def load(cls, path):
		path = Path(path)
		with path.open() as f:
			data = json.load(f)
		return cls(data, source_path=path)

	def save(self, path=None):
		out = Path(path) if path else self.source_path
		if out is None:
			raise ValueError("no path to save to")
		with out.open("w") as f:
			json.dump(self.data, f, indent="\t", ensure_ascii=False)
			f.write("\n")
