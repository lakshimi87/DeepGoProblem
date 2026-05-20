"""Discover problem files under problems/ grouped by difficulty."""

from pathlib import Path

from .problem import Problem


DIFFICULTIES = ("easy", "medium", "hard")


def root_dir():
	return Path(__file__).resolve().parent.parent


def problems_dir():
	return root_dir() / "problems"


def list_problems(difficulty=None):
	base = problems_dir()
	folders = [difficulty] if difficulty else DIFFICULTIES
	out = []
	for d in folders:
		folder = base / d
		if not folder.is_dir():
			continue
		for p in sorted(folder.glob("*.json")):
			try:
				out.append(Problem.load(p))
			except Exception as e:
				print(f"warn: could not load {p}: {e}")
	out.sort(key=lambda x: (DIFFICULTIES.index(x.difficulty) if x.difficulty in DIFFICULTIES else 99, x.id))
	return out


def find_problem(problem_id):
	for p in list_problems():
		if p.id == problem_id:
			return p
	return None
