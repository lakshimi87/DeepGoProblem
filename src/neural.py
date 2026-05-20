"""Policy CNN over the 9x9 region. Torch is imported lazily so importing this
module does not require torch to be installed.
"""

from . import region as reg


def _torch():
	import torch
	import torch.nn as nn
	return torch, nn


def build_model(channels=32):
	torch, nn = _torch()

	class PolicyCNN(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
			self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
			self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
			self.head = nn.Conv2d(channels, 1, 1)
			self.act = nn.ReLU()

		def forward(self, x):
			x = self.act(self.conv1(x))
			x = self.act(self.conv2(x))
			x = self.act(self.conv3(x))
			x = self.head(x)
			b, _, h, w = x.shape
			return x.view(b, h * w)

	return PolicyCNN()


def board_to_tensor(board, region, player_color):
	"""Encode the 9x9 region as a (3, 9, 9) tensor: own / opp / board-edge mask."""
	torch, _ = _torch()
	import numpy as np

	from . import board as bd

	(off_r, off_c), _ = reg.region_bounds(region)
	own = np.zeros((reg.VIEW, reg.VIEW), dtype="float32")
	opp = np.zeros((reg.VIEW, reg.VIEW), dtype="float32")
	edge = np.zeros((reg.VIEW, reg.VIEW), dtype="float32")
	es = reg.edges(region)
	if "top" in es:
		edge[0, :] = 1.0
	if "bottom" in es:
		edge[-1, :] = 1.0
	if "left" in es:
		edge[:, 0] = 1.0
	if "right" in es:
		edge[:, -1] = 1.0
	for r in range(reg.VIEW):
		for c in range(reg.VIEW):
			v = board.at(off_r + r, off_c + c)
			if v == player_color:
				own[r, c] = 1.0
			elif v == bd.opponent(player_color):
				opp[r, c] = 1.0
	return torch.from_numpy(np.stack([own, opp, edge], axis=0))


def move_to_index(coord):
	r, c = reg.parse_local(coord)
	return r * reg.VIEW + c


def index_to_move(idx):
	r, c = divmod(idx, reg.VIEW)
	return reg.format_local(r, c)
