"""Policy + value CNN over the 11x11 region. Torch is imported lazily so importing this
module does not require torch to be installed.

The network has two heads:
  * policy: per-cell logits over the VIEW*VIEW action space (softmaxed at use sites)
  * value:  scalar in [-1, +1] from the side-to-move's perspective

`build_model()` returns a module whose forward(x) returns the tuple
`(policy_logits[B, VIEW*VIEW], value[B])`.
"""

from . import region as reg


def _torch():
	import torch
	import torch.nn as nn
	return torch, nn


def build_model(channels=32):
	torch, nn = _torch()

	class PolicyValueCNN(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
			self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
			self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
			self.policy_head = nn.Conv2d(channels, 1, 1)
			# Value head: 1x1 conv to compress, then two FC layers ending in tanh.
			self.value_conv = nn.Conv2d(channels, 8, 1)
			self.value_fc1 = nn.Linear(8 * reg.VIEW * reg.VIEW, 64)
			self.value_fc2 = nn.Linear(64, 1)
			self.act = nn.ReLU()

		def forward(self, x):
			x = self.act(self.conv1(x))
			x = self.act(self.conv2(x))
			x = self.act(self.conv3(x))

			p = self.policy_head(x)
			b, _, h, w = p.shape
			policy_logits = p.view(b, h * w)

			v = self.act(self.value_conv(x))
			v = v.view(b, -1)
			v = self.act(self.value_fc1(v))
			v = torch.tanh(self.value_fc2(v)).view(b)

			return policy_logits, v

	return PolicyValueCNN()


def board_to_tensor(board, region, player_color):
	"""Encode the 11x11 region as a (3, 11, 11) tensor: own / opp / board-edge mask."""
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


def move_to_index(coord, region):
	"""Global Go coord -> flat view-cell index for the CNN policy head."""
	vw = reg.global_to_view(coord, region)
	if vw is None:
		raise ValueError(f"coord {coord} outside region {region}")
	r, c = vw
	return r * reg.VIEW + c


def index_to_move(idx, region):
	"""Flat view-cell index -> global Go coord."""
	r, c = divmod(idx, reg.VIEW)
	off_r, off_c = reg.REGIONS[region]
	return reg.format_global(off_r + r, off_c + c)


def global_to_index(gr, gc, region):
	"""Global (row, col) -> flat view-cell index. Returns None if outside region."""
	off_r, off_c = reg.REGIONS[region]
	vr, vc = gr - off_r, gc - off_c
	if 0 <= vr < reg.VIEW and 0 <= vc < reg.VIEW:
		return vr * reg.VIEW + vc
	return None
