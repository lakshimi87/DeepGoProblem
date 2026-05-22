"""Supervised training: predict the correct first move in each problem's tree."""

from pathlib import Path

from . import database


def main(args):
	try:
		import torch
		import torch.nn as nn
		import torch.optim as optim
	except ImportError:
		print("torch is required. run ./setup.sh --train")
		return

	from . import neural

	epochs = 100
	if "--epochs" in args:
		epochs = int(args[args.index("--epochs") + 1])
	reset = "--reset" in args

	problems = database.list_problems()
	X, Y = [], []
	for p in problems:
		correct = [m for m, e in p.tree.items() if e.get("correct")]
		if not correct:
			continue
		t = neural.board_to_tensor(p.initial_board(), p.region, p.player)
		X.append(t)
		Y.append(neural.move_to_index(correct[0], p.region))

	if not X:
		print("no problems with a labelled correct first move. nothing to train.")
		return

	X = torch.stack(X)
	Y = torch.tensor(Y, dtype=torch.long)

	out = Path(__file__).resolve().parent.parent / "models" / "policy.pt"

	model = neural.build_model()
	if not reset and out.exists():
		model.load_state_dict(torch.load(out, map_location="cpu"))
		print(f"resuming from {out}")
	else:
		print("starting from a freshly initialised model")
	opt = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.CrossEntropyLoss()

	print(f"training on {len(X)} samples for {epochs} epochs")
	model.train()
	for epoch in range(epochs):
		opt.zero_grad()
		logits = model(X)
		loss = loss_fn(logits, Y)
		loss.backward()
		opt.step()
		if (epoch + 1) % max(1, epochs // 10) == 0:
			acc = (logits.argmax(-1) == Y).float().mean().item()
			print(f"  epoch {epoch+1:4d}   loss={loss.item():.4f}   acc={acc*100:5.1f}%")

	out.parent.mkdir(exist_ok=True)
	torch.save(model.state_dict(), out)
	print(f"saved {out}")
