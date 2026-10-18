"""03 - Karpathy's policy gradient, the method of the original notebook, step by step.

1. checks that autograd on ``-A log pi(a|x)`` gives exactly Karpathy's hand-written gradient
   ``A * (y - p)`` (the "fake label" trick);
2. trains the 2-layer MLP on frame differences for 150k steps (~1.5 min). This is the slowest
   learner of the repo per sample (no baseline, no CNN, Monte Carlo returns): expect the
   training score to creep up from -3.5, not to win yet. The full run (2M steps) is in the README.

    python examples/03_karpathy_policy_gradient.py
"""
import torch

from pong_rl.algorithms.pg import pg_loss
from pong_rl.config import make_config
from pong_rl.policy import DOWN, UP
from pong_rl.train import run

# 1. the gradient identity
logits = torch.randn(5, requires_grad=True)
actions = torch.tensor([UP, DOWN, UP, DOWN, UP])
adv = torch.randn(5)
pg_loss(logits, actions, adv).backward()
y = (actions == UP).float()
print("autograd  -dL/dlogit :", (-logits.grad).numpy().round(4))
print("Karpathy  A (y - p)  :", (adv * (y - torch.sigmoid(logits))).detach().numpy().round(4))

# 2. a short run
cfg = make_config("pg", steps=150_000, eval_every=50_000, final_eval_episodes=20, out="examples/output/03_pg")
res = run(cfg)
print(f"pg after {cfg.steps:,} steps: score {res['mean']:+.2f}")
