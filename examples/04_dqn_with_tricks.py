"""04 - DQN with all its upgrades switched on: double, dueling, prioritized replay, 3-step returns.

The same training loop as ``pong-train --algo dueling-dqn --per --n-step 3``, run for 40k steps
(~2.5 min). Prints the TD loss and mean Q value as they move, then the score.
DQN is the most sample-efficient method here: it reuses every transition ~8 times from replay.

    python examples/04_dqn_with_tricks.py
"""
from pong_rl.config import make_config
from pong_rl.metrics import load_log
from pong_rl.train import run

cfg = make_config("dueling-dqn", per=True, n_step=3, steps=40_000, eval_every=20_000, learning_starts=2_000,
                  final_eval_episodes=20, out="examples/output/04_dqn")
res = run(cfg)
for row in load_log(cfg.out)["losses"][::4]:
    print(f"step {row['step']:>6,d}  td loss {row['loss']:.4f}  mean Q {row['q_mean']:+.3f}  eps {row['eps']:.2f}")
print(f"score after {cfg.steps:,} steps: {res['mean']:+.2f}")
