"""02 - Train an agent from pixels in about 3 minutes: A2C on the small Pong.

A2C is the quickest learner on one CPU thread here: 16 games in parallel, 5-step rollouts,
one gradient step per rollout. The score goes from about -3.5 (random) to a winning policy
(> 0) in ~150k steps. Checkpoint, log and curve land in examples/output/02_a2c/.

    python examples/02_train_a2c.py                  # ~3 min on one thread
"""
from pong_rl.config import make_config
from pong_rl.train import run

cfg = make_config("a2c", steps=150_000, eval_every=25_000, eval_episodes=10, final_eval_episodes=20,
                  out="examples/output/02_a2c")
res = run(cfg)
print(f"final score {res['mean']:+.2f} (random play scores about -3.5), see {cfg.out}/curve.png")
