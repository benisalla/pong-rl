"""02 - Train an agent from pixels in a few minutes: A2C on the small Pong.

A2C is the quickest learner on one CPU thread here: 16 games in parallel, 5-step rollouts,
one gradient step per rollout. The score goes from about -3.3 (random) to a winning policy
(> 0) after ~175k steps; the default budget is 400k (the learning rate decays to 0 over it).
Checkpoint, log and curve land in examples/output/02_a2c/.

    python examples/02_train_a2c.py                  # ~2 min on one thread (more on a loaded machine)
"""
from pong_rl.config import make_config
from pong_rl.train import run

cfg = make_config("a2c", steps=400_000, eval_every=25_000, eval_episodes=10, final_eval_episodes=20,
                  out="examples/output/02_a2c")
res = run(cfg)
print(f"final score {res['mean']:+.2f} (random play scores about -3.3), see {cfg.out}/curve.png")
