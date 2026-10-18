"""05 - Load a checkpoint, score it on 30 fixed games and save a GIF of it playing.

Uses the A2C agent from example 02 (run it first) or any ``model.pt`` given as argument, e.g.
the full runs ``runs/mini/ppo/model.pt`` from ``scripts/compare_all.py``.

    python examples/05_evaluate_and_gif.py [path/to/model.pt]
"""
import sys

from pong_rl.evaluate import evaluate
from pong_rl.play import play
from pong_rl.policy import load_policy
from pong_rl.viz import save_gif

path = sys.argv[1] if len(sys.argv) > 1 else "examples/output/02_a2c/model.pt"
policy = load_policy(path)
res = evaluate(policy, episodes=30)
print(f"{policy.cfg.algo}: score {res['mean']:+.2f} +- {res['std']:.2f}, win rate {res['win_rate']:.0%}")
frames, scores = play(policy, games=1)
save_gif(frames[:900], "examples/output/05_agent.gif")
print(f"GIF of one game (score {scores[0]:+.0f}) -> examples/output/05_agent.gif")
