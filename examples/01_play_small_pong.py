"""01 - Meet the small Pong: random play, a hand-written tracker, and what the agent sees.

No learning yet. This shows the two numbers every method has to beat (a random agent and a
scripted "follow the ball" player), the preprocessing figure (raw frame -> 20x20 binary ->
frame difference / frame stack) and a GIF of the scripted player.

    python examples/01_play_small_pong.py            # ~20 s
Outputs: examples/output/01_preprocessing.png, examples/output/01_tracker.gif
"""
import os

import numpy as np

from pong_rl.envs import MiniPong
from pong_rl.envs.mini_pong import DOWN, NOOP, UP
from pong_rl.viz import plot_preprocessing, save_gif

OUT = "examples/output"
os.makedirs(OUT, exist_ok=True)


def tracker(env: MiniPong) -> int:
    """Move the paddle centre towards the ball: the policy every RL agent here has to discover."""
    centre, ball = env.agent_y + env.ph / 2, env.by + 1
    return UP if ball < centre - 1 else DOWN if ball > centre + 1 else NOOP


def play(policy, games=30, gif=None):
    env, scores, frames = MiniPong(), [], []
    for g in range(games):
        env.reset(seed=10_000 + g)
        done, score = False, 0.0
        while not done:
            if gif and g == 0:
                frames.append(env.render_rgb(8))
            _, r, term, trunc, _ = env.step(policy(env))
            score, done = score + r, term or trunc
        scores.append(score)
    if gif:
        save_gif(frames[:600], gif)
    return np.mean(scores), np.std(scores)


rng = np.random.default_rng(0)
print("score = points won - points lost, first to 5 (range -5 .. +5), 30 games each")
for name, policy, gif in [("random", lambda e: int(rng.integers(3)), None), ("NOOP", lambda e: NOOP, None),
                          ("tracker", tracker, f"{OUT}/01_tracker.gif")]:
    mean, std = play(policy, gif=gif)
    print(f"{name:8s}: {mean:+.2f} +- {std:.2f}")
plot_preprocessing(f"{OUT}/01_preprocessing.png", env="mini")
print(f"wrote {OUT}/01_preprocessing.png and {OUT}/01_tracker.gif")
