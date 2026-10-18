"""``pong-play``: watch a trained agent, as a GIF (and the score of each game).

    pong-play runs/ppo/model.pt --gif assets/ppo_mini_pong.gif --games 1
"""
from __future__ import annotations

import argparse

import numpy as np

from .envs import make_env
from .evaluate import EVAL_SEED
from .policy import Policy, load_policy
from .utils import set_threads
from .viz import save_gif


def play(policy: Policy, games: int = 1, seed: int = EVAL_SEED, greedy: bool | None = None, max_frames: int = 3000,
         scale: int = 8, every: int = 1):
    """Play ``games`` games and return (rgb frames, list of scores)."""
    cfg = policy.cfg
    greedy = policy.kind == "q" if greedy is None else greedy  # same rule as evaluate()
    env = make_env(cfg.env, obs=cfg.obs, stack=cfg.stack, sticky=cfg.sticky, frameskip=cfg.frameskip,
                   render=cfg.env == "ale")
    frames, scores = [], []
    for g in range(games):
        obs, _ = env.reset(seed=seed + g)
        done, score, t = False, 0.0, 0
        while not done:
            if t % every == 0 and len(frames) < max_frames:
                frames.append(env.render_rgb(scale if cfg.env == "mini" else 1))
            a = int(policy.act(obs[None], greedy=greedy)[0])
            obs, r, term, trunc, _ = env.step(a)
            score += r
            done = term or trunc
            t += 1
        scores.append(score)
    env.close()
    return frames, scores


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play games with a trained agent and save a GIF.")
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--gif", default="", help="output GIF path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--greedy", action="store_true", help="argmax actions for the policy-gradient methods too")
    a = p.parse_args(argv)
    set_threads(1)
    pol = load_policy(a.checkpoint)
    frames, scores = play(pol, a.games, greedy=True if a.greedy else None, max_frames=a.max_frames)
    print(f"scores: {scores}  mean {np.mean(scores):+.2f}")
    if a.gif:
        save_gif(frames, a.gif, a.fps)
        print(f"wrote {a.gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
