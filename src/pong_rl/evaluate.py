"""Evaluation used by every method: mean game score over N full games.

Each agent plays its own policy: the policy-gradient family (pg, reinforce, a2c, ppo) samples
from pi(a|s), the stochastic policy it was trained to optimise; the DQN family takes
argmax_a Q(s, a) (epsilon = 0). ``greedy=True`` forces argmax for everyone (``pong-eval --greedy``):
for Karpathy's UP/DOWN-only policy that is usually *worse*, because a paddle that must always move
jitters around the ball instead of hovering under it.

Score = points won - points lost in one game (first to 5 on the small game, so in [-5, 5];
first to 21 on ALE Pong, so in [-21, 21]). The N episodes run as N parallel copies with fixed
seeds (10_000 + i), so two methods are always scored on the same serves.

CLI: ``pong-eval runs/ppo/model.pt --episodes 30``
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .envs import make_env
from .policy import Policy, load_policy
from .utils import set_threads

EVAL_SEED = 10_000


def evaluate(policy: Policy, episodes: int = 10, greedy: bool | None = None, eps: float = 0.0,
             seed: int = EVAL_SEED, env: str | None = None) -> dict:
    cfg = policy.cfg
    if greedy is None:
        greedy = policy.kind == "q"
    envs = [make_env(env or cfg.env, obs=cfg.obs, stack=cfg.stack, sticky=cfg.sticky,
                     frameskip=cfg.frameskip) for _ in range(episodes)]
    obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    alive = np.ones(episodes, dtype=bool)
    scores, lengths = np.zeros(episodes), np.zeros(episodes, dtype=int)
    wins = np.zeros(episodes, dtype=bool)
    was_training = policy.model.training
    policy.model.eval()
    while alive.any():
        idx = np.flatnonzero(alive)
        actions = policy.act(obs[idx], greedy=greedy, eps=eps)
        for j, a in zip(idx, actions):
            o, r, term, trunc, info = envs[j].step(int(a))
            obs[j] = o
            scores[j] += r
            lengths[j] += 1
            if term or trunc:
                alive[j] = False
                wins[j] = term and scores[j] > 0
    policy.model.train(was_training)
    for e in envs:
        e.close()
    return {"mean": float(scores.mean()), "std": float(scores.std()), "min": float(scores.min()),
            "max": float(scores.max()), "win_rate": float(wins.mean()), "mean_length": float(lengths.mean()),
            "episodes": episodes, "scores": scores.tolist()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint: mean score over N games.")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--greedy", action="store_true", help="argmax actions for the policy-gradient methods too")
    p.add_argument("--eps", type=float, default=0.0, help="epsilon-greedy noise (DQN family)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    a = p.parse_args(argv)
    set_threads(1)
    pol = load_policy(a.checkpoint)
    res = evaluate(pol, a.episodes, greedy=True if a.greedy else None, eps=a.eps)
    if a.json:
        print(json.dumps(res))
    else:
        print(f"{pol.cfg.algo} on {pol.cfg.env}: score {res['mean']:+.2f} +- {res['std']:.2f} "
              f"(min {res['min']:+.0f}, max {res['max']:+.0f}), win rate {res['win_rate']:.0%}, "
              f"{res['episodes']} games")


if __name__ == "__main__":
    main()
