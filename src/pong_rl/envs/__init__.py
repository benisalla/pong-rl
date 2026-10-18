"""Environments: the small numpy Pong, ALE Pong, preprocessing and a tiny vector env.

``make_env("mini" | "ale", obs="diff" | "stack")`` is the only entry point the algorithms use, so
every method sees exactly the same observations.
"""
from __future__ import annotations

import numpy as np

from .atari import AtariPong
from .mini_pong import MiniPong
from .wrappers import PixelObs, preprocess

__all__ = ["AtariPong", "MiniPong", "PixelObs", "VecEnv", "make_env", "preprocess"]


def make_env(env: str = "mini", obs: str = "stack", stack: int = 4, sticky: float = 0.0,
             seed: int | None = None, frameskip: int = 0, render: bool = False, **game_kwargs) -> PixelObs:
    """Build a game and wrap it in the shared pixel preprocessing. ``frameskip=0``: game default (mini 2, ALE 4)."""
    if env == "mini":
        game = MiniPong(sticky=sticky, seed=seed, frameskip=frameskip or 2, **game_kwargs)
    elif env == "ale":
        game = AtariPong(frameskip=frameskip or 4, sticky=sticky, seed=seed, render=render, **game_kwargs)
    else:
        raise ValueError(f"unknown env {env!r}, use 'mini' or 'ale'")
    return PixelObs(game, kind=env, obs=obs, stack=stack)


class VecEnv:
    """N copies of an env stepped in lock-step in one process (what A2C and PPO need).

    ``step`` auto-resets finished copies: the returned observation is already the first one of the
    next episode, ``infos[i]["final_obs"]`` is the last observation of the finished one (DQN needs
    it to bootstrap through a time-limit cut), and ``infos[i]["episode"]`` carries its score and length.
    Stepping in one process is the simplest choice and is fast enough for the small game.
    """

    def __init__(self, n: int, seed: int = 0, **env_kwargs):
        self.envs = [make_env(seed=seed + i, **env_kwargs) for i in range(n)]
        self.n = n
        self.n_actions = self.envs[0].n_actions
        self.observation_shape = self.envs[0].observation_shape
        self._seed = seed

    def reset(self) -> np.ndarray:
        return np.stack([e.reset(seed=self._seed + i)[0] for i, e in enumerate(self.envs)])

    def step(self, actions):
        obs, rews, terms, truncs, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(int(a))
            if te or tr:
                info = dict(info, final_obs=o)
                o, _ = e.reset()
            obs.append(o)
            rews.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        return np.stack(obs), np.array(rews, np.float32), np.array(terms), np.array(truncs), infos

    def close(self) -> None:
        for e in self.envs:
            e.close()
