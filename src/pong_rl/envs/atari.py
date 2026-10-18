"""ALE Pong (``ALE/Pong-v5``) through gymnasium + ale-py, with the reduced action set.

ale-py >= 0.9 ships the Atari ROMs, so ``pip install gymnasium ale-py`` is enough and the
environment runs headless (no display, no ``render_mode`` needed).

* ``frameskip=4``: the emulator repeats each action for 4 frames and returns the last frame
  (the agent decides at 15 Hz instead of 60 Hz; 4x fewer forward passes, same game).
* ``repeat_action_probability`` (``--sticky``): with this probability the previous action is
  repeated instead of the new one, the standard way to make ALE stochastic (Machado et al.,
  2018). 0.25 is the ``v5`` default; 0.0 reproduces the classic deterministic DQN setting.
* Pong has 6 actions but only three matter: NOOP, UP (ALE ``RIGHT`` = 2) and DOWN (ALE
  ``LEFT`` = 3). Every agent in this repo outputs 0/1/2 and ``ACTION_MAP`` translates.
"""
from __future__ import annotations

import numpy as np

ACTION_MAP = (0, 2, 3)  # our NOOP, UP, DOWN -> ALE action ids
ACTION_MEANINGS = ["NOOP", "UP", "DOWN"]


class AtariPong:
    """Thin wrapper around ``gym.make("ALE/Pong-v5")`` exposing the 3-action interface."""

    n_actions = 3
    action_meanings = ACTION_MEANINGS
    observation_shape = (210, 160, 3)

    def __init__(self, frameskip: int = 4, sticky: float = 0.0, max_steps: int = 27_000,
                 seed: int | None = None, render: bool = False):
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Pong-v5",
            frameskip=frameskip,
            repeat_action_probability=sticky,
            max_episode_steps=max_steps,  # 27k agent steps = 108k frames = 30 min of play
            full_action_space=False,
            render_mode="rgb_array" if render else None,
        )
        self._seed = seed

    def reset(self, seed: int | None = None):
        seed = self._seed if seed is None else seed
        self._seed = None  # seed only the first reset, later resets continue the RNG
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: int):
        return self.env.step(ACTION_MAP[int(action)])

    def render_rgb(self, scale: int = 1) -> np.ndarray:
        frame = self.env.render()
        return np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8)) if scale > 1 else frame

    def close(self) -> None:
        self.env.close()
