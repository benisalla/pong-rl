"""Pixel preprocessing shared by both games, and the two ways of showing motion.

``preprocess`` turns a raw frame into a small binary image (1 where there is a paddle or the
ball, 0 elsewhere):

* ALE Pong, 210x160x3 -> 80x80, exactly Karpathy's ``prepro``: crop the score bar and the
  bottom border (rows 35..195), keep every second pixel and one colour channel, erase the two
  background colours (144 and 109), set everything else to 1.
* small Pong, 40x40 -> 20x20: the same "downsample by 2" step (as a 2x2 max so the 2x2 ball
  never disappears), then binarise. The ALE Karpathy frame is 80x80 = 6400 inputs, the small one
  20x20 = 400: about 16x less compute per step, which is what makes CPU training take minutes.

A single frame cannot tell you where the ball is going. Two fixes, chosen with ``--obs``:

* ``diff``  : x_t = f_t - f_{t-1}, one channel with values in {-1, 0, 1} (Karpathy). The ball
  shows up as a +1 blob in front of a -1 blob, which encodes its velocity. Anything that does not
  move (a still paddle) disappears, which is the price of the trick.
* ``stack`` : x_t = [f_{t-k+1}, ..., f_t], k channels in {0, 1} (DQN, Mnih et al. 2015). Nothing
  is lost, the network has to work out motion itself; k times more input.

Observations are stored as small integers (int8 for diff, uint8 for stack) so the DQN replay
buffer stays small; the networks call ``.float()`` on them.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def preprocess(frame: np.ndarray, kind: str) -> np.ndarray:
    """Raw frame -> (H, W) binary uint8 image (80x80 for ALE, 20x20 for the small game)."""
    if kind == "ale":
        f = frame[35:195:2, ::2, 0].copy()  # crop + downsample by 2 + red channel -> 80x80
        f[(f == 144) | (f == 109)] = 0  # erase the background
        return (f != 0).astype(np.uint8)  # paddles and ball -> 1
    if kind == "mini":
        # 40x40 -> 20x20: max over 2x2 blocks (a 2x2 ball covers 1 to 4 output pixels, never 0)
        f = frame.reshape(frame.shape[0] // 2, 2, frame.shape[1] // 2, 2).max(axis=(1, 3))
        return (f > 0).astype(np.uint8)
    raise ValueError(f"unknown frame kind {kind!r}")


class PixelObs:
    """Wrap a game: preprocess every frame and build the ``diff`` or ``stack`` observation.

    Also tracks the running episode score so training code can log it without peeking into the game.
    """

    def __init__(self, game, kind: str, obs: str = "stack", stack: int = 4):
        if obs not in ("diff", "stack"):
            raise ValueError("obs must be 'diff' or 'stack'")
        self.game, self.kind, self.mode, self.k = game, kind, obs, stack
        self.n_actions = game.n_actions
        self.action_meanings = game.action_meanings
        h, w = (80, 80) if kind == "ale" else (game.size // 2, game.size // 2)
        self.observation_shape = (1, h, w) if obs == "diff" else (stack, h, w)
        self.frames: deque = deque(maxlen=stack)
        self.prev = None

    def _obs(self, f: np.ndarray) -> np.ndarray:
        if self.mode == "diff":
            # first frame of an episode: no previous frame, Karpathy uses zeros
            x = np.zeros_like(f, dtype=np.int8) if self.prev is None else f.astype(np.int8) - self.prev.astype(np.int8)
            self.prev = f
            return x[None]
        self.frames.append(f)
        while len(self.frames) < self.k:  # start of episode: repeat the first frame
            self.frames.append(f)
        return np.stack(self.frames)

    def reset(self, seed: int | None = None):
        frame, info = self.game.reset(seed=seed)
        self.prev = None
        self.frames.clear()
        self.ep_return, self.ep_len = 0.0, 0
        return self._obs(preprocess(frame, self.kind)), info

    def step(self, action: int):
        frame, r, term, trunc, info = self.game.step(action)
        self.ep_return += r
        self.ep_len += 1
        if term or trunc:
            info = dict(info, episode={"r": self.ep_return, "l": self.ep_len})
        return self._obs(preprocess(frame, self.kind)), float(r), term, trunc, info

    def render_rgb(self, scale: int = 8) -> np.ndarray:
        return self.game.render_rgb(scale)

    def close(self) -> None:
        self.game.close()
