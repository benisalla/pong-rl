"""Replay memory for the DQN family.

* ``ReplayBuffer``: circular arrays (obs, action, reward, next_obs, done, discount); uniform sampling.
  Memory = capacity x 2 x obs bytes: 50k transitions of 4x40x40 uint8 = 320 MB, of 1x40x40
  int8 (``--obs diff``) = 160 MB. For ALE 4x80x80 the notebook uses 100k = 2.6 GB.
* ``PrioritizedReplayBuffer`` (Schaul et al. 2016): sample transition i with probability
  p_i^alpha / sum_j p_j^alpha where p_i = |TD error| + eps, and correct the bias with importance
  weights w_i = (N P(i))^-beta / max_j w_j. A sum tree gives O(log N) sampling and updates.
* ``NStepAccumulator``: turns 1-step transitions into n-step ones,
  (s_t, a_t, r_t + g r_{t+1} + ... + g^{n-1} r_{t+n-1}, s_{t+n}, done, g^n), cut at episode ends.
  The stored discount g^n (or g^k for a truncated tail) is what the target multiplies max Q by.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, obs_dtype=np.uint8, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done, discount) -> int:
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i] = action, reward
        self.dones[i], self.discounts[i] = float(done), discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _gather(self, idx):
        return dict(obs=self.obs[idx], actions=self.actions[idx], rewards=self.rewards[idx],
                    next_obs=self.next_obs[idx], dones=self.dones[idx], discounts=self.discounts[idx],
                    idx=idx, weights=np.ones(len(idx), dtype=np.float32))

    def sample(self, batch_size: int, beta: float = 0.0) -> dict:
        return self._gather(self.rng.integers(0, self.size, size=batch_size))

    def update_priorities(self, idx, td_errors) -> None:  # uniform replay ignores priorities
        pass


class SumTree:
    """Binary tree whose leaves hold priorities and whose internal nodes hold sums of their children.

    ``find(u)`` walks from the root to the leaf where the cumulative sum crosses u, so sampling
    proportionally to priority is a uniform draw in [0, total) followed by one O(log N) walk.
    """

    def __init__(self, capacity: int):
        self.n = 1
        while self.n < capacity:
            self.n *= 2
        self.tree = np.zeros(2 * self.n, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, i: int, p: float) -> None:
        j = i + self.n
        self.tree[j] = p
        j //= 2
        while j >= 1:
            self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
            j //= 2

    def find(self, u: float) -> int:
        j = 1
        while j < self.n:
            left = 2 * j
            if u < self.tree[left]:
                j = left
            else:
                u -= self.tree[left]
                j = left + 1
        return j - self.n

    def get(self, i: int) -> float:
        return float(self.tree[i + self.n])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_shape, obs_dtype=np.uint8, alpha: float = 0.6, eps: float = 1e-3, seed: int = 0):
        super().__init__(capacity, obs_shape, obs_dtype, seed)
        self.tree = SumTree(capacity)
        self.alpha, self.eps = alpha, eps
        self.max_priority = 1.0

    def add(self, *args) -> int:
        i = super().add(*args)
        self.tree.update(i, self.max_priority ** self.alpha)  # new transitions get seen at least once
        return i

    def sample(self, batch_size: int, beta: float = 0.4) -> dict:
        # stratified: one draw per equal slice of the total mass, lower variance than iid draws
        seg = self.tree.total / batch_size
        u = (np.arange(batch_size) + self.rng.random(batch_size)) * seg
        idx = np.array([min(self.tree.find(x), self.size - 1) for x in u])
        probs = np.array([self.tree.get(i) for i in idx]) / self.tree.total
        w = (self.size * probs) ** (-beta)
        batch = self._gather(idx)
        batch["weights"] = (w / w.max()).astype(np.float32)
        return batch

    def update_priorities(self, idx, td_errors) -> None:
        p = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(p.max()))
        for i, pi in zip(idx, p):
            self.tree.update(int(i), float(pi) ** self.alpha)


class NStepAccumulator:
    """Keeps the last n transitions of one env and emits n-step transitions."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.q: deque = deque()

    def _emit(self):
        R, g = 0.0, 1.0
        for (_, _, r, _, _) in self.q:
            R += g * r
            g *= self.gamma
        s, a = self.q[0][0], self.q[0][1]
        _, _, _, s_next, done = self.q[-1]
        return s, a, R, s_next, done, g  # g = gamma ** (number of rewards summed)

    def push(self, s, a, r, s_next, done, truncated=False) -> list:
        """Add one transition, return the list of n-step transitions that are now complete."""
        self.q.append((s, a, r, s_next, done))
        out = []
        if done or truncated:  # flush everything: each remaining start gets a shorter tail
            while self.q:
                out.append(self._emit())
                self.q.popleft()
        elif len(self.q) == self.n:
            out.append(self._emit())
            self.q.popleft()
        return out
