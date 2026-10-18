"""Seeding, threads and the return / advantage helpers every algorithm shares."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_threads(n: int = 1) -> None:
    """One thread by default: on a shared CPU this is both polite and, for small nets, not slower."""
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)


def get_device(name: str = "cpu") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discount_rewards(rewards, gamma: float, reset_on_point: bool = True) -> np.ndarray:
    """G_t = r_t + gamma G_{t+1}, computed backwards over one episode.

    ``reset_on_point`` is Karpathy's Pong-specific trick: every non-zero reward ends a rally, so
    the running sum restarts there and an action is only credited with the point it helped decide.
    """
    r = np.asarray(rewards, dtype=np.float32)
    out = np.zeros_like(r)
    running = 0.0
    for t in reversed(range(len(r))):
        if reset_on_point and r[t] != 0:
            running = 0.0
        running = running * gamma + r[t]
        out[t] = running
    return out


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalised Advantage Estimation (Schulman et al. 2016) over a (T, N) rollout.

    delta_t = r_t + gamma V(s_{t+1}) (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma lam (1 - done_t) A_{t+1}
    lam = 1 gives Monte Carlo returns minus V (high variance), lam = 0 the one-step TD error
    (high bias). Returns (advantages, returns = advantages + values), both (T, N).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return end + (start - end) * max(0.0, 1.0 - t / max(1, duration))
