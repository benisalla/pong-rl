"""What a trained agent *is* at play time: a network plus a rule to turn its output into an action.

* ``pg``                : P(UP) = sigmoid(logit); greedy picks UP if P(UP) > 0.5, else DOWN.
                          (Karpathy's agent only has UP / DOWN, never NOOP.)
* ``reinforce/a2c/ppo`` : softmax over (NOOP, UP, DOWN); greedy = argmax, else sample.
* ``*dqn``              : argmax_a Q(s, a); epsilon-greedy while exploring.

Checkpoints are ``{"config", "obs_shape", "n_actions", "state_dict"}``; ``load_policy`` rebuilds it.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .models import ActorCritic, KarpathyMLP, QNetwork
from .utils import get_device

UP, DOWN = 1, 2


def build_model(cfg: Config, obs_shape, n_actions: int, device: str | None = None) -> torch.nn.Module:
    if cfg.algo == "pg":
        model = KarpathyMLP(int(np.prod(obs_shape)), cfg.hidden)
    elif cfg.algo in ("reinforce", "a2c", "ppo"):
        model = ActorCritic(obs_shape, n_actions, cfg.encoder)
    else:
        model = QNetwork(obs_shape, n_actions, dueling=cfg.dueling, encoder=cfg.encoder)
    return model.to(get_device(device or "cpu"))


class Policy:
    def __init__(self, cfg: Config, model: torch.nn.Module, n_actions: int = 3, seed: int = 0):
        self.cfg, self.model, self.n_actions = cfg, model, n_actions
        self.kind = "pg" if cfg.algo == "pg" else ("ac" if cfg.algo in ("reinforce", "a2c", "ppo") else "q")
        self.rng = np.random.default_rng(seed)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = True, eps: float = 0.0) -> np.ndarray:
        """obs: (B, C, H, W) -> actions (B,) in {0: NOOP, 1: UP, 2: DOWN}."""
        x = torch.as_tensor(obs, device=self.device)
        if self.kind == "pg":
            p_up = torch.sigmoid(self.model(x)).cpu().numpy()
            up = p_up > 0.5 if greedy else self.rng.random(len(p_up)) < p_up
            return np.where(up, UP, DOWN)
        if self.kind == "ac":
            logits, _ = self.model(x)
            if greedy:
                return logits.argmax(1).cpu().numpy()
            return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        a = self.model(x).argmax(1).cpu().numpy()
        if eps > 0:
            rand = self.rng.random(len(a)) < eps
            a = np.where(rand, self.rng.integers(0, self.n_actions, len(a)), a)
        return a


def save_checkpoint(path: str, cfg: Config, model, obs_shape, n_actions: int, extra: dict | None = None) -> None:
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"config": cfg.to_dict(), "obs_shape": tuple(obs_shape), "n_actions": n_actions,
                "state_dict": state, **(extra or {})}, path)


def load_policy(path: str, device: str = "cpu") -> Policy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg, ck["obs_shape"], ck["n_actions"])
    model.load_state_dict(ck["state_dict"])
    model.to(get_device(device)).eval()
    return Policy(cfg, model, ck["n_actions"])
