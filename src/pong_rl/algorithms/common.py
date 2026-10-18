"""Small pieces every training loop shares: env construction and periodic evaluation."""
from __future__ import annotations

from ..config import Config
from ..envs import VecEnv, make_env
from ..evaluate import evaluate
from ..metrics import RunLogger
from ..policy import Policy


def env_kwargs(cfg: Config) -> dict:
    return dict(env=cfg.env, obs=cfg.obs, stack=cfg.stack, sticky=cfg.sticky, frameskip=cfg.frameskip)


def single_env(cfg: Config):
    return make_env(seed=cfg.seed, **env_kwargs(cfg))


def vec_env(cfg: Config) -> VecEnv:
    return VecEnv(cfg.n_envs, seed=cfg.seed, **env_kwargs(cfg))


class Evaluator:
    """Calls ``evaluate`` every ``cfg.eval_every`` env steps and logs the result."""

    def __init__(self, cfg: Config, logger: RunLogger, model, n_actions: int):
        self.cfg, self.logger = cfg, logger
        self.policy = Policy(cfg, model, n_actions)
        self.next_at = cfg.eval_every

    def __call__(self, step: int) -> None:
        if self.cfg.eval_every <= 0 or step < self.next_at:
            return
        while self.next_at <= step:
            self.next_at += self.cfg.eval_every
        self.logger.eval(step, evaluate(self.policy, self.cfg.eval_episodes))
