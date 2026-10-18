"""One flat config for every method, plus the per-method defaults.

Flat on purpose: ``pong-train --algo X --help`` shows every knob, and the checkpoint stores the
whole dict so ``pong-eval`` / ``pong-play`` can rebuild the exact same network and env.
Only the fields an algorithm reads matter to it; the table below says which.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

ALGOS = ("pg", "reinforce", "a2c", "ppo", "dqn", "double-dqn", "dueling-dqn")


@dataclass
class Config:
    # ---- shared
    algo: str = "ppo"
    env: str = "mini"  # mini (built-in 40x40 numpy Pong) | ale (ALE/Pong-v5)
    obs: str = "stack"  # diff (f_t - f_{t-1}, 1 channel) | stack (last `stack` frames)
    stack: int = 4
    sticky: float = 0.0  # probability of repeating the previous action (ALE v5 uses 0.25)
    frameskip: int = 0  # 0 = game default: 2 on the small game, 4 on ALE
    seed: int = 0
    steps: int = 300_000  # environment steps (agent decisions) to train for
    gamma: float = 0.97  # the small game decides a point within ~20 decisions; ALE runs use 0.99
    lr: float = 2.5e-4
    encoder: str = "auto"  # auto | small | nature (CNN methods)
    threads: int = 1
    device: str = "cpu"  # cpu | cuda | auto (cuda when available; the notebook uses it)
    eval_every: int = 25_000  # env steps between greedy evaluations
    eval_episodes: int = 10
    final_eval_episodes: int = 30
    out: str = ""  # run directory (default runs/<algo>)
    max_minutes: float = 0.0  # stop early after this much wall time (0 = no limit), a safety net on a shared CPU
    # ---- pg (Karpathy) and reinforce
    hidden: int = 200  # pg: hidden units of the 2-layer MLP
    batch_episodes: int = 10  # episodes per update
    reset_on_point: bool = True  # restart the discounted sum at every point (Pong trick)
    # ---- a2c / ppo / reinforce
    n_envs: int = 8
    n_steps: int = 128  # rollout length per env (a2c: 5)
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    clip: float = 0.1  # ppo
    epochs: int = 4  # ppo
    minibatches: int = 4  # ppo
    anneal_lr: bool = True
    # ---- dqn family
    buffer_size: int = 50_000
    batch_size: int = 32
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_fraction: float = 0.2  # fraction of `steps` over which epsilon decays linearly
    double: bool = False
    dueling: bool = False
    per: bool = False  # prioritized replay
    per_alpha: float = 0.6
    per_beta: float = 0.4  # annealed to 1 over training
    n_step: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Defaults that differ per method (small-game settings; the notebook overrides them for ALE).
ALGO_DEFAULTS: dict[str, dict] = {
    "pg": dict(obs="diff", lr=3e-3, steps=2_000_000, batch_episodes=2),
    "reinforce": dict(lr=1e-3, steps=1_200_000, n_envs=8, batch_episodes=2),
    "a2c": dict(lr=7e-4, steps=400_000, n_envs=16, n_steps=5, gae_lambda=1.0),
    "ppo": dict(lr=1e-3, steps=400_000, n_envs=8, n_steps=64, minibatches=2),
    "dqn": dict(lr=2.5e-4, steps=180_000, n_envs=8, batch_size=64, train_freq=8, eps_fraction=0.1),
    "double-dqn": dict(lr=2.5e-4, steps=180_000, n_envs=8, batch_size=64, train_freq=8, eps_fraction=0.1,
                       double=True),
    "dueling-dqn": dict(lr=2.5e-4, steps=180_000, n_envs=8, batch_size=64, train_freq=8, eps_fraction=0.1,
                        double=True, dueling=True),
}


def make_config(algo: str, **overrides) -> Config:
    """Base defaults <- per-method defaults <- your overrides (``None`` values are ignored)."""
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}, choose from {ALGOS}")
    d = dict(ALGO_DEFAULTS[algo])
    d.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(algo=algo, **{k: v for k, v in d.items() if k != "algo"})
    if not cfg.out:
        cfg.out = f"runs/{algo}"
    return cfg
