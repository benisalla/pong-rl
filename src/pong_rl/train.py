"""``pong-train``: train one method, evaluate it, save ``model.pt``, ``log.json`` and ``curve.png``.

    pong-train --algo ppo                          # small Pong, about 10 min on one CPU thread
    pong-train --algo pg --obs diff                # Karpathy's method, as in the original notebook
    pong-train --algo dqn --per --n-step 3         # DQN + prioritized replay + 3-step returns
    pong-train --algo ppo --env ale --steps 5000   # ALE Pong smoke run (real runs: the notebook)

Every ``Config`` field is a flag (``--n-envs``, ``--lr``, ``--gamma`` ...); unset flags take the
method's default from ``config.ALGO_DEFAULTS``.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields

from .algorithms import TRAINERS
from .config import ALGOS, Config, make_config
from .evaluate import evaluate
from .metrics import RunLogger
from .models import n_params
from .policy import Policy, save_checkpoint
from .utils import set_seed, set_threads


def run(cfg: Config, plot: bool = True) -> dict:
    """Train ``cfg.algo`` and return the final evaluation (also written to ``<out>/log.json``)."""
    set_threads(cfg.threads)
    set_seed(cfg.seed)
    logger = RunLogger(cfg.out, cfg.to_dict())
    print(f"[{cfg.algo}] env={cfg.env} obs={cfg.obs} steps={cfg.steps:,} -> {cfg.out}", flush=True)
    model, obs_shape, n_actions = TRAINERS[cfg.algo](cfg, logger)
    train_time = logger.elapsed()
    save_checkpoint(os.path.join(cfg.out, "model.pt"), cfg, model, obs_shape, n_actions)
    res = evaluate(Policy(cfg, model, n_actions), cfg.final_eval_episodes)
    res["params"] = n_params(model)
    res["train_time_s"] = round(train_time, 1)
    logger.finish(res, logger.last_step)
    print(f"[{cfg.algo}] final score {res['mean']:+.2f} +- {res['std']:.2f} over {res['episodes']} games, "
          f"win rate {res['win_rate']:.0%}, {n_params(model):,} params, {train_time / 60:.1f} min", flush=True)
    if plot:
        from .viz import plot_runs

        plot_runs([cfg.out], os.path.join(cfg.out, "curve.png"), title=f"{cfg.algo} on {cfg.env}")
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a pixel Pong agent.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="ppo")
    for f in fields(Config):
        if f.name == "algo":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            name = f.type if isinstance(f.type, str) else f.type.__name__
            typ = {"int": int, "float": float, "str": str}.get(name, str)
            p.add_argument(flag, type=typ, default=None, help=f"(base default: {f.default})")
    return p


def main(argv=None) -> None:
    a = vars(build_parser().parse_args(argv))
    algo = a.pop("algo")
    cfg = make_config(algo, **a)
    t = time.time()
    run(cfg)
    print(f"done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
