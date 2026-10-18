"""Karpathy's policy gradient ("Pong from Pixels", 2016), the method of the original notebook.

Network: x (frame difference, 1600 or 6400 pixels) -> 200 ReLU -> 1 sigmoid = p = P(UP | x).
Actions: UP with probability p, DOWN otherwise (no NOOP).

REINFORCE gradient for one step, with y = 1 if the sampled action was UP else 0:

    log pi(a|x) = y log p + (1 - y) log(1 - p)
    d log pi(a|x) / d logit = y - p                (Karpathy's "fake label" gradient)

so ``loss = -sum_t A_t log pi(a_t|x_t)`` with autograd gives exactly his hand-written backward
pass (``tests/test_algorithms.py`` checks this). The advantage A_t is the discounted return,
restarted at every point (``reset_on_point``) and standardised per episode (zero mean, unit std):
roughly half the actions get pushed up, half pushed down, which lowers the variance.
Gradients are summed over ``batch_episodes`` games, then one RMSprop step (decay 0.99).
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from ..metrics import RunLogger
from ..policy import UP, build_model
from ..utils import discount_rewards
from .common import Evaluator, single_env


def pg_loss(logits: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """-sum_t A_t log pi(a_t | x_t) for the Bernoulli UP/DOWN policy."""
    y = (actions == UP).float()
    logp = -torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return -(logp * advantages).sum()


def train(cfg: Config, logger: RunLogger):
    env = single_env(cfg)
    obs_shape, n_actions = env.observation_shape, env.n_actions
    model = build_model(cfg, obs_shape, n_actions, cfg.device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)
    evaluator = Evaluator(cfg, logger, model, n_actions)
    rng = np.random.default_rng(cfg.seed)
    dev = next(model.parameters()).device

    obs, _ = env.reset(seed=cfg.seed)
    xs, acts, rews = [], [], []
    step, episodes = 0, 0
    opt.zero_grad()
    while not logger.done(step, cfg.steps):
        with torch.no_grad():
            p_up = torch.sigmoid(model(torch.as_tensor(obs[None], device=dev))).item()
        action = UP if rng.random() < p_up else 2  # roll the dice
        xs.append(obs)
        acts.append(action)
        obs, r, term, trunc, info = env.step(action)
        rews.append(r)
        step += 1
        if term or trunc:
            episodes += 1
            logger.episode(step, info["episode"]["r"], info["episode"]["l"])
            adv = discount_rewards(rews, cfg.gamma, cfg.reset_on_point)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            logits = model(torch.as_tensor(np.stack(xs), device=dev))
            loss = pg_loss(logits, torch.as_tensor(acts, device=dev), torch.as_tensor(adv, device=dev))
            loss.backward()  # gradients accumulate over the batch of episodes
            if episodes % cfg.batch_episodes == 0:
                opt.step()
                opt.zero_grad()
                logger.loss(step, loss=loss.item() / len(xs), p_up=float(torch.sigmoid(logits.detach()).mean()))
            xs, acts, rews = [], [], []
            obs, _ = env.reset()
        evaluator(step)
        logger.maybe_print(step)
    env.close()
    return model, obs_shape, n_actions
