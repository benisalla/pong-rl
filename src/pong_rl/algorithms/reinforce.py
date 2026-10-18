"""REINFORCE with a learned baseline, on a CNN, with all three actions.

Same Monte Carlo idea as ``pg`` but with the two classic upgrades:

* a convolutional actor-critic (``models.ActorCritic``) instead of the MLP, softmax over
  NOOP / UP / DOWN;
* a baseline: A_t = G_t - V(s_t). Subtracting anything that does not depend on the action keeps
  the gradient unbiased, and V(s) (trained by regression on G_t) removes the part of the return
  that was going to happen anyway, so the variance drops.

    loss = -mean_t [A_t log pi(a_t|s_t)] + c_v mean_t (V(s_t) - G_t)^2 - c_e mean_t H(pi(.|s_t))

Episodes come from ``n_envs`` copies running side by side; once ``batch_episodes`` complete games
are collected, one Adam step. Still Monte Carlo: nothing is learnt until a game ends.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..metrics import RunLogger
from ..policy import build_model
from ..utils import discount_rewards
from .common import Evaluator, vec_env


def train(cfg: Config, logger: RunLogger):
    envs = vec_env(cfg)
    obs_shape, n_actions = envs.observation_shape, envs.n_actions
    model = build_model(cfg, obs_shape, n_actions, cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    evaluator = Evaluator(cfg, logger, model, n_actions)
    dev = next(model.parameters()).device

    obs = envs.reset()
    traj = [{"obs": [], "act": [], "rew": []} for _ in range(envs.n)]
    batch = []  # finished episodes: (obs, actions, returns)
    step = 0
    while not logger.done(step, cfg.steps):
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, device=dev))
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        next_obs, rews, terms, truncs, infos = envs.step(actions)
        step += envs.n
        for i in range(envs.n):
            traj[i]["obs"].append(obs[i])
            traj[i]["act"].append(actions[i])
            traj[i]["rew"].append(rews[i])
            if terms[i] or truncs[i]:
                logger.episode(step, infos[i]["episode"]["r"], infos[i]["episode"]["l"])
                G = discount_rewards(traj[i]["rew"], cfg.gamma, cfg.reset_on_point)
                batch.append((np.stack(traj[i]["obs"]), np.array(traj[i]["act"]), G))
                traj[i] = {"obs": [], "act": [], "rew": []}
        obs = next_obs

        if len(batch) >= cfg.batch_episodes:
            x = torch.as_tensor(np.concatenate([b[0] for b in batch]), device=dev)
            a = torch.as_tensor(np.concatenate([b[1] for b in batch]), device=dev)
            G = torch.as_tensor(np.concatenate([b[2] for b in batch]), device=dev)
            batch = []
            n = len(a)
            chunks = [slice(s, s + 2048) for s in range(0, n, 2048)]  # bounded memory
            with torch.no_grad():
                v_old = torch.cat([model(x[c])[1] for c in chunks])
            adv = G - v_old  # the baseline: how much better than expected
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            opt.zero_grad()
            losses = []
            for c in chunks:  # gradients of the chunks add up to the full-batch gradient
                logits, v = model(x[c])
                dist = torch.distributions.Categorical(logits=logits)
                pi_loss = -(dist.log_prob(a[c]) * adv[c]).sum() / n
                v_loss = F.mse_loss(v, G[c], reduction="sum") / n
                ent = dist.entropy().sum() / n
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                loss.backward()
                losses.append((pi_loss.item(), v_loss.item(), ent.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            pl, vl, en = np.sum(losses, axis=0)
            logger.loss(step, pi_loss=pl, v_loss=vl, entropy=en)
        evaluator(step)
        logger.maybe_print(step)
    envs.close()
    return model, obs_shape, n_actions
