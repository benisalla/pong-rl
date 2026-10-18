"""PPO with the clipped surrogate objective (Schulman et al. 2017), the fast one here.

Same rollouts as A2C (``n_envs`` x ``n_steps``), but each batch is reused for ``epochs`` passes of
``minibatches`` minibatches, which is where the sample efficiency comes from. Reusing data makes
the policy drift away from the one that collected it, so the update is kept close with a ratio clip:

    rho_t   = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
    L_clip  = -mean_t min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t)
    loss    = L_clip + c_v mean (V(s_t) - R_t)^2 - c_e mean H(pi(.|s_t))

A_t comes from GAE(gamma, lambda = 0.95) and is normalised per minibatch; R_t = A_t + V_old(s_t).
Once rho leaves [1 - eps, 1 + eps] in the direction the advantage pushes, the gradient of that
sample is zero: no single batch can move the policy far.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from ..metrics import RunLogger
from ..policy import build_model
from ..utils import gae, linear_schedule
from .a2c import collect_rollout
from .common import Evaluator, vec_env


def ppo_loss(model, obs, actions, old_logp, advantages, returns, cfg: Config):
    logits, values = model(obs)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(actions)
    ratio = torch.exp(logp - old_logp)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pi_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    v_loss = 0.5 * (returns - values).pow(2).mean()
    entropy = dist.entropy().mean()
    clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean()
    loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    return loss, dict(pi_loss=pi_loss.item(), v_loss=v_loss.item(), entropy=entropy.item(),
                      clip_frac=clip_frac.item(), approx_kl=(old_logp - logp).mean().item())


def train(cfg: Config, logger: RunLogger):
    envs = vec_env(cfg)
    obs_shape, n_actions = envs.observation_shape, envs.n_actions
    model = build_model(cfg, obs_shape, n_actions, cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    evaluator = Evaluator(cfg, logger, model, n_actions)
    rng = np.random.default_rng(cfg.seed)
    obs, step = envs.reset(), 0
    batch = cfg.n_envs * cfg.n_steps
    mb = batch // cfg.minibatches
    while not logger.done(step, cfg.steps):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = linear_schedule(cfg.lr, 0.0, cfg.steps, step)
        roll, obs, step = collect_rollout(model, envs, obs, cfg.n_steps, step, logger)
        adv, ret = gae(roll["rewards"], roll["values"], roll["dones"], roll["last_value"], cfg.gamma, cfg.gae_lambda)
        flat = lambda t: t.reshape(-1, *t.shape[2:])  # noqa: E731
        b_obs, b_act, b_logp, b_adv, b_ret = (flat(roll["obs"]), flat(roll["actions"]), flat(roll["logp"]),
                                              flat(adv), flat(ret))
        for _ in range(cfg.epochs):
            perm = rng.permutation(batch)
            for s in range(0, batch, mb):
                i = torch.as_tensor(perm[s:s + mb], device=b_obs.device)
                loss, stats = ppo_loss(model, b_obs[i], b_act[i], b_logp[i], b_adv[i], b_ret[i], cfg)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
        logger.loss(step, **stats)
        evaluator(step)
        logger.maybe_print(step)
    envs.close()
    return model, obs_shape, n_actions
