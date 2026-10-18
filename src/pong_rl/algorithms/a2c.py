"""A2C, the synchronous advantage actor-critic (Mnih et al. 2016, without the asynchrony).

Instead of waiting for whole games (REINFORCE), learn from short n-step rollouts of ``n_envs``
parallel games and bootstrap the rest with the critic:

    G_t = r_t + g r_{t+1} + ... + g^{n-1} r_{t+n-1} + g^n V(s_{t+n})      (0 past a game end)
    A_t = G_t - V(s_t)
    loss = -mean[A_t log pi(a_t|s_t)] + c_v mean[(G_t - V(s_t))^2] - c_e mean[H(pi(.|s_t))]

``gae`` with lambda = 1 computes exactly these n-step returns. Many envs x few steps gives a
batch of decorrelated samples (n_envs = 16, n_steps = 5 -> 80 per update), and every update is
a single gradient step: on-policy, data is thrown away after use.
"""
from __future__ import annotations

import torch

from ..config import Config
from ..metrics import RunLogger
from ..policy import build_model
from ..utils import gae, linear_schedule
from .common import Evaluator, vec_env


def actor_critic_loss(model, obs, actions, advantages, returns, cfg: Config):
    logits, values = model(obs)
    dist = torch.distributions.Categorical(logits=logits)
    pi_loss = -(dist.log_prob(actions) * advantages).mean()
    v_loss = 0.5 * (returns - values).pow(2).mean()
    entropy = dist.entropy().mean()
    return pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy, pi_loss, v_loss, entropy


def collect_rollout(model, envs, obs, n_steps: int, step: int, logger: RunLogger):
    """Run every env for ``n_steps``; returns tensors shaped (T, N, ...) plus the last obs."""
    buf_obs, buf_act, buf_rew, buf_done, buf_val, buf_logp = [], [], [], [], [], []
    dev = next(model.parameters()).device
    for _ in range(n_steps):
        x = torch.as_tensor(obs, device=dev)
        with torch.no_grad():
            logits, v = model(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
        next_obs, r, term, trunc, infos = envs.step(a.cpu().numpy())
        step += envs.n
        for info in infos:
            if "episode" in info:
                logger.episode(step, info["episode"]["r"], info["episode"]["l"])
        buf_obs.append(x)
        buf_act.append(a)
        buf_rew.append(torch.as_tensor(r, device=dev))
        buf_done.append(torch.as_tensor(term | trunc, dtype=torch.float32, device=dev))
        buf_val.append(v)
        buf_logp.append(dist.log_prob(a))
        obs = next_obs
    with torch.no_grad():
        _, last_v = model(torch.as_tensor(obs, device=dev))
    roll = dict(obs=torch.stack(buf_obs), actions=torch.stack(buf_act), rewards=torch.stack(buf_rew),
                dones=torch.stack(buf_done), values=torch.stack(buf_val), logp=torch.stack(buf_logp),
                last_value=last_v)
    return roll, obs, step


def train(cfg: Config, logger: RunLogger):
    envs = vec_env(cfg)
    obs_shape, n_actions = envs.observation_shape, envs.n_actions
    model = build_model(cfg, obs_shape, n_actions, cfg.device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)
    evaluator = Evaluator(cfg, logger, model, n_actions)
    obs, step, updates = envs.reset(), 0, 0
    while not logger.done(step, cfg.steps):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = linear_schedule(cfg.lr, 0.0, cfg.steps, step)
        roll, obs, step = collect_rollout(model, envs, obs, cfg.n_steps, step, logger)
        adv, ret = gae(roll["rewards"], roll["values"], roll["dones"], roll["last_value"], cfg.gamma, cfg.gae_lambda)
        flat = lambda t: t.reshape(-1, *t.shape[2:])  # noqa: E731  (T, N, ...) -> (T*N, ...)
        loss, pl, vl, ent = actor_critic_loss(model, flat(roll["obs"]), flat(roll["actions"]),
                                              flat(adv), flat(ret), cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        updates += 1
        if updates % 50 == 0:
            logger.loss(step, pi_loss=pl.item(), v_loss=vl.item(), entropy=ent.item())
        evaluator(step)
        logger.maybe_print(step)
    envs.close()
    return model, obs_shape, n_actions
