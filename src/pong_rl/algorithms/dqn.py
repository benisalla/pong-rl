"""DQN (Mnih et al. 2015) and its upgrades, all behind flags of one training loop.

Learn Q(s, a) by regression on a bootstrapped target, act epsilon-greedily on it:

    y = r + g^n (1 - done) max_a' Q_target(s', a')                       DQN
    y = r + g^n (1 - done) Q_target(s', argmax_a' Q_online(s', a'))      double DQN (--double)
    loss = mean_i w_i huber(Q_online(s_i, a_i) - y_i)

The tricks that make it work from pixels:
* replay buffer: train on random old transitions, not on the last (highly correlated) frames;
* target network: a frozen copy, refreshed every ``target_update`` steps, so the target does not
  move with every gradient step;
* Huber loss + gradient clipping: one surprising reward cannot blow up the update;
* epsilon-greedy from 1.0 down to ``eps_end`` over ``eps_fraction`` of training.
Upgrades: double DQN removes the max-over-noise overestimation; dueling (--dueling) shares one
V(s) across actions (most Pong frames do not care which way you move); prioritized replay (--per)
replays surprising transitions more; n-step returns (--n-step) propagate the point reward back
n frames per update instead of one.

``--algo double-dqn`` = ``dqn --double``, ``--algo dueling-dqn`` = ``dqn --double --dueling``.

Acting uses ``n_envs`` games at once (one batched forward pass instead of n), and the replay ratio
is fixed by ``train_freq``: one gradient step of ``batch_size`` samples per ``train_freq`` env
steps, whatever ``n_envs`` is.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from ..config import Config
from ..metrics import RunLogger
from ..policy import Policy, build_model
from ..utils import linear_schedule
from .common import Evaluator, vec_env


def _t(x, dev):
    return torch.as_tensor(x, device=dev)


def _device(model) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def td_targets(online, target, batch: dict, double: bool) -> torch.Tensor:
    dev = _device(online)
    next_obs = _t(batch["next_obs"], dev)
    with torch.no_grad():
        q_next_target = target(next_obs)
        if double:
            a_star = online(next_obs).argmax(1, keepdim=True)  # online net chooses ...
            q_next = q_next_target.gather(1, a_star).squeeze(1)  # ... target net evaluates
        else:
            q_next = q_next_target.max(1).values
    r, g, done = _t(batch["rewards"], dev), _t(batch["discounts"], dev), _t(batch["dones"], dev)
    return r + g * (1.0 - done) * q_next


def dqn_loss(online, target, batch: dict, double: bool):
    dev = _device(online)
    q = online(_t(batch["obs"], dev)).gather(1, _t(batch["actions"], dev)[:, None]).squeeze(1)
    y = td_targets(online, target, batch, double)
    td = q - y
    loss = (_t(batch["weights"], dev) * F.smooth_l1_loss(q, y, reduction="none")).mean()
    return loss, td.detach().cpu().numpy(), q.mean().item()


def train(cfg: Config, logger: RunLogger):
    envs = vec_env(cfg)  # a few envs acted on with one batched forward pass (cheaper on a CPU)
    obs_shape, n_actions = envs.observation_shape, envs.n_actions
    online = build_model(cfg, obs_shape, n_actions, cfg.device)
    target = copy.deepcopy(online)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr, eps=1.5e-4)
    evaluator = Evaluator(cfg, logger, online, n_actions)
    actor = Policy(cfg, online, n_actions, seed=cfg.seed)

    obs = envs.reset()
    Buffer = PrioritizedReplayBuffer if cfg.per else ReplayBuffer
    kw = dict(alpha=cfg.per_alpha) if cfg.per else {}
    buffer = Buffer(cfg.buffer_size, obs_shape, obs.dtype, seed=cfg.seed, **kw)
    nsteps = [NStepAccumulator(cfg.n_step, cfg.gamma) for _ in range(envs.n)]
    eps_steps = int(cfg.eps_fraction * cfg.steps)
    step, updates_due, last_target, last_log = 0, 0.0, 0, 0

    while not logger.done(step, cfg.steps):
        eps = linear_schedule(cfg.eps_start, cfg.eps_end, eps_steps, step)
        actions = actor.act(obs, eps=eps)
        next_obs, rews, terms, truncs, infos = envs.step(actions)
        step += envs.n
        for i in range(envs.n):
            # on a time-limit cut (truncated) the game did not really end: bootstrap from its last frame
            real_next = infos[i].get("final_obs", next_obs[i])
            for tr in nsteps[i].push(obs[i], actions[i], rews[i], real_next, terms[i], truncs[i]):
                buffer.add(*tr)
            if "episode" in infos[i]:
                logger.episode(step, infos[i]["episode"]["r"], infos[i]["episode"]["l"])
        obs = next_obs

        if step >= cfg.learning_starts:
            updates_due += envs.n / cfg.train_freq  # one update per `train_freq` env steps
            while updates_due >= 1:
                updates_due -= 1
                beta = linear_schedule(cfg.per_beta, 1.0, cfg.steps, step) if cfg.per else 0.0
                batch = buffer.sample(cfg.batch_size, beta=beta)
                loss, td, q_mean = dqn_loss(online, target, batch, cfg.double)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                buffer.update_priorities(batch["idx"], td)
            if step - last_log >= 2000:
                last_log = step
                logger.loss(step, loss=loss.item(), q_mean=q_mean, eps=eps)
        if step - last_target >= cfg.target_update:
            last_target = step
            target.load_state_dict(online.state_dict())
        evaluator(step)
        logger.maybe_print(step)
    envs.close()
    return online, obs_shape, n_actions
