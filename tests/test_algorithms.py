"""Losses match their equations, and every method learns a small pixel task."""
import numpy as np
import torch

from pong_rl.algorithms.dqn import dqn_loss, td_targets
from pong_rl.algorithms.pg import pg_loss
from pong_rl.algorithms.ppo import ppo_loss
from pong_rl.config import make_config
from pong_rl.models import ActorCritic, KarpathyMLP, QNetwork
from pong_rl.policy import DOWN, UP


def test_pg_loss_gradient_is_karpathys_fake_label_gradient():
    """d(-loss)/d logit = A (y - p), the hand-written backward pass of the original notebook."""
    torch.manual_seed(0)
    logits = torch.randn(6, requires_grad=True)
    actions = torch.tensor([UP, DOWN, UP, UP, DOWN, DOWN])
    adv = torch.randn(6)
    pg_loss(logits, actions, adv).backward()
    y = (actions == UP).float()
    karpathy = adv * (y - torch.sigmoid(logits))
    assert torch.allclose(-logits.grad, karpathy, atol=1e-6)


def test_ppo_clip_zeroes_gradient_outside_trust_region():
    cfg = make_config("ppo", ent_coef=0.0, vf_coef=0.0)
    torch.manual_seed(0)
    model = ActorCritic((1, 20, 20), 3)
    obs = torch.randint(0, 2, (4, 1, 20, 20), dtype=torch.int8)
    actions = torch.tensor([0, 1, 2, 0])
    with torch.no_grad():
        logp = torch.distributions.Categorical(logits=model(obs)[0]).log_prob(actions)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    # pretend the old policy was much less likely to pick the actions with positive advantage:
    # ratio >> 1 + eps for those -> clipped -> no gradient from them
    old = logp - torch.tensor([2.0, 2.0, 2.0, 2.0])
    loss, stats = ppo_loss(model, obs, actions, old, adv, torch.zeros(4), cfg)
    assert stats["clip_frac"] == 1.0
    loss.backward()
    g = sum(p.grad.abs().sum() for p in model.pi.parameters())
    assert g > 0  # negative-advantage samples with ratio > 1+eps are NOT clipped (min picks them)
    model.zero_grad()
    loss, _ = ppo_loss(model, obs[:2], actions[:2], old[:2], adv[:2], torch.zeros(2), cfg)
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.pi.parameters()) == 0


def test_double_dqn_target_uses_online_argmax():
    class Fixed(torch.nn.Module):
        def __init__(self, q):
            super().__init__()
            self.q = torch.tensor(q)

        def forward(self, x):
            return self.q.expand(len(x), -1)

    online, target = Fixed([[0.0, 5.0, 1.0]]), Fixed([[3.0, 1.0, 2.0]])
    batch = dict(next_obs=np.zeros((1, 1, 2, 2), np.int8), rewards=np.array([1.0], np.float32),
                 discounts=np.array([0.5], np.float32), dones=np.array([0.0], np.float32))
    assert np.isclose(td_targets(online, target, batch, double=False).item(), 1 + 0.5 * 3.0)
    assert np.isclose(td_targets(online, target, batch, double=True).item(), 1 + 0.5 * 1.0)
    batch["dones"] = np.array([1.0], np.float32)
    assert np.isclose(td_targets(online, target, batch, double=True).item(), 1.0)


def pixel_bandit(n, rng):
    """A one-step pixel task: a dot in the top half means UP pays +1, bottom half means DOWN pays +1."""
    top = rng.random(n) < 0.5
    x = np.zeros((n, 1, 20, 20), np.int8)
    rows = np.where(top, rng.integers(1, 9, n), rng.integers(11, 19, n))
    x[np.arange(n), 0, rows, rng.integers(2, 18, n)] = 1
    return x, np.where(top, UP, DOWN)


def test_pg_mlp_learns_pixel_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = KarpathyMLP(400, 64)
    opt = torch.optim.RMSprop(m.parameters(), lr=3e-3)
    for _ in range(150):
        x, best = pixel_bandit(64, rng)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.as_tensor(x))).numpy()
        a = np.where(rng.random(64) < p, UP, DOWN)
        r = np.where(a == best, 1.0, -1.0).astype(np.float32)
        opt.zero_grad()
        pg_loss(m(torch.as_tensor(x)), torch.as_tensor(a), torch.as_tensor(r)).backward()
        opt.step()
    x, best = pixel_bandit(500, rng)
    acc = (np.where(m(torch.as_tensor(x)).detach().numpy() > 0, UP, DOWN) == best).mean()
    assert acc > 0.9


def test_actor_critic_learns_pixel_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    m = ActorCritic((1, 20, 20), 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(150):
        x, best = pixel_bandit(64, rng)
        logits, v = m(torch.as_tensor(x))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        r = torch.as_tensor((a.numpy() == best).astype(np.float32))
        loss = -(dist.log_prob(a) * (r - v.detach())).mean() + 0.5 * (r - v).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, best = pixel_bandit(500, rng)
    acc = (m(torch.as_tensor(x))[0].argmax(1).numpy() == best).mean()
    assert acc > 0.9


def test_dqn_overfits_a_fixed_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    x, best = pixel_bandit(32, rng)
    batch = dict(obs=x, next_obs=x, actions=best, rewards=np.ones(32, np.float32),
                 discounts=np.full(32, 0.99, np.float32), dones=np.ones(32, np.float32),
                 weights=np.ones(32, np.float32))
    q = QNetwork((1, 20, 20), 3, dueling=True)
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    first = None
    for _ in range(200):
        loss, _, _ = dqn_loss(q, q, batch, double=True)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * first
