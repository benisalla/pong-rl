import numpy as np
import torch

from pong_rl.buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer, SumTree
from pong_rl.models import ActorCritic, ConvEncoder, KarpathyMLP, QNetwork
from pong_rl.utils import discount_rewards, gae


def test_discount_rewards_resets_at_points():
    r = [0, 0, 1, 0, -1]
    g = discount_rewards(r, 0.5, reset_on_point=True)
    assert np.allclose(g, [0.25, 0.5, 1.0, -0.5, -1.0])
    g2 = discount_rewards(r, 0.5, reset_on_point=False)
    assert np.isclose(g2[2], 1.0 + 0.5 * (0 + 0.5 * -1))


def test_gae_limits():
    T, N = 6, 2
    torch.manual_seed(0)
    r, v = torch.randn(T, N), torch.randn(T, N)
    d = torch.zeros(T, N)
    d[2, 0] = 1.0
    last = torch.randn(N)
    adv0, _ = gae(r, v, d, last, 0.9, 0.0)  # lambda 0: one-step TD error
    nv = torch.cat([v[1:], last[None]])
    assert torch.allclose(adv0, r + 0.9 * nv * (1 - d) - v, atol=1e-6)
    _, ret1 = gae(r, v, d, last, 0.9, 1.0)  # lambda 1: bootstrapped Monte Carlo return
    exp = torch.zeros(T, N)
    run = last.clone()
    for t in reversed(range(T)):
        run = r[t] + 0.9 * run * (1 - d[t])
        exp[t] = run
    assert torch.allclose(ret1, exp, atol=1e-5)


def test_models_shapes_small_and_nature():
    assert ConvEncoder((4, 20, 20)).kind == "small"
    assert ConvEncoder((4, 80, 80)).kind == "nature"
    x = torch.zeros(5, 4, 80, 80, dtype=torch.uint8)
    logits, v = ActorCritic((4, 80, 80), 3)(x)
    assert logits.shape == (5, 3) and v.shape == (5,)
    assert QNetwork((1, 20, 20), 3)(torch.zeros(2, 1, 20, 20, dtype=torch.int8)).shape == (2, 3)
    assert KarpathyMLP(400)(torch.zeros(7, 1, 20, 20)).shape == (7,)


def test_dueling_head_advantages_have_zero_mean():
    q = QNetwork((4, 20, 20), 3, dueling=True)
    x = torch.randint(0, 2, (4, 4, 20, 20), dtype=torch.uint8)
    z = q.encoder(x)
    a = q.advantage(z)
    out = q(x)
    assert torch.allclose(out.mean(1, keepdim=True), q.value(z), atol=1e-5)
    assert torch.allclose(out - q.value(z), a - a.mean(1, keepdim=True), atol=1e-5)


def test_sumtree_samples_proportionally():
    t = SumTree(4)
    for i, p in enumerate([1.0, 0.0, 3.0, 0.0]):
        t.update(i, p)
    assert t.total == 4.0
    rng = np.random.default_rng(0)
    idx = [t.find(u) for u in rng.random(4000) * t.total]
    counts = np.bincount(idx, minlength=4) / 4000
    assert counts[1] == counts[3] == 0 and abs(counts[2] - 0.75) < 0.03


def test_replay_buffers_roundtrip_and_priorities():
    for B in (ReplayBuffer, PrioritizedReplayBuffer):
        buf = B(10, (1, 2, 2), np.int8, seed=0)
        for i in range(15):  # wraps around
            buf.add(np.full((1, 2, 2), i % 3, np.int8), i % 3, float(i), np.zeros((1, 2, 2), np.int8), False, 0.99)
        assert len(buf) == 10
        b = buf.sample(8, beta=0.5)
        assert b["obs"].shape == (8, 1, 2, 2) and b["weights"].shape == (8,)
        assert np.all(b["obs"][:, 0, 0, 0] == b["actions"])
    buf.update_priorities(np.arange(10), np.r_[np.zeros(9), 100.0])
    assert (buf.sample(64)["idx"] == 9).mean() > 0.8  # the surprising transition dominates


def test_nstep_accumulator():
    acc = NStepAccumulator(3, 0.5)
    out = []
    for t in range(4):
        out += acc.push(t, t, 1.0, t + 1, False)
    assert len(out) == 2
    s, a, R, s_next, done, g = out[0]
    assert (s, s_next) == (0, 3) and np.isclose(R, 1 + 0.5 + 0.25) and np.isclose(g, 0.125) and not done
    out = acc.push(4, 4, 1.0, 5, True)  # episode end flushes the queue with shorter tails
    assert [o[0] for o in out] == [2, 3, 4] and all(o[4] for o in out)
    assert np.isclose(out[-1][2], 1.0) and np.isclose(out[-1][5], 0.5)
