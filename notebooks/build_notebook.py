"""Generate notebooks/train_colab_kaggle.ipynb from the package (run: python notebooks/build_notebook.py).

The notebook imports everything from ``pong_rl`` (installed from GitHub), shows the code of each
learning rule with ``inspect.getsource`` so the explanation sits next to the real implementation,
and trains the real ALE Pong runs on a GPU.
"""
from __future__ import annotations

import os

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Pong from pixels with every deep RL method: Kaggle / Colab notebook

This notebook goes with [`benisalla/pong-rl`](https://github.com/benisalla/pong-rl). It explains, step by
step and next to the real code, how an agent learns Pong **from pixels only**, then trains on
**ALE Pong** (`ALE/Pong-v5`) on a GPU, which a laptop CPU cannot do in reasonable time.

1. Setup (install the package, check the GPU and ALE)
2. Pixel preprocessing: from a 210x160 colour frame to an 80x80 binary image
3. Seeing motion: frame difference vs frame stacking
4. Policy gradient from pixels (Karpathy's derivation), REINFORCE with a baseline
5. Actor-critic: A2C, then PPO
6. DQN and its tricks (target network, replay, double, dueling, prioritized replay, n-step)
7. All seven methods on the small built-in Pong (minutes)
8. **The real runs: PPO and DQN on ALE Pong** (hours on a GPU)

Two games appear below. The **small Pong** (`env="mini"`, 40x40 numpy game, 20x20 after
preprocessing) is where every method learns in minutes. **ALE Pong** (`env="ale"`) is the Atari game,
where a winning agent needs millions of frames. Every number in this notebook says which one it comes from.
""")

md("""
## 1. Setup

The repository is private for now: on Kaggle add a secret `GITHUB_TOKEN` (Add-ons -> Secrets), on Colab
set it in the first cell, or upload the repo folder and `pip install -e` it. Once it is public the plain
`pip install git+https://github.com/benisalla/pong-rl` works. `ale-py >= 0.9` ships the Atari ROMs:
no ROM download, no licence step.
""")

code("""
import os, subprocess, sys
token = os.environ.get("GITHUB_TOKEN", "")
try:  # Kaggle secrets
    from kaggle_secrets import UserSecretsClient
    token = token or UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = f"git+https://{token + '@' if token else ''}github.com/benisalla/pong-rl.git"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", url], check=True)
""")

code("""
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

import pong_rl
from pong_rl.config import make_config
from pong_rl.envs import make_env, preprocess
from pong_rl.train import run
from pong_rl.evaluate import evaluate
from pong_rl.policy import load_policy
from pong_rl.play import play
from pong_rl.viz import plot_preprocessing, plot_runs, save_gif
from pong_rl.compare import summarize, markdown_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("pong_rl", pong_rl.__version__, "| torch", torch.__version__, "| device", DEVICE)
env = make_env("ale", obs="diff", seed=0)          # ALE/Pong-v5, headless
obs, _ = env.reset(seed=0)
print("ALE Pong observation after preprocessing + diff:", obs.shape, obs.dtype)
""")

md("""
## 2. Pixel preprocessing

An Atari frame is 210x160x3 bytes. Most of it is useless for Pong: the score bar, the borders, the
colours. Karpathy's `prepro` keeps only what matters:

1. **crop** rows 35..195 (drop the score and the bottom border): 160x160;
2. **downsample** by 2, keeping one pixel in two and one colour channel: 80x80;
3. **erase the background** (values 144 and 109), set paddles and ball to 1: a binary image.

6400 numbers instead of 100 800, and every one of them is meaningful. Below: the real function from the package.
""")

code("""
print(inspect.getsource(preprocess))
plot_preprocessing("ale_preprocessing.png", env="ale", steps=60)
display(Image("ale_preprocessing.png"))
""")

md("""
## 3. Seeing motion: `--obs diff` vs `--obs stack`

A single frame tells where the ball **is**, not where it **goes**, and the right move depends on
the direction. Two classic fixes:

* **frame difference** (Karpathy): x_t = f_t - f_{t-1}. The ball becomes a +1 blob in front of a -1
  blob: position, direction and speed in one channel. Still objects disappear (a paddle that does not move).
* **frame stack** (DQN): the last 4 frames as 4 channels. Nothing is lost; the network learns motion itself,
  at 4x the input size.

Both live in `PixelObs`; every method accepts either with `obs="diff"` or `obs="stack"`.
""")

code("""
from pong_rl.envs.wrappers import PixelObs
print(inspect.getsource(PixelObs._obs))
plot_preprocessing("mini_preprocessing.png", env="mini", steps=30)
display(Image("mini_preprocessing.png"))   # the small game: 40x40 -> 20x20, same idea
""")

md("""
## 4. Policy gradient from pixels

**The policy.** Karpathy's network maps the 6400-pixel difference image to one number,
p = sigma(W2 relu(W1 x)) = P(UP | x). The agent samples UP with probability p, else DOWN.

**The gradient.** We want to maximise J = E[sum_t r_t]. The score-function identity gives

    grad J = E[ sum_t A_t grad log pi(a_t | x_t) ]

where A_t says how good the action turned out (here: the discounted return). For a Bernoulli policy,
with y_t = 1 if the action was UP,

    log pi(a_t | x_t) = y_t log p_t + (1 - y_t) log(1 - p_t)
    d log pi / d logit = y_t - p_t

This is Karpathy's "fake label" trick: treat the sampled action as if it were the correct label of a
classifier, compute the usual classification gradient (y - p), and scale it by A_t. Good outcomes
make the taken actions more likely, bad outcomes less likely.

**Two Pong-specific details.** (1) The discounted sum restarts at every point: each non-zero reward
ends a rally, and an action is credited only with the point it helped decide. (2) The returns of a
game are standardised, so about half the actions are pushed up and half down: a crude baseline.

In PyTorch that whole derivation is one line: `loss = -(A * log_pi).sum()`. The check below
shows autograd gives exactly Karpathy's hand-written gradient.
""")

code("""
from pong_rl.algorithms.pg import pg_loss
from pong_rl.utils import discount_rewards
from pong_rl.policy import UP, DOWN
print(inspect.getsource(pg_loss))
print(inspect.getsource(discount_rewards))

logits = torch.randn(6, requires_grad=True)
actions = torch.tensor([UP, DOWN, UP, UP, DOWN, DOWN]); adv = torch.randn(6)
pg_loss(logits, actions, adv).backward()
y = (actions == UP).float()
print("autograd          :", (-logits.grad).numpy().round(4))
print("Karpathy A (y - p):", (adv * (y - torch.sigmoid(logits))).detach().numpy().round(4))
print("discounted, reset at points:", discount_rewards([0, 0, 1, 0, 0, -1], 0.9))
""")

md("""
**REINFORCE with a baseline** keeps the same Monte Carlo gradient but (a) uses a CNN and all three
actions (NOOP, UP, DOWN) and (b) subtracts a learned value: A_t = G_t - V(s_t). Subtracting anything
that does not depend on the action leaves the gradient unbiased; subtracting what you *expected* to
get removes most of the noise. V is trained by regression on G_t, and an entropy bonus keeps the
policy from collapsing too early.
""")

code("""
from pong_rl.algorithms import reinforce
src = inspect.getsource(reinforce.train)
print(src[src.index("        if len(batch) >= cfg.batch_episodes"):src.index("            pl, vl, en")])
""")

md("""
## 5. Actor-critic: A2C and PPO

Monte Carlo methods wait for the end of a game and their targets are noisy. **A2C** learns from short
rollouts (16 games x 5 steps) and lets the critic fill in the rest:

    G_t = r_t + g r_{t+1} + ... + g^{n-1} r_{t+n-1} + g^n V(s_{t+n}),      A_t = G_t - V(s_t)

Less variance, some bias (V is imperfect), and an update every 5 steps instead of every game.

**PPO** collects the same kind of rollout but reuses it for several epochs of minibatch updates,
which is where its sample efficiency comes from. To stop the reused data from dragging the policy too
far from the one that collected it, it clips the probability ratio:

    rho_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
    L = -mean min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t)

Once a sample's ratio has moved more than eps in the direction its advantage pushes, its gradient is
zero. Advantages come from GAE: an exponentially weighted mix (lambda) of 1-step, 2-step, ... advantages,
trading bias (lambda = 0) for variance (lambda = 1).
""")

code("""
from pong_rl.algorithms.ppo import ppo_loss
from pong_rl.utils import gae
print(inspect.getsource(gae))
print(inspect.getsource(ppo_loss))
""")

md("""
## 6. DQN and its tricks

DQN learns Q(s, a), the return of taking a in s and playing well afterwards, by regression on a
bootstrapped target, and acts epsilon-greedily on it:

    y = r + g^n (1 - done) max_a' Q_target(s', a')

Why each trick is there:

* **Replay buffer**: consecutive frames are almost identical; training on them in order is like
  training a classifier on one class at a time. Random old transitions decorrelate the batch, and
  each transition is reused many times.
* **Target network**: a frozen copy refreshed every few thousand steps, so the target does not move
  with every gradient step.
* **Huber loss + gradient clipping**: one surprising point cannot blow up the update.
* **Double DQN**: max over noisy estimates is biased upwards; the online net *chooses* a', the
  target net *evaluates* it.
* **Dueling head**: Q = V(s) + A(s, a) - mean_a A(s, a); in most Pong frames the action does not
  matter yet, and V learns that once for all actions.
* **Prioritized replay**: replay transitions with large TD error more often, correct the bias with
  importance weights.
* **n-step returns**: the reward of a point travels back n steps per update instead of one.
""")

code("""
from pong_rl.algorithms.dqn import td_targets, dqn_loss
from pong_rl.buffers import PrioritizedReplayBuffer, NStepAccumulator
from pong_rl.models import QNetwork
print(inspect.getsource(td_targets))
print(inspect.getsource(QNetwork.forward))
print(inspect.getsource(PrioritizedReplayBuffer.sample))
""")

md("""
## 7. All seven methods on the small Pong

Same game, same preprocessing, same evaluation (mean score over 30 games on fixed serves, score =
points won - points lost, first to 5, so in [-5, 5]; a random agent scores about -3.3). These are the
repository's default budgets. On a Kaggle CPU/GPU session each takes a few minutes; set
`SMALL_GAME_FRACTION = 1.0` for the full budgets used in the README, or keep it small for a quick tour.
""")

code("""
SMALL_GAME_FRACTION = 0.25   # 1.0 = the README budgets (10-20 min per method on one CPU thread)
ALGOS = ["pg", "reinforce", "a2c", "ppo", "dqn", "double-dqn", "dueling-dqn"]
small_runs = []
for algo in ALGOS:
    base = make_config(algo)
    extra = dict(per=True, n_step=3) if algo == "dueling-dqn" else {}
    cfg = make_config(algo, steps=int(base.steps * SMALL_GAME_FRACTION), out=f"runs/mini/{algo}",
                      device="cpu", **extra)   # small nets: the CPU is as fast as the GPU here
    run(cfg, plot=False)
    small_runs.append(cfg.out)
display(Markdown(markdown_table([summarize(r) for r in small_runs])))
plot_runs(small_runs, "mini_curves.png", title="small Pong, every method")
display(Image("mini_curves.png"))
""")

md("""
## 8. The real runs: ALE Pong on a GPU

Now the Atari game: 80x80 frames, frame skip 4, the Nature-DQN CNN, gamma = 0.99, first to 21.
A random agent scores about -20.5. Published curves reach +18..+21 after ~10 M frames for PPO and
~4 M frames for DQN variants.

**Budgets and estimated times** (Kaggle T4/P100; estimates, environment stepping runs on the CPU in one
process, so the GPU sometimes waits on the emulator):

| run | agent steps (x4 frames) | expected | est. time |
|---|---|---|---|
| PPO | 2.5 M (10 M frames) | +18..+21 | 2-3 h |
| dueling double DQN, 3-step | 1 M (4 M frames) | +15..+20 | 3-4 h |
| Karpathy pg | 2 M (8 M frames) | still below 0 (it needs ~days) | 1-2 h |

`FULL = False` runs a short version of each (about 30-40 min in total) so "Run all" finishes; set
`FULL = True` for the real budgets (a Kaggle session lasts up to 12 h; `max_minutes` stops a run
cleanly before that and still saves the model and log).
""")

code("""
FULL = False
ale_common = dict(env="ale", gamma=0.99, device=DEVICE, encoder="nature", eval_every=100_000 if FULL else 50_000,
                  eval_episodes=3, final_eval_episodes=5 if FULL else 2)
ppo_cfg = make_config("ppo", steps=2_500_000 if FULL else 150_000, n_envs=8, n_steps=128, minibatches=4,
                      epochs=4, lr=2.5e-4, clip=0.1, ent_coef=0.01, max_minutes=200 if FULL else 15,
                      out="runs/ale/ppo", **ale_common)
res_ppo = run(ppo_cfg)
display(Image("runs/ale/ppo/curve.png"))
""")

code("""
dqn_cfg = make_config("dueling-dqn", n_step=3, steps=1_000_000 if FULL else 100_000, n_envs=4,
                      buffer_size=100_000 if FULL else 30_000, batch_size=32, train_freq=4,
                      learning_starts=20_000 if FULL else 5_000, target_update=1_000, lr=1e-4,
                      eps_end=0.01, eps_fraction=0.1, max_minutes=240 if FULL else 15,
                      out="runs/ale/dueling-dqn", **ale_common)
res_dqn = run(dqn_cfg)
display(Image("runs/ale/dueling-dqn/curve.png"))
""")

code("""
pg_cfg = make_config("pg", steps=2_000_000 if FULL else 100_000, lr=1e-3, batch_episodes=10, obs="diff",
                     max_minutes=120 if FULL else 8, out="runs/ale/pg", **ale_common)
res_pg = run(pg_cfg)
""")

md("""
### Compare, evaluate, watch

The table below is the ALE comparison of this session (fill the README's ALE section from it after a
`FULL = True` run). The GIF shows the PPO agent (green paddle, right).
""")

code("""
ale_runs = ["runs/ale/pg", "runs/ale/ppo", "runs/ale/dueling-dqn"]
display(Markdown(markdown_table([summarize(r) for r in ale_runs])))
plot_runs(ale_runs, "ale_curves.png", title="ALE Pong")
display(Image("ale_curves.png"))
policy = load_policy("runs/ale/ppo/model.pt", device=DEVICE)
frames, scores = play(policy, games=1, max_frames=1500, every=2)
save_gif(frames, "ppo_ale_pong.gif", fps=30)
print("score of this game:", scores)
display(Image("ppo_ale_pong.gif"))
""")

md("""
### What to look at

* **Sample efficiency vs compute.** DQN reuses every transition many times from replay and needs the
  fewest frames; PPO needs more frames but each is cheap and it parallelises over environments; the
  Monte Carlo pg needs by far the most.
* **diff vs stack.** Rerun any method with `obs="diff"`: one channel instead of four, a smaller input,
  but a still paddle becomes invisible.
* **Sticky actions.** `sticky=0.25` (the `v5` default) makes the game stochastic: a policy that
  memorised a winning sequence breaks, a policy that reacts to the ball does not.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
}
out = os.path.join(os.path.dirname(__file__), "train_colab_kaggle.ipynb")
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
