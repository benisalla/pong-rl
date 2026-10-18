# Training

```bash
pong-train --algo ppo                                # small game, default budget (~10 min, 1 thread)
pong-train --algo pg                                 # Karpathy, frame differences (the notebook's method)
pong-train --algo dqn --per --n-step 3               # any DQN upgrade is a flag
pong-train --algo a2c --obs diff --steps 200000      # every method accepts --obs diff|stack
pong-train --algo ppo --env ale --gamma 0.99 --steps 4096 --n-envs 4    # ALE smoke run
pong-train --algo ppo --help                         # every knob
```

A run writes `runs/<algo>/` (or `--out`): `model.pt` (config + weights), `log.json` (every
training game, every evaluation, losses), `curve.png`. `--max-minutes` stops a run early on a
shared machine; the log records the steps actually done.

## One flat config (`config.py`)

`Config` holds every knob; `ALGO_DEFAULTS` overrides a few per method; your flags override both.
**Why flat.** `pong-train --help` lists everything, the checkpoint stores the whole thing, and
`pong-eval` / `pong-play` rebuild exactly the same network and environment from it.

Defaults for the small game (all on one CPU thread):

| method | steps | parallel games | lr / optimiser | per update | other |
|---|---|---|---|---|---|
| `pg` | 2 M | 1 | 3e-3 RMSprop | 2 games | MLP 200 hidden, `--obs diff` |
| `reinforce` | 1.2 M | 8 | 1e-3 Adam | 2 games | baseline V(s), entropy 0.01 |
| `a2c` | 400 k | 16 | 7e-4 RMSprop, linear decay | 16 x 5 steps | n-step returns (GAE lambda 1) |
| `ppo` | 400 k | 8 | 1e-3 Adam, linear decay | 8 x 64 steps, 4 epochs x 2 minibatches | GAE 0.95, clip 0.1 |
| `dqn` | 180 k | 8 (acting) | 2.5e-4 Adam | batch 64 every 8 steps | buffer 50 k, target every 1 k, eps 1 -> 0.02 over 10% |
| `double-dqn` | 180 k | 8 | same | same | `--double` |
| `dueling-dqn` | 180 k | 8 | same | same | `--double --dueling` |

Shared: gamma = 0.97, seed 0, 4-frame stack, small CNN.

## Choices that mattered (measured, not guessed)

Short probes (< 3 min each, ~40-300 k steps) before the real runs:

* **gamma 0.99 -> 0.97.** On the small game a point is decided ~10-20 decisions after the move that
  mattered. With 0.99 PPO's training score went from -3.5 to -2.5 in 120 k steps; with 0.97 it
  reached -0.4 in 140 k. A shorter horizon means lower-variance returns for the same credit
  assignment. ALE runs keep 0.99 (a point takes ~50-100 decisions there).
* **Orthogonal init in the CNN.** With PyTorch's default init on sparse binary frames, PPO's
  entropy stayed at ln 3 = 1.0986 and the clip fraction at 0 for 150 k steps: the policy head got
  almost no gradient. Orthogonal init with gain sqrt(2) fixed it.
* **Frame skip 2 and 20x20 frames on the small game.** Each halves the compute per game; together
  ~4x more training per minute, which on a shared CPU decides whether a run takes 10 minutes or 40.
* **More, smaller updates for the Monte Carlo methods.** One update per 10 (pg) or 16 (REINFORCE)
  games gave a few hundred updates in a probe and no visible learning; 2 games per update moved them.
* **Vectorised acting for DQN.** One forward pass for 8 games instead of 8 passes for 1 game took DQN
  from ~190 to ~310 steps/s; the replay ratio (64 samples per 8 steps) is unchanged.

## Speed on this machine (shared 8-core CPU under heavy load, one thread per run)

| method | env steps / s (small game) | env steps / s (ALE smoke run) |
|---|---|---|
| `pg` | ~2 000 | ~380 |
| `reinforce` | ~1 400 | |
| `a2c` | ~1 100 | |
| `ppo` | ~700 | ~65 |
| DQN family | ~300 | ~37 |

The DQN family is the slowest per step (a gradient step every 8 steps on 64 samples) and the
fastest per sample; PPO sits in between; pg is cheap per step and needs many steps.

## ALE Pong: the real runs (Kaggle / Colab GPU)

`notebooks/train_colab_kaggle.ipynb` installs the package from GitHub and trains on `ALE/Pong-v5`
with `--device auto`. Budgets and *estimated* times on a Kaggle T4 / P100 (not measured here):

| run | frames (x4 steps) | settings | expected result | est. time |
|---|---|---|---|---|
| PPO | 10 M (2.5 M steps) | 8 envs x 128 steps, lr 2.5e-4, 4 epochs x 4 minibatches, gamma 0.99, clip 0.1 | ~ +18 to +21 | 2-3 h |
| double + dueling DQN, 3-step | 4 M (1 M steps) | buffer 100 k (2.6 GB RAM), lr 1e-4, batch 32 every 4 steps, target every 1 k, eps 1 -> 0.01 over 10% | ~ +15 to +20 | 3-4 h |
| Karpathy pg | 8 M (2 M steps) | lr 1e-3 RMSprop, batch 10 games, gamma 0.99, `--obs diff` | still below 0 at this budget; Karpathy reported ~3 days on CPU to beat the AI | 1-2 h (CPU-bound env) |

The notebook runs a short version of each by default (so "Run all" finishes) and says which
numbers to raise for the full budget. Environment stepping is single-process here, so expect the
GPU to wait on the emulator part of the time.

## Running all methods and comparing

```bash
python scripts/compare_all.py --algos pg reinforce a2c ppo dqn double-dqn dueling-dqn --out-root runs/mini
python scripts/compare_all.py --report-only --out-root runs/mini   # table + figures from existing runs
```
