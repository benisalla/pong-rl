# Legacy notebook

[`RL_PONG.ipynb`](RL_PONG.ipynb) is the original coursework notebook, copied verbatim (outputs
included). It follows Andrej Karpathy's "Pong from Pixels" (2016):

1. the original numpy script (Python 2, `Pong-v0`), kept as a commented cell;
2. a Python 3 / gymnasium port of it (`ALE/Pong-v5`), also commented out;
3. two cells creating `ALE/Pong-v5` with `PassiveEnvChecker` / `OrderEnforcing` and rendering a frame;
4. a PyTorch rewrite: a 2-layer policy network (`NNPolicy`, 6400 -> 200 -> 1 sigmoid), the
   discounted-reward function with Pong's reset at every point (plus a small plot of it), and a
   training loop with RMSprop on frame differences. Its output shows the first two games of
   training (-20 and about -19), as expected at the start: Karpathy's agent needed days of CPU
   time to win.

Everything it does lives on in this repository as `pong-train --algo pg` (`src/pong_rl/algorithms/pg.py`),
with the same preprocessing (`envs/wrappers.py: preprocess`), the same frame difference
(`--obs diff`), the same discounting (`utils.discount_rewards`) and the same RMSprop update.

## Bugs found and how the package fixes them

| # | where | problem | effect | fix in `pong_rl` |
|---|---|---|---|---|
| 1 | PyTorch loss (last cell) | `loss = -(log(probs) * epa * epr).mean()` with `epa = 1` for UP and `0` for DOWN | every DOWN action contributes **zero** gradient, so the agent can only learn when UP was good or bad, never DOWN; the log-likelihood of a Bernoulli action is `y log p + (1 - y) log(1 - p)` | `algorithms.pg.pg_loss` uses the full Bernoulli log-likelihood (BCE with logits); `tests/test_algorithms.py` checks the gradient equals Karpathy's `A (y - p)` |
| 2 | PyTorch loss | `torch.log(sigmoid(x))` | `log(0) = -inf` once the policy saturates, NaN weights | work with logits: `binary_cross_entropy_with_logits` is stable |
| 3 | training cell | `PolicyNetwork(D, H, 1)` but the class is named `NNPolicy` | `NameError` on a fresh kernel (it only ran thanks to an older kernel state) | one `KarpathyMLP` class in `models.py` |
| 4 | discount cell | `dis_rewards = dis_rewards(rewards, gamma)` overwrites the function with its result, and the training loop calls `discount_rewards(...)`, which is not defined in the PyTorch part | `NameError` / `TypeError` on a fresh kernel | `utils.discount_rewards(rewards, gamma, reset_on_point)` |
| 5 | training cell | `batch_size = 10` is defined but the optimiser steps after **every** episode, and the loss is a mean over time steps | not Karpathy's update (sum over steps, accumulate 10 games); the step size depends on game length | `pg.train` sums over steps and accumulates gradients over `--batch-episodes` games before one RMSprop step |
| 6 | training cell | `render = True` with `render_mode="rgb_array"` | renders every frame into an unused array: slower for nothing | no rendering during training; `pong-play` makes GIFs |
| 7 | `prepro` | slicing gives a view, and `frame[...] = 0` writes into the environment's observation array | silent in-place modification of the env output | `preprocess` copies before editing |
| 8 | training cell | `while True:` with no stopping rule or evaluation | no way to tell whether it learns except reading prints | `--steps`, `--max-minutes`, periodic evaluation on fixed games, `log.json`, curves |
| 9 | original numpy cell | Python 2 (`iteritems`, `xrange`, `print` statement), `np.float` (removed in NumPy 1.24), the old 4-tuple `env.step` API, `policy_backward` reading the global `epx` | does not run on a current stack | Python 3.10+, gymnasium 5-tuple API, no globals |
| 10 | environment | `ALE/Pong-v5` defaults to sticky actions (p = 0.25) while Karpathy used deterministic `Pong-v0` | a harder, stochastic game than the one in the blog post (not a bug, but a silent change) | explicit `--sticky` flag (default 0.0), `--frameskip` |
| 11 | discount cell | `rewards` is assigned three times; only the last list is plotted | the plot shows a single -1 at the end, not the multi-point example written first | `tests/test_components.py::test_discount_rewards_resets_at_points` |
