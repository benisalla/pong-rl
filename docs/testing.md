# Tests

```bash
OMP_NUM_THREADS=1 pytest -q          # 34 tests, < 1 min on one CPU thread
```

| file | what it checks |
|---|---|
| `tests/test_envs.py` | small Pong frames, scores, seeding, paddle physics, frame skip, sticky actions; Karpathy preprocessing on a synthetic ALE frame; `diff` / `stack` shapes and values; vector env auto-reset; ALE Pong starts headless |
| `tests/test_components.py` | discounting with the Pong reset; GAE against its lambda = 0 (TD error) and lambda = 1 (n-step return) limits; network shapes for 20x20 and 80x80; dueling head identities; sum tree sampling proportions; replay buffers and priorities; n-step accumulator |
| `tests/test_algorithms.py` | `pg_loss` gradient == Karpathy's `A (y - p)`; PPO clip gives zero gradient outside the trust region; double DQN target uses the online argmax; **learns a tiny problem**: the MLP policy gradient and the CNN actor-critic each reach > 90% on a one-step pixel task (dot in the top half -> UP pays), DQN overfits a fixed batch (loss / 20) |
| `tests/test_training.py` | all seven methods train 600 steps, save, reload, evaluate; the four CLIs (`pong-train`, `pong-eval`, `pong-play`, `pong-compare`); config defaults and overrides; `--max-minutes` stops early |

**Why a pixel bandit for "it learns".** A full Pong run takes minutes; a one-step task where the
right action is written in the pixels is learnt in seconds and fails loudly if the sign of a
gradient, a log-probability or an advantage is wrong, which is what typically breaks an RL loop.
