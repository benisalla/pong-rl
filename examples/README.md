# Examples

Short scripts that use `pong_rl` from Python, in the order you would explore the repo. Run them
from the repository root on a CPU; outputs go to `examples/output/` (git-ignored).

| script | what it shows | time (1 thread) |
|---|---|---|
| [`01_play_small_pong.py`](01_play_small_pong.py) | the game, a random and a scripted player, the preprocessing figure, a GIF | ~20 s |
| [`02_train_a2c.py`](02_train_a2c.py) | train A2C from pixels to a winning policy | ~3 min |
| [`03_karpathy_policy_gradient.py`](03_karpathy_policy_gradient.py) | Karpathy's "fake label" gradient = autograd, then a short pg run | ~2 min |
| [`04_dqn_with_tricks.py`](04_dqn_with_tricks.py) | double + dueling DQN with prioritized replay and 3-step returns | ~3 min |
| [`05_evaluate_and_gif.py`](05_evaluate_and_gif.py) | load a checkpoint, score it on 30 games, save a GIF | ~20 s |
| [`06_ale_pong_smoke_test.py`](06_ale_pong_smoke_test.py) | ALE Pong headless, Karpathy preprocessing on a real frame, a short PPO run | ~2 min |

```bash
python examples/01_play_small_pong.py
python examples/02_train_a2c.py && python examples/05_evaluate_and_gif.py
```
