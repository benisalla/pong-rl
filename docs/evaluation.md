# Evaluation

One function, `evaluate.evaluate(policy, episodes)`, scores every method the same way:

* **score of a game** = points won - points lost: in [-5, 5] on the small game (first to 5),
  [-21, 21] on ALE Pong (first to 21). > 0 means the agent beats the opponent on average.
* **N games on fixed serves**: copy i is reset with seed 10 000 + i, so every method faces exactly
  the same serves and opponent noise. Training uses other seeds.
* **reported**: mean, std, min, max, win rate (games won), mean game length (decisions).
* **the policy used**: each agent plays its own policy. The policy-gradient methods (`pg`,
  `reinforce`, `a2c`, `ppo`) sample a ~ pi(.|s), the stochastic policy they optimise; the DQN family
  plays argmax_a Q(s, a) with epsilon = 0. `pong-eval --greedy` forces argmax for everyone.

**Why sample for the policy-gradient agents.** Their objective is the expected return of the
stochastic policy, and that is what the training curve shows. Making them greedy can go either
way. Early in training it hurt: in a 3-minute probe, a half-trained `pg` scored about 1 point lower
greedy than sampled, because an agent that only has UP and DOWN can never stay still and jitters
around the ball. Once trained it helps. On the final small-game runs (30 games):

| method | sampled (default) | `--greedy` |
|---|---|---|
| `pg` | +2.80 | +3.33 |
| `reinforce` | +4.23 | +4.83 |
| `a2c` | +2.67 | +3.97 |
| `ppo` | +4.17 | +4.60 |

The README table reports the sampled scores, so all four are measured as the policy they optimise.

During training an evaluation runs every `--eval-every` steps (10 games), and at the end on
`--final-eval-episodes` games (30). The README's "final score" column is that last one.

```bash
pong-eval runs/mini/ppo/model.pt --episodes 30          # mean +- std, min, max, win rate
pong-eval runs/mini/ppo/model.pt --greedy
pong-play runs/mini/ppo/model.pt --gif assets/ppo.gif    # watch it
pong-compare runs/mini/* --plot assets/curves.png --markdown assets/table.md
```

## The curves (`viz.plot_runs`)

Left: training score, a moving average over 30 games, against environment steps (exploration
noise included: epsilon-greedy for DQN, sampling for the others). Right: the periodic 10-game
evaluation. `--plot-time` draws the same against wall-clock minutes, which is the fairer axis on a
CPU: a DQN step costs ~4x a PPO step.

## Comparing methods honestly

One seed per method, one CPU thread, budgets of 2-6 minutes each. These are learning-speed
demonstrations on a small game, not benchmarks: with other seeds the ranking between close
methods can change. What is robust is the order of magnitude: bootstrapping methods (A2C, PPO,
DQN) learn in a few hundred thousand steps, Monte Carlo methods (pg, REINFORCE) need millions.
