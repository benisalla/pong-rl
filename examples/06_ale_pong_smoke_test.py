"""06 - The real thing: ALE Pong from gymnasium + ale-py, same code, a few thousand frames.

Checks that ``ALE/Pong-v5`` runs headless, draws the Karpathy preprocessing on a real frame
(210x160x3 -> 80x80 binary), and runs PPO for 4096 agent steps (16k frames with frameskip 4)
to show the pipeline works and the losses move. Winning ALE Pong takes millions of frames:
that run is in notebooks/train_colab_kaggle.ipynb, on a GPU.

    python examples/06_ale_pong_smoke_test.py         # ~2 min
"""
from pong_rl.config import make_config
from pong_rl.metrics import load_log
from pong_rl.train import run
from pong_rl.viz import plot_preprocessing

plot_preprocessing("examples/output/06_ale_preprocessing.png", env="ale", steps=60)
cfg = make_config("ppo", env="ale", gamma=0.99, lr=2.5e-4, n_envs=4, n_steps=128, steps=4096, eval_every=0,
                  final_eval_episodes=1, out="examples/output/06_ale_ppo")
res = run(cfg)
for row in load_log(cfg.out)["losses"]:
    print(f"step {row['step']:>5,d}  pi_loss {row['pi_loss']:+.4f}  v_loss {row['v_loss']:.4f}  "
          f"entropy {row['entropy']:.3f}  approx_kl {row['approx_kl']:.4f}")
print(f"one ALE game after {cfg.steps} steps: {res['mean']:+.0f} (an untrained agent loses about -21)")
