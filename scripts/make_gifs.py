"""Regenerate the README's before/after GIFs from the trained checkpoints in runs/mini/.

    python scripts/make_gifs.py                    # needs runs/mini/<algo>/model.pt (scripts/compare_all.py)
    python scripts/make_gifs.py --ale-checkpoint runs/ale_smoke/ppo/model.pt

Writes:
* assets/before_after.gif  : an untrained agent next to the trained DQN agent, same game seed,
                             with the running score above each panel;
* assets/methods_grid.gif  : one panel per method (plus the untrained agent), same seed;
* assets/ale_smoke.gif     : a short ALE Pong clip of the barely trained PPO smoke-run agent.
All the panels of one GIF play in lock-step; a finished game freezes on its last frame.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pong_rl.config import make_config  # noqa: E402
from pong_rl.envs import make_env  # noqa: E402
from pong_rl.evaluate import EVAL_SEED  # noqa: E402
from pong_rl.policy import Policy, build_model, load_policy  # noqa: E402
from pong_rl.utils import set_seed  # noqa: E402

ORDER = ["pg", "reinforce", "a2c", "ppo", "dqn", "double-dqn", "dueling-dqn"]
FONT = ImageFont.load_default()


def untrained_policy(seed: int = 0) -> Policy:
    """A freshly initialised DQN network: what every agent looks like before training."""
    set_seed(seed)
    cfg = make_config("dqn")
    return Policy(cfg, build_model(cfg, (cfg.stack, 20, 20), 3), 3)


def play_lockstep(policies, seed: int, max_steps: int, scale: int, env: str = "mini"):
    """Play one game per policy on the same seed; returns per-step lists of (rgb frame, score) per panel."""
    envs, obs, done, score, last = [], [], [], [], []
    for p in policies:
        e = make_env(env, obs=p.cfg.obs, stack=p.cfg.stack, render=env == "ale")
        o, _ = e.reset(seed=seed)
        envs.append(e)
        obs.append(o)
        done.append(False)
        score.append([0, 0])
        last.append(None)
    frames = []
    for _ in range(max_steps):
        row = []
        for i, (p, e) in enumerate(zip(policies, envs)):
            if not done[i]:
                last[i] = e.render_rgb(scale) if env == "mini" else e.render_rgb()
                a = int(p.act(obs[i][None])[0])
                obs[i], r, term, trunc, _ = e.step(a)
                if r > 0:
                    score[i][0] += 1
                elif r < 0:
                    score[i][1] += 1
                done[i] = term or trunc
            row.append((last[i], tuple(score[i]), done[i]))
        frames.append(row)
        if all(done):
            break
    for e in envs:
        e.close()
    return frames


def panel(img: np.ndarray, title: str, score, finished: bool, width: int) -> Image.Image:
    """Frame with a caption bar above it: title and the running score."""
    im = Image.fromarray(img)
    if im.width != width:
        im = im.resize((width, int(im.height * width / im.width)), Image.NEAREST)
    bar = 30
    out = Image.new("RGB", (im.width, im.height + bar), (245, 245, 245))
    out.paste(im, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((4, 2), title, fill=(20, 20, 20), font=FONT)
    tag = "  final" if finished else ""
    colour = (30, 120, 30) if score[0] > score[1] else (170, 40, 40) if score[1] > score[0] else (60, 60, 60)
    d.text((4, 16), f"agent {score[0]} - {score[1]} opponent{tag}", fill=colour, font=FONT)
    return out


def compose(frames, titles, cols: int, width: int, every: int = 1, gap: int = 4):
    imgs = []
    for row in frames[::every] + [frames[-1]] * 12:  # hold the final frame a moment
        panels = [panel(f, t, s, d, width) for (f, s, d), t in zip(row, titles)]
        rows = (len(panels) + cols - 1) // cols
        w, h = panels[0].size
        canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
        for k, p in enumerate(panels):
            r, c = divmod(k, cols)
            canvas.paste(p, (c * (w + gap), r * (h + gap)))
        imgs.append(canvas.quantize(colors=64, dither=Image.Dither.NONE))
    return imgs


def save(imgs, path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path}: {len(imgs)} frames, {os.path.getsize(path) / 1e6:.2f} MB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/mini")
    p.add_argument("--assets", default="assets")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--best", default="dqn", help="trained agent for the before/after GIF")
    p.add_argument("--ale-checkpoint", default="runs/ale_smoke/ppo/model.pt")
    a = p.parse_args()
    torch.set_num_threads(1)

    # 1. before / after
    trained = load_policy(os.path.join(a.runs, a.best, "model.pt"))
    frames = play_lockstep([untrained_policy(), trained], a.seed, max_steps=700, scale=8)
    save(compose(frames, ["before training (random weights)", f"after training ({a.best})"], cols=2,
                 width=320), os.path.join(a.assets, "before_after.gif"), fps=25)

    # 2. one panel per method
    algos = [x for x in ORDER if os.path.exists(os.path.join(a.runs, x, "model.pt"))]
    pols = [untrained_policy()] + [load_policy(os.path.join(a.runs, x, "model.pt")) for x in algos]
    frames = play_lockstep(pols, a.seed + 1, max_steps=700, scale=4)
    save(compose(frames, ["untrained"] + algos, cols=4, width=160, every=2),
         os.path.join(a.assets, "methods_grid.gif"), fps=15)

    # 3. ALE smoke-run agent, honestly labelled
    if os.path.exists(a.ale_checkpoint):
        ale = load_policy(a.ale_checkpoint)
        frames = play_lockstep([ale], 0, max_steps=240, scale=1, env="ale")
        steps = ale.cfg.steps
        save(compose(frames, [f"ALE Pong, {ale.cfg.algo} after {steps:,} steps: barely trained"], cols=1,
                     width=320, every=2), os.path.join(a.assets, "ale_smoke.gif"), fps=15)


if __name__ == "__main__":
    main()
