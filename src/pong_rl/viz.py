"""Figures: learning curves, the preprocessing pipeline, and GIFs of an agent playing."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import load_log, smooth  # noqa: E402

COLORS = {"pg": "#8c564b", "reinforce": "#9467bd", "a2c": "#2ca02c", "ppo": "#1f77b4",
          "dqn": "#ff7f0e", "double-dqn": "#d62728", "dueling-dqn": "#e377c2"}


def plot_runs(run_dirs, out_path: str, title: str = "", window: int = 30, x: str = "steps") -> None:
    """Training score (smoothed, thin) and greedy evaluation score (dots + line) vs env steps or minutes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in run_dirs:
        log = load_log(d)
        name = log["config"].get("label") or log["config"]["algo"]
        c = COLORS.get(log["config"]["algo"])
        ep = np.array(log["episodes"]) if log["episodes"] else np.zeros((0, 4))
        ev = log["evals"]
        xi = 0 if x == "steps" else 3
        if len(ep):
            xs = ep[:, xi] if x == "steps" else ep[:, 3] / 60
            axes[0].plot(xs, smooth(ep[:, 1], window), color=c, label=name, lw=1.5)
        if ev:
            xs = [e["step"] if x == "steps" else e["time"] / 60 for e in ev]
            axes[1].plot(xs, [e["mean"] for e in ev], "o-", color=c, label=name, ms=3, lw=1.5)
    xlabel = "environment steps" if x == "steps" else "wall-clock minutes (1 CPU thread)"
    axes[0].set_title(f"training score (moving average of {window} games)")
    axes[1].set_title("greedy evaluation score")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("points won - points lost")
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8, loc="lower right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_preprocessing(out_path: str, env: str = "mini", steps: int = 30, seed: int = 0) -> None:
    """Raw frame, binary frame, frame difference and a 4-frame stack, side by side."""
    from .envs import make_env, preprocess

    e = make_env(env, obs="stack", stack=4, seed=seed)
    e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        stack, *_ = e.step(int(rng.integers(3)))
    raw = e.game.render_rgb(4) if env == "mini" else e.game.env.unwrapped.ale.getScreenRGB()
    f = preprocess(raw if env == "ale" else e.game.render(), env)
    diff = stack[-1].astype(int) - stack[-2].astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    axes[0].imshow(raw)
    axes[0].set_title("raw frame")
    axes[1].imshow(f, cmap="gray")
    axes[1].set_title(f"preprocessed {f.shape[0]}x{f.shape[1]} binary")
    axes[2].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("--obs diff: f_t - f_(t-1)  (red +1, blue -1)")
    grid = np.ones((2 * f.shape[0] + 1, 2 * f.shape[1] + 1)) * 0.5  # 2x2 grid with grey separators
    for k, fr in enumerate(stack[-4:]):
        r, c = divmod(k, 2)
        r0, c0 = r * (f.shape[0] + 1), c * (f.shape[1] + 1)
        grid[r0:r0 + f.shape[0], c0:c0 + f.shape[1]] = fr
    axes[3].imshow(grid, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("--obs stack: last 4 frames")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_gif(frames, path: str, fps: int = 30) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
