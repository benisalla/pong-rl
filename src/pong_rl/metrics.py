"""Run logging: training episodes, periodic evaluations and losses, saved as ``log.json``.

The same file format for every method is what makes ``pong-compare`` and the curves trivial.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


class RunLogger:
    def __init__(self, out_dir: str, cfg: dict, print_every: float = 30.0):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "log.json")
        self.data = {"config": cfg, "episodes": [], "evals": [], "losses": [], "final": None}
        self.t0 = time.time()
        self.print_every, self._last_print = print_every, 0.0
        self.algo = cfg.get("algo", "?")
        self.max_seconds = 60.0 * cfg.get("max_minutes", 0.0)
        self.last_step = 0

    def done(self, step: int, total: int) -> bool:
        """The training loops' stop condition: step budget reached, or out of wall time."""
        self.last_step = step
        return step >= total or (self.max_seconds > 0 and self.elapsed() >= self.max_seconds)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def episode(self, step: int, ret: float, length: int) -> None:
        self.data["episodes"].append([int(step), float(ret), int(length), round(self.elapsed(), 2)])

    def loss(self, step: int, **values) -> None:
        self.data["losses"].append({"step": int(step), **{k: float(v) for k, v in values.items()}})

    def eval(self, step: int, res: dict) -> None:
        self.data["evals"].append({"step": int(step), "mean": res["mean"], "std": res["std"],
                                   "win_rate": res["win_rate"], "time": round(self.elapsed(), 2)})
        print(f"[{self.algo}] step {step:>9,d}  eval score {res['mean']:+.2f} +- {res['std']:.2f}  "
              f"win {res['win_rate']:.0%}  ({self.elapsed() / 60:.1f} min)", flush=True)
        self.save()

    def maybe_print(self, step: int) -> None:
        if self.elapsed() - self._last_print < self.print_every:
            return
        self._last_print = self.elapsed()
        eps = self.data["episodes"][-20:]
        mean = np.mean([e[1] for e in eps]) if eps else float("nan")
        last = self.data["losses"][-1] if self.data["losses"] else {}
        extra = "  ".join(f"{k} {v:.3g}" for k, v in last.items() if k != "step")
        print(f"[{self.algo}] step {step:>9,d}  train score (last 20) {mean:+.2f}  {extra}  "
              f"{step / max(self.elapsed(), 1e-9):.0f} steps/s", flush=True)

    def finish(self, res: dict, steps: int) -> None:
        self.data["final"] = {**res, "steps": int(steps), "time_s": round(self.elapsed(), 1)}
        self.save()

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f)


def load_log(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "log.json")
    with open(path) as f:
        return json.load(f)


def smooth(y, window: int = 20) -> np.ndarray:
    """Trailing moving average (same length as y), for noisy episode-score curves."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out
