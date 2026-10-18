"""``pong-compare``: one table and one figure from several run directories.

    pong-compare runs/pg runs/reinforce runs/a2c runs/ppo runs/dqn runs/double-dqn runs/dueling-dqn \
        --plot assets/mini_pong_curves.png --markdown assets/mini_pong_results.md

Columns: final greedy score (mean +- std over the final evaluation games), win rate, best
periodic evaluation, environment steps, first step where the periodic evaluation was > 0 (the
agent beats the opponent on average), wall-clock training time and parameter count.
"""
from __future__ import annotations

import argparse
import os

from .metrics import load_log
from .viz import plot_runs


def summarize(run_dir: str) -> dict:
    log = load_log(run_dir)
    cfg, fin, evals = log["config"], log.get("final") or {}, log["evals"]
    first_win = next((e["step"] for e in evals if e["mean"] > 0), None)
    return {
        "method": cfg["algo"] + (" +per" if cfg.get("per") else "")
        + (f" n={cfg['n_step']}" if cfg.get("n_step", 1) > 1 else ""),
        "obs": cfg["obs"], "env": cfg["env"],
        "score": fin.get("mean"), "std": fin.get("std"), "win_rate": fin.get("win_rate"),
        "best_eval": max((e["mean"] for e in evals), default=None),
        "steps": fin.get("steps", cfg["steps"]), "first_win_step": first_win,
        "minutes": (fin.get("train_time_s") or fin.get("time_s") or 0) / 60, "params": fin.get("params"),
        "episodes": fin.get("episodes"),
    }


def markdown_table(rows) -> str:
    head = ("| method | obs | final score | win rate | best eval | env steps | first eval > 0 at | "
            "train time | params |\n|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        score = f"**{r['score']:+.2f}** +- {r['std']:.2f}" if r["score"] is not None else "n/a"
        win = f"{r['win_rate']:.0%}" if r["win_rate"] is not None else "n/a"
        best = f"{r['best_eval']:+.1f}" if r["best_eval"] is not None else "n/a"
        fw = f"{r['first_win_step']:,}" if r["first_win_step"] else "never"
        params = f"{r['params'] / 1e3:.0f} k" if r["params"] else "n/a"
        lines.append(f"| `{r['method']}` | {r['obs']} | {score} | {win} | {best} | {r['steps']:,} | {fw} | "
                     f"{r['minutes']:.1f} min | {params} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare finished runs: markdown table + learning curves.")
    p.add_argument("runs", nargs="+")
    p.add_argument("--plot", default="", help="curves figure (x = env steps)")
    p.add_argument("--plot-time", default="", help="curves figure (x = wall-clock minutes)")
    p.add_argument("--markdown", default="", help="write the table to this file too")
    p.add_argument("--title", default="")
    a = p.parse_args(argv)
    runs = [r for r in a.runs if os.path.exists(os.path.join(r, "log.json"))]
    table = markdown_table([summarize(r) for r in runs])
    print(table)
    if a.markdown:
        os.makedirs(os.path.dirname(a.markdown) or ".", exist_ok=True)
        with open(a.markdown, "w") as f:
            f.write(table + "\n")
    if a.plot:
        plot_runs(runs, a.plot, a.title)
    if a.plot_time:
        plot_runs(runs, a.plot_time, a.title, x="time")


if __name__ == "__main__":
    main()
