"""Train several methods one after the other with their default settings, then compare them.

    python scripts/compare_all.py --algos ppo a2c --out-root runs/mini
    python scripts/compare_all.py --report-only --out-root runs/mini     # table + figures only

Every method uses the same env, the same preprocessing, the same seed and the same evaluation
(``evaluate.evaluate``, 30 games on fixed serves). The table goes to stdout and to
``assets/mini_pong_results.md``, the curves to ``assets/mini_pong_curves*.png``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pong_rl.compare import main as compare_main  # noqa: E402
from pong_rl.config import ALGOS, make_config  # noqa: E402
from pong_rl.train import run  # noqa: E402

ORDER = ["pg", "reinforce", "a2c", "ppo", "dqn", "double-dqn", "dueling-dqn"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=list(ALGOS))
    p.add_argument("--env", default="mini")
    p.add_argument("--out-root", default="runs/mini")
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per", action="store_true", help="prioritized replay for the DQN family")
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--assets", default="assets")
    a = p.parse_args()
    if not a.report_only:
        for algo in a.algos:
            dqn = "dqn" in algo
            cfg = make_config(algo, env=a.env, seed=a.seed, out=os.path.join(a.out_root, algo),
                              max_minutes=a.max_minutes, per=a.per if dqn else None,
                              n_step=a.n_step if dqn else None)
            run(cfg)
    runs = [os.path.join(a.out_root, x) for x in ORDER if os.path.exists(os.path.join(a.out_root, x, "log.json"))]
    game = "small" if a.env == "mini" else "ALE"
    compare_main(runs + ["--markdown", os.path.join(a.assets, f"{a.env}_pong_results.md"),
                         "--plot", os.path.join(a.assets, f"{a.env}_pong_curves.png"),
                         "--plot-time", os.path.join(a.assets, f"{a.env}_pong_curves_time.png"),
                         "--title", f"every method on the {game} Pong (one seed, one CPU thread)"])


if __name__ == "__main__":
    main()
