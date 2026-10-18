"""Every method runs end to end for a few hundred steps; the CLIs work."""
import json
import os

import pytest

from pong_rl import compare, evaluate, play, train
from pong_rl.config import ALGOS, make_config
from pong_rl.policy import load_policy

TINY = dict(steps=600, eval_every=300, eval_episodes=2, final_eval_episodes=2, learning_starts=100,
            batch_size=8, buffer_size=500, target_update=100, batch_episodes=1, n_envs=2, n_steps=16)


@pytest.mark.parametrize("algo", ALGOS)
def test_every_method_trains_saves_and_reloads(algo, tmp_path):
    extra = dict(per=True, n_step=3) if algo == "dueling-dqn" else {}
    cfg = make_config(algo, out=str(tmp_path / algo), **TINY, **extra)
    res = train.run(cfg, plot=False)
    assert -5 <= res["mean"] <= 5
    log = json.load(open(tmp_path / algo / "log.json"))
    assert 600 <= log["final"]["steps"] < 700 and log["evals"]
    pol = load_policy(str(tmp_path / algo / "model.pt"))
    assert pol.cfg.algo == algo
    assert evaluate.evaluate(pol, episodes=2)["episodes"] == 2


def test_cli_train_eval_play_compare(tmp_path, capsys):
    out = str(tmp_path / "ppo")
    train.main(["--algo", "ppo", "--steps", "300", "--n-envs", "2", "--n-steps", "16", "--eval-every", "0",
                "--final-eval-episodes", "2", "--obs", "diff", "--out", out])
    assert os.path.exists(os.path.join(out, "curve.png"))
    evaluate.main([os.path.join(out, "model.pt"), "--episodes", "2"])
    gif = str(tmp_path / "a.gif")
    play.main([os.path.join(out, "model.pt"), "--gif", gif, "--max-frames", "50"])
    assert os.path.getsize(gif) > 0
    md = str(tmp_path / "t.md")
    compare.main([out, "--markdown", md, "--plot", str(tmp_path / "c.png")])
    assert "`ppo`" in open(md).read()
    assert "score" in capsys.readouterr().out


def test_config_defaults_and_overrides():
    assert make_config("pg").obs == "diff"
    assert make_config("dueling-dqn").dueling and make_config("dueling-dqn").double
    assert make_config("ppo", lr=1e-3, obs=None).lr == 1e-3
    with pytest.raises(ValueError):
        make_config("sarsa")


def test_max_minutes_stops_training_early(tmp_path):
    cfg = make_config("pg", out=str(tmp_path / "pg"), steps=10_000_000, max_minutes=0.05, eval_every=0,
                      final_eval_episodes=1)
    train.run(cfg, plot=False)
    log = json.load(open(tmp_path / "pg" / "log.json"))
    assert log["final"]["steps"] < 10_000_000
