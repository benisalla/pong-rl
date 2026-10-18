import numpy as np
import pytest

from pong_rl.envs import MiniPong, VecEnv, make_env, preprocess
from pong_rl.envs.mini_pong import DOWN, UP


def play(env, policy, seed=0):
    env.reset(seed=seed)
    total, done, n = 0.0, False, 0
    while not done:
        _, r, term, trunc, _ = env.step(policy(env))
        total += r
        done = term or trunc
        n += 1
    return total, n, term


def tracker(env):
    c, b = env.agent_y + env.ph / 2, env.by + 1
    return UP if b < c - 1 else (DOWN if b > c + 1 else 0)


def test_frame_shape_and_values():
    env = MiniPong(seed=0)
    f, info = env.reset(seed=0)
    assert f.shape == (40, 40) and f.dtype == np.uint8
    assert set(np.unique(f)) <= {0, 255}
    assert (f > 0).sum() == 8 + 8 + 4  # two paddles of 8 pixels and a 2x2 ball


def test_game_ends_at_points_to_win_and_scores_are_bounded():
    env = MiniPong(seed=0, points_to_win=3)
    total, _, term = play(env, lambda e: 0)
    assert term and -3 <= total <= 3 and max(env.score_agent, env.score_opp) == 3


def test_tracking_policy_beats_random():
    env = MiniPong(seed=1)
    rng = np.random.default_rng(0)
    rand = np.mean([play(env, lambda e: int(rng.integers(3)), seed=s)[0] for s in range(10)])
    track = np.mean([play(env, tracker, seed=s)[0] for s in range(10)])
    assert rand < -1.5 and track > 2.0  # the game is winnable and random play loses


def test_same_seed_same_game():
    a, b = MiniPong(), MiniPong()
    fa, _ = a.reset(seed=3)
    fb, _ = b.reset(seed=3)
    for t in range(50):
        oa = a.step(t % 3)
        ob = b.step(t % 3)
        assert np.array_equal(oa[0], ob[0]) and oa[1] == ob[1]


def test_paddle_moves_and_frameskip():
    env = MiniPong(seed=0, frameskip=2)
    env.reset(seed=0)
    y = env.agent_y
    env.step(UP)
    assert env.agent_y == y - 2 * env.paddle_speed and env.t == 2


def test_sticky_actions_repeat_previous_action():
    env = MiniPong(seed=0, sticky=1.0, frameskip=1)
    env.reset(seed=0)
    y = env.agent_y
    env.step(UP)  # sticky=1 always repeats the previous action (NOOP after reset)
    assert env.agent_y == y


def test_preprocess_ale_like_frame():
    frame = np.full((210, 160, 3), 144, dtype=np.uint8)  # background colour
    frame[100:110, 140:142] = (92, 186, 92)  # a paddle
    out = preprocess(frame, "ale")
    assert out.shape == (80, 80) and out.dtype == np.uint8
    assert set(np.unique(out)) == {0, 1} and out.sum() == 5 * 1  # 10x2 pixels -> 5x1 after ::2


@pytest.mark.parametrize("obs,shape,dtype", [("diff", (1, 20, 20), np.int8), ("stack", (4, 20, 20), np.uint8)])
def test_pixel_obs_modes(obs, shape, dtype):
    env = make_env("mini", obs=obs, seed=0)
    o, _ = env.reset(seed=0)
    assert o.shape == shape and o.dtype == dtype and env.observation_shape == shape
    if obs == "diff":
        assert not o.any()  # first frame of an episode: zeros, like Karpathy
    for _ in range(3):
        o, *_ = env.step(UP)
    if obs == "diff":
        assert set(np.unique(o)) <= {-1, 0, 1} and (o == 1).any() and (o == -1).any()
    else:
        assert not np.array_equal(o[0], o[-1])  # the stack holds different frames


def test_episode_info_and_vecenv_autoreset():
    v = VecEnv(3, seed=0, env="mini", obs="stack")
    obs = v.reset()
    assert obs.shape == (3, 4, 20, 20)
    finished = 0
    for _ in range(2000):
        obs, r, te, tr, infos = v.step(np.zeros(3, dtype=int))
        finished += sum("episode" in i for i in infos)
        if finished:
            break
    assert finished > 0
    ep = next(i["episode"] for i in infos if "episode" in i)
    assert -5 <= ep["r"] <= 5 and ep["l"] > 0


def test_ale_pong_runs_headless():
    pytest.importorskip("ale_py")
    env = make_env("ale", obs="diff", seed=0)
    o, _ = env.reset(seed=0)
    assert o.shape == (1, 80, 80)
    for _ in range(30):
        o, r, term, trunc, _ = env.step(UP)
    assert o.dtype == np.int8 and env.n_actions == 3
    env.close()
