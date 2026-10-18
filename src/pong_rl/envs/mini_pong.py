"""A small Pong written in numpy: 40x40 frames, one paddle per side, one ball.

Why it exists: real Atari Pong needs millions of frames before a pixel agent wins, which is
days on a CPU. This game keeps everything that makes Pong a *pixel* RL problem (you only see
frames, the ball's velocity is not in a single frame, rewards are +1/-1 per point, the game
ends when one side reaches ``points_to_win``) but it is small enough that every method in
this repo learns to beat the built-in opponent in minutes.

Layout (x grows to the right, y grows down)::

    x=1                                   x=38
    |  opponent paddle (built-in AI)       |  agent paddle (you), like the green paddle in Atari Pong
    |              ball (2x2)              |

Actions follow the reduced Atari Pong set used everywhere in this repo:
``0 = NOOP, 1 = UP, 2 = DOWN`` (ALE actions 0, 2 and 3; see ``envs.atari.ACTION_MAP``).

One call to ``step`` repeats the action for ``frameskip`` game ticks (2 by default, ALE uses 4)
and returns the last frame: the agent decides half as often, which halves the compute and makes
each decision matter more. One game tick:
1. the agent paddle moves ``paddle_speed`` pixels (or stays, or with probability ``sticky``
   repeats the previous action, like ALE's sticky actions);
2. the opponent moves at most ``opponent_speed`` pixels towards the ball and freezes with
   probability ``opponent_noise`` (so it is beatable: steep balls get past it);
3. the ball moves ``(vx, vy)``, bounces off the top and bottom walls, and either bounces off a
   paddle (the vertical speed depends on where it hits the paddle: the edges give steep balls,
   exactly the trick a good Pong player learns) or crosses the goal line: +1 if it passed the
   opponent, -1 if it passed the agent, then a new serve from the centre.
"""
from __future__ import annotations

import numpy as np

NOOP, UP, DOWN = 0, 1, 2
ACTION_MEANINGS = ["NOOP", "UP", "DOWN"]


class MiniPong:
    """Gymnasium-style API: ``reset(seed) -> (frame, info)``, ``step(a) -> (frame, r, terminated, truncated, info)``.

    Frames are ``uint8`` arrays of shape (40, 40): 0 background, 255 for paddles and ball.
    """

    n_actions = 3
    action_meanings = ACTION_MEANINGS

    def __init__(
        self,
        size: int = 40,
        paddle_height: int = 8,
        paddle_speed: int = 2,
        ball_speed_x: int = 2,
        opponent_speed: int = 2,
        opponent_noise: float = 0.3,
        points_to_win: int = 5,
        max_steps: int = 4000,  # game ticks
        sticky: float = 0.0,
        frameskip: int = 2,
        seed: int | None = None,
    ):
        self.size = size
        self.ph = paddle_height
        self.paddle_speed = paddle_speed
        self.ball_speed_x = ball_speed_x
        self.opponent_speed = opponent_speed
        self.opponent_noise = opponent_noise
        self.points_to_win = points_to_win
        self.max_steps = max_steps
        self.sticky = sticky
        self.frameskip = frameskip
        self.rng = np.random.default_rng(seed)
        self.agent_x = size - 2  # paddle column of the agent (right side)
        self.opp_x = 1  # paddle column of the opponent (left side)
        self.observation_shape = (size, size)

    # ------------------------------------------------------------------ game state
    def _serve(self, towards: int) -> None:
        """Put the ball in the centre, moving towards ``towards`` (+1 agent, -1 opponent)."""
        self.bx = self.size // 2 - 1
        self.by = int(self.rng.integers(self.size // 4, 3 * self.size // 4))
        self.vx = towards * self.ball_speed_x
        self.vy = int(self.rng.choice([-1, 1]))

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mid = (self.size - self.ph) // 2
        self.agent_y = mid  # top of the agent paddle
        self.opp_y = mid
        self.score_agent = 0
        self.score_opp = 0
        self.t = 0
        self.last_action = NOOP
        self._serve(towards=int(self.rng.choice([-1, 1])))
        return self.render(), {"score": (0, 0)}

    @staticmethod
    def _bounce_vy(ball_center: float, paddle_top: int, ph: int) -> int:
        """Vertical speed after a paddle hit: centre gives +-1, the outer quarters give +-2."""
        offset = ball_center - (paddle_top + ph / 2)  # in [-ph/2-1, ph/2+1]
        sign = 1 if offset >= 0 else -1
        return sign * (2 if abs(offset) >= ph / 4 else 1)

    def step(self, action: int):
        """Repeat ``action`` for ``frameskip`` game ticks (like ALE), sum the rewards, return the last frame."""
        total, terminated, truncated = 0.0, False, False
        for _ in range(self.frameskip):
            r, terminated, truncated = self._tick(int(action))
            total += r
            if terminated or truncated:
                break
        return self.render(), total, terminated, truncated, {"score": (self.score_agent, self.score_opp)}

    def _tick(self, action: int):
        if self.sticky > 0 and self.rng.random() < self.sticky:
            action = self.last_action
        self.last_action = action
        self.t += 1
        lim = self.size - self.ph

        # 1. agent paddle
        if action == UP:
            self.agent_y = max(0, self.agent_y - self.paddle_speed)
        elif action == DOWN:
            self.agent_y = min(lim, self.agent_y + self.paddle_speed)

        # 2. opponent paddle: follow the ball centre, limited speed, sometimes frozen
        if self.rng.random() >= self.opponent_noise:
            target = self.by + 1 - self.ph / 2
            delta = max(-self.opponent_speed, min(self.opponent_speed, round(target - self.opp_y)))
            self.opp_y = max(0, min(lim, self.opp_y + delta))

        # 3. ball
        reward = 0.0
        self.bx += self.vx
        self.by += self.vy
        if self.by < 0:  # top wall
            self.by, self.vy = -self.by, -self.vy
        elif self.by > self.size - 2:  # bottom wall (ball is 2 pixels tall)
            self.by, self.vy = 2 * (self.size - 2) - self.by, -self.vy

        center = self.by + 1
        if self.vx > 0 and self.bx + 1 >= self.agent_x:  # reached the agent's column
            if self.agent_y - 1 <= center <= self.agent_y + self.ph:
                self.bx = self.agent_x - 2
                self.vx = -self.vx
                self.vy = self._bounce_vy(center, self.agent_y, self.ph)
            else:
                reward = -1.0
                self.score_opp += 1
                self._serve(towards=-1)
        elif self.vx < 0 and self.bx <= self.opp_x:  # reached the opponent's column
            if self.opp_y - 1 <= center <= self.opp_y + self.ph:
                self.bx = self.opp_x + 1
                self.vx = -self.vx
                self.vy = self._bounce_vy(center, self.opp_y, self.ph)
            else:
                reward = 1.0
                self.score_agent += 1
                self._serve(towards=1)

        terminated = max(self.score_agent, self.score_opp) >= self.points_to_win
        truncated = (not terminated) and self.t >= self.max_steps
        return reward, terminated, truncated

    # ------------------------------------------------------------------ pixels
    def render(self) -> np.ndarray:
        """The observation: a (size, size) uint8 frame. This is all the agent ever sees."""
        f = np.zeros((self.size, self.size), dtype=np.uint8)
        f[self.agent_y:self.agent_y + self.ph, self.agent_x] = 255
        f[self.opp_y:self.opp_y + self.ph, self.opp_x] = 255
        bx = max(0, min(self.size - 2, self.bx))
        f[self.by:self.by + 2, bx:bx + 2] = 255
        return f

    def render_rgb(self, scale: int = 8) -> np.ndarray:
        """A coloured, upscaled frame for GIFs (agent green, opponent orange, ball white)."""
        h = w = self.size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (20, 24, 32)
        img[self.agent_y:self.agent_y + self.ph, self.agent_x] = (92, 186, 92)
        img[self.opp_y:self.opp_y + self.ph, self.opp_x] = (213, 130, 74)
        bx = int(np.clip(self.bx, 0, self.size - 2))
        img[self.by:self.by + 2, bx:bx + 2] = (236, 236, 236)
        return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

    def close(self) -> None:
        pass
