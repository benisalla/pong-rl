"""Networks. Every method picks from these four, so differences in results come from the
learning rule, not from the architecture.

* ``KarpathyMLP``  : flatten -> Linear(D, 200) -> ReLU -> Linear(200, 1) -> sigmoid = P(UP),
  with D = 6400 on ALE and 400 on the small game.
  The network from "Pong from Pixels" (2016), only two actions (UP / DOWN).
* ``ConvEncoder``  : pixels -> feature vector. Two sizes:
    - ``nature`` (Mnih et al. 2015): conv 8x8/4 (32) -> 4x4/2 (64) -> 3x3/1 (64) -> Linear 512.
      For 80x80 inputs: 80 -> 19 -> 8 -> 6, so 64*6*6 = 2304 features before the Linear.
    - ``small``: conv 4x4/2 (16) -> 3x3/2 (32) -> Linear 256, for the 20x20 frames of the small
      game: 20 -> 9 -> 4, so 32*4*4 = 512 features. The nature net would squeeze a 20x20 frame
      below 1x1, and it would be ~20x slower on a CPU for no gain on such a simple picture.
  ``encoder="auto"`` picks ``small`` for frames up to 48 pixels and ``nature`` above.
* ``ActorCritic``  : encoder -> policy logits (n_actions) and a state value V(s).
  Used by REINFORCE (V is the baseline), A2C and PPO.
* ``QNetwork``     : encoder -> Q(s, .) for DQN / double DQN; ``dueling=True`` splits the head into
  V(s) + A(s, a) - mean_a A(s, a) (Wang et al. 2016).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KarpathyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 200):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Karpathy's "Xavier" init: N(0, 1/fan_in), no biases in the original; zero biases here
        for layer in (self.fc1, self.fc2):
            nn.init.normal_(layer.weight, std=layer.in_features ** -0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> logit of P(UP), shape (B,)."""
        h = torch.relu(self.fc1(x.float().flatten(1)))
        return self.fc2(h).squeeze(-1)


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], kind: str = "auto"):
        super().__init__()
        c, h, w = obs_shape
        if kind == "auto":
            kind = "small" if max(h, w) <= 48 else "nature"
        if kind == "nature":
            convs = [nn.Conv2d(c, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                     nn.Conv2d(64, 64, 3, 1), nn.ReLU()]
            out = 512
        elif kind == "small":
            convs = [nn.Conv2d(c, 16, 4, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU()]
            out = 256
        else:
            raise ValueError(f"unknown encoder {kind!r}")
        self.kind = kind
        self.convs = nn.Sequential(*convs, nn.Flatten())
        with torch.no_grad():
            n_flat = self.convs(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, out), nn.ReLU())
        self.out_dim = out
        # Orthogonal init with ReLU gain sqrt(2): the inputs are sparse binary frames (a few dozen
        # pixels on), and PyTorch's default init would shrink the features layer after layer until
        # the policy head gets almost no gradient.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                _ortho(m, 2 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.convs(x.float()))


def _ortho(module: nn.Module, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain)
    nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions: int, encoder: str = "auto"):
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, encoder)
        self.pi = nn.Linear(self.encoder.out_dim, n_actions)
        self.v = nn.Linear(self.encoder.out_dim, 1)
        _ortho(self.pi, 0.01)  # near-uniform initial policy: explore first
        _ortho(self.v, 1.0)

    def forward(self, x: torch.Tensor):
        """x: (B, C, H, W) -> (logits (B, A), value (B,))."""
        z = self.encoder(x)
        return self.pi(z), self.v(z).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions: int, dueling: bool = False, encoder: str = "auto"):
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, encoder)
        self.dueling = dueling
        d = self.encoder.out_dim
        if dueling:
            self.value = nn.Linear(d, 1)
            self.advantage = nn.Linear(d, n_actions)
        else:
            self.q = nn.Linear(d, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> Q values (B, A)."""
        z = self.encoder(x)
        if not self.dueling:
            return self.q(z)
        a = self.advantage(z)
        # subtracting the mean makes V and A identifiable (otherwise V+c, A-c gives the same Q)
        return self.value(z) + a - a.mean(dim=1, keepdim=True)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
