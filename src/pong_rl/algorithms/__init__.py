"""Training loops, one module per method. Each exposes ``train(cfg, logger) -> (model, obs_shape, n_actions)``."""
import torch

from . import a2c, dqn, pg, ppo, reinforce

# argument checks on every Categorical cost ~10% of a CPU step; the inputs here are always valid
torch.distributions.Distribution.set_default_validate_args(False)

TRAINERS = {
    "pg": pg.train,
    "reinforce": reinforce.train,
    "a2c": a2c.train,
    "ppo": ppo.train,
    "dqn": dqn.train,
    "double-dqn": dqn.train,
    "dueling-dqn": dqn.train,
}
