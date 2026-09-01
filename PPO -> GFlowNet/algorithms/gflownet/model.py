"""Networks for the conditional Trajectory-Balance GFlowNet.

See ``formulation.md`` in this package for the derivation; this module only
implements it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class GFlowNetPolicy(nn.Module):
    r"""Forward policy :math:`P_F(a \mid z_t)` plus conditional :math:`\log Z_\theta(c)`.

    Parameters
    ----------
    obs_dim   : width of the environment observation (6 for PointMaze:
                ``[x, y, vx, vy, gx, gy]``).
    n_actions : size of the discrete action set (9, bang-bang).
    hidden    : hidden width of both heads.

    The policy input is the observation with the normalised step index
    ``t / T`` appended (``obs_dim + 1``), which is what makes the GFlowNet DAG
    a tree and forces ``P_B = 1``.  The ``log Z`` head sees only the *initial*
    observation, because the partition function is a property of the
    (start, goal) instance, not of the current state.
    """

    def __init__(self, obs_dim: int = 6, n_actions: int = 9, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.pf = _mlp(obs_dim + 1, n_actions, hidden)
        self.logZ = _mlp(obs_dim, 1, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Logits of :math:`P_F(\\cdot \\mid z)`; ``z`` is ``(..., obs_dim + 1)``."""
        return self.pf(z)

    def log_pf(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``log P_F(a | z)`` for a batch of (state, action) pairs."""
        logits = self.pf(z)
        return torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def log_partition(self, cond: torch.Tensor) -> torch.Tensor:
        """``log Z_theta(c)`` for a batch of initial observations."""
        return self.logZ(cond).squeeze(-1)

    def policy_parameters(self):
        return self.pf.parameters()

    def partition_parameters(self):
        return self.logZ.parameters()
