import torch
import torch.nn as nn


# The point mass reaches speeds of about 5, an order of magnitude above the
# position range, so velocities are rescaled before entering the network.
VELOCITY_SCALE = 5.0

# Fourier frequencies applied to position and goal. A plain MLP on raw
# coordinates smooths across the maze walls, which is exactly the mistake
# that makes the agent turn into the block in the middle of the U instead
# of going around it. Sinusoidal features let it represent that boundary
# sharply.
FOURIER_FREQUENCIES = (1.0, 2.0, 4.0)


class GFlowNet(nn.Module):
    """Goal-conditioned GFlowNet in the edge-flow parameterisation.

    The network outputs one log-flow per action, log F(s, a, g). Everything
    else is derived from it:

        P_F(a | s, g) = softmax_a log F(s, a, g)
        log F(s, g)   = logsumexp_a log F(s, a, g)

    so the policy and the state flow can never disagree, and both are
    anchored by the terminal reward through the subtrajectory balance
    condition optimised in train.py.
    """

    def __init__(self, state_dim=6, num_actions=4, hidden_dim=256):

        super().__init__()

        self.state_dim = state_dim

        self.register_buffer(
            "frequencies",
            torch.tensor(FOURIER_FREQUENCIES) * torch.pi
        )

        # raw state + displacement to goal + distance to goal
        base_dim = state_dim + 3

        # sin and cos of every frequency, for both position and goal
        fourier_dim = 2 * 2 * 2 * len(FOURIER_FREQUENCIES)

        self.network = nn.Sequential(
            nn.Linear(base_dim + fourier_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_actions)
        )

    def features(self, state):

        position = state[..., 0:2]
        velocity = state[..., 2:4]
        goal = state[..., 4:6]

        displacement = goal - position

        distance = torch.linalg.vector_norm(
            displacement,
            dim=-1,
            keepdim=True
        )

        coordinates = torch.cat([position, goal], dim=-1)

        angles = coordinates.unsqueeze(-1) * self.frequencies

        angles = angles.flatten(start_dim=-2)

        return torch.cat(
            [
                position,
                velocity / VELOCITY_SCALE,
                goal,
                displacement,
                distance,
                torch.sin(angles),
                torch.cos(angles)
            ],
            dim=-1
        )

    def forward(self, state):

        log_edge_flows = self.network(self.features(state))

        return log_edge_flows

    def action_distribution(self, state):

        log_edge_flows = self.forward(state)

        probabilities = torch.softmax(log_edge_flows, dim=-1)

        return probabilities

    def log_state_flow(self, state):

        log_edge_flows = self.forward(state)

        return torch.logsumexp(log_edge_flows, dim=-1)
