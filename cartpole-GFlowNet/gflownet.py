import torch
import torch.nn as nn


# Flow values grow to several hundred at start states, because
# log F(s) is roughly (500 - t) * (log 2 + 1 / TEMPERATURE).
# Multiplying a small raw output by a fixed scale keeps the
# network itself operating at order 1.
FLOW_SCALE = 100.0


class GFlowNetPolicy(nn.Module):
    """
    GFlowNet with a Detailed Balance parameterisation.

    Trajectory Balance uses one scalar log_Z for a whole
    trajectory, which has to travel ~250 nats by gradient
    descent before anything works. Detailed Balance instead
    learns a state flow log F(s), so the balance condition is
    checked on every transition:

        log F(s) + log P_F(a | s) = 1 / TEMPERATURE + log F(s')

    with log F(terminal) = 0.

    Input:
        5 values: 4 CartPole observations + t / MAX_STEPS

    Outputs:
        policy_logits: logits for two actions (0 = left, 1 = right)
        log_flow:      scalar log F(s)

    The timestep is part of the state on purpose. CartPole-v1
    truncates at 500 steps, so without a clock the same
    observation could sit at two different depths and the flow
    through it would be ill defined. With the clock the state
    graph is a proper DAG, truncation is a genuine terminal
    state, and every state has exactly one parent - which is why
    the backward policy drops out (log P_B = 0).
    """

    def __init__(self, hidden_size=128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_size, 2)

        self.flow_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Return (policy_logits, log_flow) for a batch of states.
        """

        features = self.trunk(state)

        policy_logits = self.policy_head(features)

        log_flow = (
            self.flow_head(features).squeeze(-1)
            * FLOW_SCALE
        )

        return policy_logits, log_flow

    def action_distribution(self, state):
        """
        Return the forward policy P_F(a | s).
        """

        policy_logits, _ = self.forward(state)

        return torch.distributions.Categorical(
            logits=policy_logits
        )
