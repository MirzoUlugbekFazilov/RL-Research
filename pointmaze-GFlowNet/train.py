import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from environment import PointMazeWrapper
from gflownet import GFlowNet


# -----------------------------
# Configuration
# -----------------------------

NUM_EPISODES = 3000
MAX_STEPS = 100

LEARNING_RATE = 1e-3

BATCH_SIZE = 256
UPDATES_PER_EPISODE = 32
LEARNING_STARTS = 20

# Longest subtrajectory used by the balance condition. One-step updates
# move the terminal anchor backwards a single step at a time, which is far
# too slow for the 40-70 step trips around the U; sampling subtrajectories
# up to this length propagates it in one update.
MAX_SUBTRAJECTORY = 8

# Terminal states get reward R = 1 (log R = 0) and every transition costs
# STEP_COST in log space. It has to exceed log(num_actions) = 1.39 or the
# sheer number of wandering trajectories outweighs the short ones and the
# flow stops pointing at the goal. The margin above 1.39 is what separates
# the actions, because the logsumexp backup adds about log(4) of
# path-counting entropy per step. The residual is divided by STEP_COST in
# the loss so this stays a policy sharpness knob and does not also rescale
# the gradients.
STEP_COST = 4.0

# Polyak averaging for the bootstrap target of the balance condition.
TARGET_UPDATE_RATE = 0.005

# How goals are drawn when a stored episode is replayed. "Future" goals are
# positions from later in that same episode and teach short hops; "pool"
# goals come from anywhere the agent has ever been, and they are what make
# the flow reach across the whole U to a distant goal.
REAL_GOAL_PROBABILITY = 0.2
FUTURE_GOAL_PROBABILITY = 0.3

# Fraction of episodes rolled out with pure argmax, matching how the policy
# is evaluated. Without these the buffer never contains the states a greedy
# run actually visits, so its mistakes compound off-distribution.
GREEDY_ROLLOUT_FRACTION = 0.5

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 600

# Greedy checkpoint selection. These episodes are evaluation only, they are
# not part of the NUM_EPISODES training budget.
EVAL_EVERY = 100
EVAL_EPISODES = 100

# Evaluation gets a longer budget than collection. The far corners of the U are
# ~6 units of corridor apart and the point mass tops out at ~0.052 units per
# step, so those episodes need ~115 steps from any policy. Scoring them at
# MAX_STEPS would censor the metric and pick checkpoints on noise in the
# handful of episodes that are unreachable anyway.
EVAL_MAX_STEPS = 150

SEED = 0

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gflownet_pointmaze.pt")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Environment
# -----------------------------

env = PointMazeWrapper(max_steps=MAX_STEPS)

eval_env = PointMazeWrapper(max_steps=EVAL_MAX_STEPS)

GOAL_RADIUS = env.goal_radius


# -----------------------------
# GFlowNet
# -----------------------------

model = GFlowNet(
    state_dim=env.state_dim,
    num_actions=env.num_actions
).to(DEVICE)


target_model = copy.deepcopy(model).to(DEVICE)

for parameter in target_model.parameters():
    parameter.requires_grad_(False)


optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Decayed towards the end so the flow settles instead of chasing a moving
# bootstrap target, which is what makes the greedy policy flip between
# actions in ambiguous states.
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPISODES * UPDATES_PER_EPISODE,
    eta_min=LEARNING_RATE / 20
)


# -----------------------------
# Episode store
# -----------------------------
#
# Whole episodes are kept, padded to MAX_STEPS, rather than loose
# transitions. Subtrajectory balance needs consecutive steps, and keeping
# episodes also means a fresh set of hindsight goals can be drawn every
# time an episode is replayed instead of freezing a few relabelled copies
# at collection time.

class EpisodeStore:

    def __init__(self, capacity, max_steps, robot_dim=4, goal_dim=2):

        self.capacity = capacity
        self.max_steps = max_steps

        self.index = 0
        self.size = 0

        self.states = np.zeros(
            (capacity, max_steps, robot_dim), dtype=np.float32
        )
        self.next_states = np.zeros(
            (capacity, max_steps, robot_dim), dtype=np.float32
        )
        self.actions = np.zeros((capacity, max_steps), dtype=np.int64)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.lengths = np.zeros(capacity, dtype=np.int64)

    def __len__(self):

        return self.size

    def add(self, states, actions, next_states, goal):

        length = len(actions)

        row = self.index

        self.states[row, :length] = states
        self.next_states[row, :length] = next_states
        self.actions[row, :length] = actions
        self.goals[row] = goal
        self.lengths[row] = length

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_positions(self, count):
        """Random visited positions, used as the pool of distant goals."""

        rows = np.random.randint(0, self.size, size=count)
        steps = (np.random.random(count) * self.lengths[rows]).astype(
            np.int64
        )

        return self.next_states[rows, steps, :2]

    def sample_subtrajectories(self, batch_size, max_length):

        rows = np.random.randint(0, self.size, size=batch_size)

        lengths = self.lengths[rows]

        starts = (np.random.random(batch_size) * lengths).astype(np.int64)

        # --- goals: real, hindsight from this episode, or from the pool ---

        goals = self.sample_positions(batch_size)

        choice = np.random.random(batch_size)

        real = choice < REAL_GOAL_PROBABILITY

        goals[real] = self.goals[rows[real]]

        future = (
            (choice >= REAL_GOAL_PROBABILITY)
            & (choice < REAL_GOAL_PROBABILITY + FUTURE_GOAL_PROBABILITY)
        )

        span = lengths[future] - starts[future]

        offsets = (np.random.random(span.shape[0]) * span).astype(np.int64)

        goals[future] = self.next_states[
            rows[future],
            starts[future] + offsets,
            :2
        ]

        # --- where the subtrajectory has to stop ---

        distances = np.linalg.norm(
            self.next_states[rows, :, :2] - goals[:, None, :],
            axis=2
        )

        steps = np.arange(self.max_steps)[None, :]

        inside = (
            (distances <= GOAL_RADIUS)
            & (steps >= starts[:, None])
            & (steps < lengths[:, None])
        )

        hits = inside.any(axis=1)

        first_hit = np.where(hits, inside.argmax(axis=1), lengths - 1)

        wanted = np.random.randint(1, max_length + 1, size=batch_size)

        available = np.where(
            hits,
            first_hit - starts + 1,
            lengths - starts
        )

        counts = np.minimum(wanted, available)

        # The subtrajectory ends on the goal only when it runs all the way
        # to the step that first entered the radius.
        terminals = hits & (starts + counts - 1 == first_hit)

        # A source state already inside the radius is terminal and has no
        # outgoing flow, so those samples must not contribute.
        valid = (
            np.linalg.norm(self.states[rows, starts, :2] - goals, axis=1)
            > GOAL_RADIUS
        )

        offsets = np.arange(max_length)[None, :]

        indices = np.minimum(starts[:, None] + offsets, self.max_steps - 1)

        mask = offsets < counts[:, None]

        batch_rows = rows[:, None]

        return (
            torch.as_tensor(self.states[batch_rows, indices], device=DEVICE),
            torch.as_tensor(self.actions[batch_rows, indices], device=DEVICE),
            torch.as_tensor(
                self.next_states[rows, starts + counts - 1], device=DEVICE
            ),
            torch.as_tensor(goals, device=DEVICE),
            torch.as_tensor(mask, device=DEVICE),
            torch.as_tensor(counts, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(terminals, device=DEVICE),
            torch.as_tensor(valid, device=DEVICE)
        )


store = EpisodeStore(NUM_EPISODES, MAX_STEPS)


# -----------------------------
# Subtrajectory balance
# -----------------------------
#
# For a run of n consecutive steps starting at s_i:
#
#   log F(s_i, a_i) + sum_{j>i} log P_F(a_j | s_j) + n * STEP_COST
#       = log F(s_end)
#
# with log F(s_end) = log R = 0 when the run ends inside the goal radius.
# The backward policy is deterministic (P_B = 1), so each state along a
# sampled trajectory has a single parent. At n = 1 this is exactly the
# detailed balance condition; longer n carries the terminal anchor several
# steps back in a single update.

def subtrajectory_balance_loss(
    states, actions, end_states, goals, mask, counts, terminals, valid
):

    steps = states.shape[1]

    goal_column = goals.unsqueeze(1).expand(-1, steps, -1)

    inputs = torch.cat([states, goal_column], dim=2)

    log_edge_flows = model(inputs)

    chosen = log_edge_flows.gather(2, actions.unsqueeze(2)).squeeze(2)

    log_state_flows = torch.logsumexp(log_edge_flows, dim=2)

    # First step contributes its edge flow, later steps their policy
    # log-probability.
    log_forward = chosen - log_state_flows

    contributions = torch.where(
        F.pad(mask[:, :-1], (1, 0), value=False),
        log_forward,
        torch.zeros_like(log_forward)
    )

    total = chosen[:, 0] + (contributions * mask).sum(dim=1)

    with torch.no_grad():

        end_inputs = torch.cat([end_states, goals], dim=1)

        end_flow = target_model.log_state_flow(end_inputs)

        end_flow = torch.where(
            terminals,
            torch.zeros_like(end_flow),
            end_flow
        )

    residual = (total + counts * STEP_COST - end_flow) / STEP_COST

    residual = residual[valid]

    if residual.numel() == 0:
        return None

    return F.huber_loss(
        residual,
        torch.zeros_like(residual),
        delta=1.0
    )


def update_target():

    with torch.no_grad():

        for online, target in zip(
            model.parameters(),
            target_model.parameters()
        ):

            target.mul_(1.0 - TARGET_UPDATE_RATE)
            target.add_(TARGET_UPDATE_RATE * online)


# -----------------------------
# Greedy evaluation
# -----------------------------

def greedy_success_rate(episodes):

    successes = 0

    for index in range(episodes):

        state = eval_env.reset(seed=100000 + index)

        for step in range(EVAL_MAX_STEPS):

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                action = torch.argmax(model(state_tensor), dim=-1).item()

            state, reward, terminated, truncated = eval_env.step(action)

            if terminated:
                successes += 1
                break

            if truncated:
                break

    return successes / episodes


# -----------------------------
# Training
# -----------------------------

recent_successes = deque(maxlen=100)

loss_value = float("nan")

best_success_rate = -1.0

os.makedirs(MODEL_DIR, exist_ok=True)


for episode in range(NUM_EPISODES):

    state = env.reset()

    goal = state[4:6].copy()

    epsilon = max(
        EPSILON_END,
        EPSILON_START
        - (EPSILON_START - EPSILON_END)
        * episode
        / EPSILON_DECAY_EPISODES
    )

    greedy_rollout = random.random() < GREEDY_ROLLOUT_FRACTION

    robot_states = []
    actions = []
    next_robot_states = []

    reached = False

    for step in range(MAX_STEPS):

        if not greedy_rollout and random.random() < epsilon:

            action = random.randrange(env.num_actions)

        else:

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                log_edge_flows = model(state_tensor)

            if greedy_rollout:

                action = torch.argmax(log_edge_flows, dim=-1).item()

            else:

                action = torch.multinomial(
                    torch.softmax(log_edge_flows, dim=-1),
                    num_samples=1
                ).item()

        next_state, reward, terminated, truncated = env.step(action)

        robot_states.append(state[:4].copy())
        actions.append(action)
        next_robot_states.append(next_state[:4].copy())

        state = next_state

        if terminated:
            reached = True

        if terminated or truncated:
            break

    recent_successes.append(float(reached))

    store.add(
        np.asarray(robot_states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(next_robot_states, dtype=np.float32),
        goal
    )

    if len(store) >= LEARNING_STARTS:

        for _ in range(UPDATES_PER_EPISODE):

            loss = subtrajectory_balance_loss(
                *store.sample_subtrajectories(BATCH_SIZE, MAX_SUBTRAJECTORY)
            )

            if loss is None:
                continue

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            scheduler.step()

            update_target()

            loss_value = loss.item()

    if (episode + 1) % EVAL_EVERY == 0:

        success_rate = greedy_success_rate(EVAL_EPISODES)

        if success_rate >= best_success_rate:

            best_success_rate = success_rate

            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Episode: {episode + 1}/{NUM_EPISODES} "
            f"| Balance loss: {loss_value:.4f} "
            f"| Sampled success: {np.mean(recent_successes) * 100:.0f}% "
            f"| Greedy success: {success_rate * 100:.0f}% "
            f"| Best: {best_success_rate * 100:.0f}% "
            f"| Epsilon: {epsilon:.2f}"
        )


env.close()

eval_env.close()

print(
    f"Training finished. Best greedy success rate: "
    f"{best_success_rate * 100:.0f}%"
)

print(f"Model saved to {MODEL_PATH}")
