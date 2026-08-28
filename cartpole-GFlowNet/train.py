import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from gflownet import GFlowNetPolicy


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPISODES = 3000

LEARNING_RATE = 1e-3

# Reward temperature.
#
# Under Detailed Balance the per-step log reward is
# 1 / TEMPERATURE, weighed against an implicit entropy weight
# of 1. Unlike the Trajectory Balance version, TEMPERATURE no
# longer controls how LONG training takes - it only rescales
# the range of log F(s).
TEMPERATURE = 2.0

# Replay buffer.
#
# The Detailed Balance condition holds for any transition, no
# matter which policy produced it, so old transitions stay
# valid training data. This is where the sample efficiency
# comes from: every transition is reused many times instead of
# being thrown away after one gradient step.
BUFFER_CAPACITY = 100_000

BATCH_SIZE = 256

# Gradient steps taken after each episode.
UPDATES_PER_EPISODE = 32

# Transitions collected before training starts.
WARMUP_TRANSITIONS = 1000

# Polyak rate for the target flow network. Detailed Balance
# bootstraps on log F(s'), so without a target network the flow
# chases its own output.
TARGET_TAU = 0.005

SEED = 42

MODEL_PATH = "models/gflownet_cartpole.pt"

MAX_STEPS = 500

EVAL_EVERY = 250

EVAL_EPISODES = 20


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# STATE CONSTRUCTION
# ============================================================

def build_state(observation, step):
    """
    Augment the CartPole observation with the normalised
    timestep, giving the 5-value state the network expects.
    """

    return np.concatenate(
        [
            observation,
            [step / MAX_STEPS],
        ]
    ).astype(np.float32)


# ============================================================
# ENVIRONMENT AND MODEL
# ============================================================

env = gym.make("CartPole-v1")

model = GFlowNetPolicy()

target_model = GFlowNetPolicy()

target_model.load_state_dict(
    model.state_dict()
)

for parameter in target_model.parameters():
    parameter.requires_grad_(False)


optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# REPLAY BUFFER
# ============================================================

buffer = deque(maxlen=BUFFER_CAPACITY)


def sample_batch(size):
    """
    Sample a batch of transitions as stacked tensors.
    """

    batch = random.sample(buffer, size)

    states = torch.from_numpy(
        np.stack([item[0] for item in batch])
    )

    actions = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    )

    next_states = torch.from_numpy(
        np.stack([item[2] for item in batch])
    )

    terminals = torch.tensor(
        [item[3] for item in batch],
        dtype=torch.float32
    )

    return states, actions, next_states, terminals


# ============================================================
# DETAILED BALANCE LOSS
# ============================================================

def detailed_balance_loss(states, actions, next_states, terminals):
    """
    Squared residual of the Detailed Balance condition, applied
    to every transition:

        log F(s) + log P_F(a | s) = 1 / TEMPERATURE + log F(s')

    with log F(terminal) = 0.

    The backward policy is absent because the timestep is part
    of the state, so every state has exactly one parent and
    log P_B = 0.
    """

    policy_logits, log_flow = model(states)

    log_prob = torch.log_softmax(
        policy_logits,
        dim=1
    ).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():

        _, next_log_flow = target_model(next_states)

        # Terminal states carry no outgoing flow.
        next_log_flow = next_log_flow * (1.0 - terminals)

    step_log_reward = 1.0 / TEMPERATURE

    residual = (
        log_flow
        + log_prob
        - step_log_reward
        - next_log_flow
    )

    return (residual ** 2).mean()


def soft_update_target():
    """
    Polyak averaging of the target network.
    """

    with torch.no_grad():

        for target_parameter, parameter in zip(
            target_model.parameters(),
            model.parameters()
        ):

            target_parameter.mul_(1.0 - TARGET_TAU)
            target_parameter.add_(TARGET_TAU * parameter)


# ============================================================
# GREEDY EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_greedy(episodes):
    """
    Run argmax episodes to measure true performance during
    training.
    """

    eval_env = gym.make("CartPole-v1")

    model.eval()

    total = 0.0

    for index in range(episodes):

        observation, info = eval_env.reset(
            seed=90000 + index
        )

        step = 0

        done = False

        while not done:

            state = torch.from_numpy(
                build_state(observation, step)
            ).unsqueeze(0)

            logits, _ = model(state)

            action = int(
                torch.argmax(logits, dim=1).item()
            )

            observation, reward, terminated, truncated, info = eval_env.step(
                action
            )

            total += reward

            step += 1

            done = terminated or truncated

    model.train()

    eval_env.close()

    return total / episodes


# ============================================================
# TRAINING
# ============================================================

reward_history = []

success_history = []

last_loss = 0.0

best_eval = 0.0

episodes_to_solve = None


print("=" * 78)
print("GFlowNet CartPole Training - Detailed Balance")
print("=" * 78)

print(f"Episodes: {NUM_EPISODES}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Temperature: {TEMPERATURE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Updates per episode: {UPDATES_PER_EPISODE}")
print(f"Buffer capacity: {BUFFER_CAPACITY}")
print(f"Maximum steps: {MAX_STEPS}")
print()


for episode in range(1, NUM_EPISODES + 1):

    observation, info = env.reset(
        seed=SEED + episode
    )

    step = 0

    done = False

    episode_reward = 0.0

    # ========================================================
    # GENERATE TRAJECTORY
    #
    # Actions are sampled from the forward policy P_F. No
    # entropy bonus is needed - a GFlowNet policy is maximum
    # entropy by construction, so an explicit entropy term
    # would double count.
    #
    # The rollout runs under no_grad. Gradients come only from
    # replayed batches, so there is no reason to build a graph
    # here or to re-forward the trajectory afterwards.
    # ========================================================

    while not done:

        state = build_state(observation, step)

        with torch.no_grad():

            distribution = model.action_distribution(
                torch.from_numpy(state).unsqueeze(0)
            )

            action = int(
                distribution.sample().item()
            )

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        step += 1

        episode_reward += reward

        done = terminated or truncated

        next_state = build_state(observation, step)

        # Both a fallen pole and truncation at 500 steps are
        # genuine terminals of this finite horizon DAG.
        buffer.append(
            (state, action, next_state, float(done))
        )

    reward_history.append(episode_reward)

    success_history.append(
        int(episode_reward >= MAX_STEPS)
    )

    # ========================================================
    # UPDATE
    # ========================================================

    if len(buffer) >= max(WARMUP_TRANSITIONS, BATCH_SIZE):

        for _ in range(UPDATES_PER_EPISODE):

            states, actions, next_states, terminals = sample_batch(
                BATCH_SIZE
            )

            loss = detailed_balance_loss(
                states,
                actions,
                next_states,
                terminals
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0
            )

            optimizer.step()

            soft_update_target()

        last_loss = loss.item()

    # ========================================================
    # LOGGING
    # ========================================================

    if episode % 100 == 0:

        recent_rewards = reward_history[-100:]

        recent_successes = success_history[-100:]

        print(
            f"Episode {episode:6d} | "
            f"Avg reward: {sum(recent_rewards) / len(recent_rewards):7.2f} | "
            f"Max: {max(recent_rewards):6.0f} | "
            f"Success: {sum(recent_successes):3d}% | "
            f"DB loss: {last_loss:9.4f} | "
            f"Buffer: {len(buffer):6d}"
        )

    if episode % EVAL_EVERY == 0:

        greedy_reward = evaluate_greedy(EVAL_EPISODES)

        print(
            f"    -> greedy eval over {EVAL_EPISODES} episodes: "
            f"{greedy_reward:.2f}"
        )

        if greedy_reward > best_eval:

            best_eval = greedy_reward

            torch.save(
                {"model_state_dict": model.state_dict()},
                MODEL_PATH
            )

        if greedy_reward >= MAX_STEPS and episodes_to_solve is None:

            episodes_to_solve = episode

            print(
                f"    -> SOLVED (greedy 500/500) at episode {episode}"
            )


env.close()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 78)
print("TRAINING FINISHED")
print("=" * 78)

print(f"Best greedy eval: {best_eval:.2f}")

if episodes_to_solve is not None:
    print(f"First perfect greedy eval at episode: {episodes_to_solve}")

print(f"Model saved to: {MODEL_PATH}")
