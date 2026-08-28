import gymnasium as gym
import numpy as np
import torch

from gflownet import GFlowNetPolicy


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/gflownet_cartpole.pt"

NUM_EVALUATION_EPISODES = 300

SUCCESS_THRESHOLD = 475

MAX_STEPS = 500


def build_state(observation, step):
    """
    Augment the CartPole observation with the normalised
    timestep, matching the state format used in training.
    """

    return np.concatenate(
        [
            observation,
            [step / MAX_STEPS],
        ]
    ).astype(np.float32)


# ============================================================
# Load environment
# ============================================================

env = gym.make("CartPole-v1")


# ============================================================
# Load model
# ============================================================

model = GFlowNetPolicy()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Evaluation
# ============================================================

rewards = []

successful_episodes = 0

print("Starting evaluation...")
print(
    f"Episodes: {NUM_EVALUATION_EPISODES}"
)

for episode in range(
    NUM_EVALUATION_EPISODES
):

    observation, info = env.reset(
        seed=10000 + episode
    )

    total_reward = 0

    step = 0

    done = False

    while not done:

        state = torch.from_numpy(
            build_state(observation, step)
        ).unsqueeze(0)

        # ----------------------------------------------------
        # IMPORTANT:
        # During evaluation we choose the most probable action.
        # We do NOT sample randomly.
        # ----------------------------------------------------

        with torch.no_grad():

            logits, _ = model(state)

            action = torch.argmax(
                logits,
                dim=1
            ).item()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        step += 1

        done = terminated or truncated

    rewards.append(
        total_reward
    )

    if total_reward >= SUCCESS_THRESHOLD:

        successful_episodes += 1


# ============================================================
# Statistics
# ============================================================

average_reward = sum(rewards) / len(rewards)

success_rate = (
    successful_episodes
    / NUM_EVALUATION_EPISODES
    * 100
)

best_reward = max(rewards)

worst_reward = min(rewards)


# ============================================================
# Results
# ============================================================

print()
print("=" * 50)
print("GFlowNet CARTPOLE EVALUATION")
print("=" * 50)

print(
    f"Episodes evaluated: {NUM_EVALUATION_EPISODES}"
)

print(
    f"Successful episodes: {successful_episodes}"
)

print(
    f"Failed episodes: "
    f"{NUM_EVALUATION_EPISODES - successful_episodes}"
)

print(
    f"Success rate: {success_rate:.2f}%"
)

print(
    f"Average reward: {average_reward:.2f}"
)

print(
    f"Best reward: {best_reward:.2f}"
)

print(
    f"Worst reward: {worst_reward:.2f}"
)

print("=" * 50)


env.close()