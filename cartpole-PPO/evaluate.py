from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cartpole_model.pt"

env = gym.make("CartPole-v1")

# Load the PyTorch checkpoint written by train.py
model = ActorCriticPolicy.load(MODEL_PATH)
model.set_training_mode(False)

episodes = 100
rewards = []

for episode in range(episodes):
    obs, info = env.reset()
    total_reward = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)

env.close()

average_reward = sum(rewards) / len(rewards)

print("Average reward:", average_reward)
print("Best episode:", max(rewards))
print("Worst episode:", min(rewards))