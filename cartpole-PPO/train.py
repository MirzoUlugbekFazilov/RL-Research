from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

# Save location: <project>/models/cartpole_model.pt, regardless of working dir
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "cartpole_model.pt"

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Create the RL agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Train the agent
model.learn(total_timesteps=70_000)

# Save the trained policy as a real PyTorch checkpoint.
# model.save() would write SB3's own .zip archive instead.
model.policy.save(MODEL_PATH)

# Confirm the file actually landed on disk
print(f"Model saved to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")

env.close()