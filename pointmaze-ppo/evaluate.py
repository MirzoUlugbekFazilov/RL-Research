import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

# Same environment used during training
env = gym.make("PointMaze_UMazeDense-v3")

# Load trained PPO
model = PPO.load("models/ppo_umaze_dense.pt")

num_episodes = 300

successful_episodes = 0
successful_steps = []
episode_rewards = []

for episode in range(num_episodes):

    obs, info = env.reset()

    total_reward = 0
    steps = 0
    success = False
    goal_step = None

    while True:

        # PPO chooses action
        action, _ = model.predict(
            obs,
            deterministic=True
        )

        # Environment executes action
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        # Calculate distance to goal
        distance_to_goal = (
            (
                (obs["achieved_goal"][0] - obs["desired_goal"][0]) ** 2
                +
                (obs["achieved_goal"][1] - obs["desired_goal"][1]) ** 2
            )
            ** 0.5
        )

        # Check goal
        if distance_to_goal <= 0.5:

            success = True

            # Record the FIRST step where goal was reached
            goal_step = steps

            break

        # Episode ended without reaching goal
        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    if success:
        successful_episodes += 1
        successful_steps.append(goal_step)


# -----------------------------
# Calculate final statistics
# -----------------------------

success_rate = (
    successful_episodes / num_episodes
) * 100

average_reward = (
    sum(episode_rewards) / num_episodes
)

if successful_steps:

    average_steps_to_goal = (
        sum(successful_steps)
        / len(successful_steps)
    )

else:

    average_steps_to_goal = 0


print("\n========== EVALUATION RESULTS ==========")

print(f"Episodes evaluated: {num_episodes}")

print(
    f"Successful episodes: "
    f"{successful_episodes}"
)

print(
    f"Failed episodes: "
    f"{num_episodes - successful_episodes}"
)

print(
    f"Success rate: "
    f"{success_rate:.2f}%"
)

print(
    f"Average reward: "
    f"{average_reward:.2f}"
)

print(
    f"Average steps to goal "
    f"(successful episodes): "
    f"{average_steps_to_goal:.2f}"
)

print("========================================")

env.close()