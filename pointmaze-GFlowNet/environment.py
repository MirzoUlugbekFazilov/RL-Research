import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)


# Distance under which PointMaze declares the goal reached.
GOAL_RADIUS = 0.45


class PointMazeWrapper:

    def __init__(self, max_steps=100):

        # continuing_task=False is required, otherwise the episode never
        # terminates when the goal is reached and "success" can never be
        # detected.
        self.env = gym.make(
            "PointMaze_UMaze-v3",
            max_episode_steps=max_steps,
            continuing_task=False
        )

        # Four discrete movements
        self.actions = np.array([
            [1.0, 0.0],    # right
            [-1.0, 0.0],   # left
            [0.0, 1.0],    # up
            [0.0, -1.0]    # down
        ], dtype=np.float32)

        self.num_actions = len(self.actions)

        # position (2) + velocity (2) + goal (2)
        self.state_dim = 6

        self.goal_radius = GOAL_RADIUS

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        state = self.get_state(obs)

        return state

    def step(self, action):

        continuous_action = self.actions[action]

        obs, reward, terminated, truncated, info = self.env.step(
            continuous_action
        )

        state = self.get_state(obs)

        return state, reward, terminated, truncated

    def get_state(self, obs):

        # observation = [x, y, vx, vy]; the velocity matters because the
        # point mass has momentum, so a position-only policy cannot brake.
        robot = obs["observation"]
        goal = obs["desired_goal"]

        state = np.concatenate([
            robot,
            goal
        ])

        return state.astype(np.float32)

    def close(self):
        self.env.close()
