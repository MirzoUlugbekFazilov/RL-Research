import numpy as np

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register_envs(gymnasium_robotics)


# -----------------------------
# Settings
# -----------------------------

ENV_ID = "PointMaze_UMazeDense-v3"

TOTAL_TIMESTEPS = 1_000_000

# The env runs at ~25k steps/s, so DummyVecEnv beats SubprocVecEnv here:
# process IPC would cost more than the simulation itself.
N_ENVS = 16

SEED = 0

MODEL_PATH = "models/ppo_umaze_dense.pt"

# Same threshold evaluate.py uses.
GOAL_RADIUS = 0.5

RUN_FINAL_EVAL = True


def linear_schedule(initial_value):
    """Decay the learning rate to 0 so the policy sharpens as it converges."""

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


# -----------------------------
# Environment
# -----------------------------

# Default env settings on purpose: continuing_task=True, reset_target=False.
# Training with reset_target=True (goal teleports the moment you arrive) was
# measurably worse -- it teaches the agent to drift toward the goal region
# instead of committing to it (49% success, 100 steps to goal).
train_env = make_vec_env(
    ENV_ID,
    n_envs=N_ENVS,
    seed=SEED,
    vec_env_cls=DummyVecEnv,
)


# -----------------------------
# Agent
# -----------------------------

model = PPO(
    "MultiInputPolicy",
    train_env,

    # 16 envs x 256 = 4096 transitions per rollout. Same rollout size as the
    # single-env baseline, but collected from 16 independent start/goal pairs,
    # which is what cuts the gradient variance.
    n_steps=256,
    batch_size=256,
    n_epochs=10,

    # Annealing to 0 over the run is what sharpens the final policy; a fixed
    # rate leaves it still jittering at the end.
    learning_rate=linear_schedule(3e-4),

    # 0.995 -> ~200 step horizon, vs ~100 at 0.99. The dense reward exp(-d) is
    # greedy: escaping one arm of the U means moving AWAY from the goal first,
    # costing reward now for a payoff ~100 steps later. The longer horizon is
    # what lets the value function see past that dip. Measured 100% vs 93%.
    gamma=0.995,
    gae_lambda=0.95,

    # 0.0 because gamma=0.995 already solves the local optimum. Note that
    # ent_coef=0.005 at gamma=0.99 reaches 99.67% on its own -- keeping
    # exploration alive and lengthening the horizon are two separate ways out
    # of the same "hug the wall" trap, so you need one of them, not both.
    ent_coef=0.0,

    clip_range=0.2,
    max_grad_norm=0.5,

    # Default is 64x64, too thin to encode a distinct route for each
    # start/goal pair.
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),

    # MlpPolicy at this size runs faster on CPU than on MPS.
    device="cpu",

    seed=SEED,
    verbose=1,
)


model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Saved unconditionally, not "best of N evaluations". Picking the best of
# several noisy 50-episode probes reliably selects the luckiest measurement
# rather than the best policy -- that mistake produced a "100%" checkpoint
# that scored 49% on the real 300-episode evaluation.
model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

train_env.close()


# -----------------------------
# Final evaluation
# -----------------------------

if RUN_FINAL_EVAL:

    # Identical to evaluate.py, just run inline so training reports a real
    # number instead of a proxy.
    env = gym.make(ENV_ID)

    successes = 0
    steps_to_goal = []

    for episode in range(300):

        obs, _ = env.reset()
        steps = 0

        while True:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            steps += 1

            distance = np.linalg.norm(
                obs["achieved_goal"] - obs["desired_goal"]
            )

            if distance <= GOAL_RADIUS:
                successes += 1
                steps_to_goal.append(steps)
                break

            if terminated or truncated:
                break

    print(f"Success rate: {100 * successes / 300:.2f}%")
    print(
        f"Average steps to goal: "
        f"{np.mean(steps_to_goal) if steps_to_goal else 0:.2f}"
    )

    env.close()
