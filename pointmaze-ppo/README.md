# PointMaze — PPO

Proximal Policy Optimization on `PointMaze_UMazeDense-v3`, a continuous-control,
goal-conditioned navigation task from
[Gymnasium-Robotics](https://robotics.farama.org/) running on MuJoCo. Trained
with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

This is the harder of the two PPO projects in this repository: unlike CartPole,
the task has a **local optimum that a naively configured agent falls straight
into**, and most of the tuning below exists to escape it.

---

## Task

| | |
|---|---|
| Environment | `PointMaze_UMazeDense-v3` (MuJoCo) |
| Observation | Dict — `observation` (x, y, vx, vy), `achieved_goal` (x, y), `desired_goal` (x, y) |
| Actions | Continuous 2-D force applied to the point mass |
| Reward | Dense, `exp(−distance)` to the goal |
| Success | Euclidean distance to goal ≤ **0.5** |
| Maze | U-shaped corridor; start and goal are randomised each episode |

### The local optimum

The maze is a **U**. The dense reward `exp(−d)` is *greedy*: it rewards being
closer to the goal right now. But escaping one arm of the U requires moving
**away** from the goal first — paying reward now for a payoff roughly 100 steps
later. An agent optimising the dense signal short-sightedly learns to press
itself against the dividing wall, as close to the goal in straight-line distance
as it can get, and never goes around.

Everything interesting in `train.py` is about seeing past that dip.

---

## Algorithm

**PPO** with SB3's `MultiInputPolicy` — required because the observation is a
dictionary; the policy encodes each sub-space and concatenates the features
before the actor and critic heads.

Training runs **16 parallel environments** in a `DummyVecEnv`. Parallelism here
is for gradient variance, not throughput: 16 envs × 256 steps gives the same
4096-transition rollout as a single-env baseline, but the transitions come from
16 independent start/goal pairs. `DummyVecEnv` beats `SubprocVecEnv` at this
scale because the environment already runs at ~25k steps/s — process IPC would
cost more than the simulation itself.

---

## Hyperparameters and why

| Parameter | Value | Reasoning |
|---|---|---|
| `gamma` | **0.995** | The single most important setting. 0.995 gives a ~200-step horizon vs ~100 at 0.99 — long enough for the value function to see past the "move away from the goal" dip. **Measured: 100% vs 93%.** |
| `ent_coef` | **0.0** | Not needed *because* `gamma` is 0.995. Note that `ent_coef=0.005` at `gamma=0.99` reaches 99.67% on its own — keeping exploration alive and lengthening the horizon are two separate exits from the same trap. You need one, not both. |
| `learning_rate` | `3e-4`, **linearly annealed to 0** | Annealing is what sharpens the final policy; a fixed rate leaves it still jittering at the end of training. |
| `net_arch` | `pi=[256,256]`, `vf=[256,256]` | SB3's default 64×64 is too thin to encode a distinct route for each start/goal pair. |
| `n_envs` | 16 (`DummyVecEnv`) | Variance reduction across independent start/goal pairs. |
| `n_steps` | 256 | 16 × 256 = 4096 transitions per rollout. |
| `batch_size` | 256 | |
| `n_epochs` | 10 | |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `max_grad_norm` | 0.5 | |
| `total_timesteps` | 1,000,000 | |
| `device` | `cpu` | An MLP this size runs faster on CPU than on MPS. |
| `seed` | 0 | |

### Environment configuration

Trained with the **default** `continuing_task=True, reset_target=False`. This
is deliberate. Training with `reset_target=True` — where the goal teleports the
moment you arrive — was measurably worse: it teaches the agent to *drift toward*
the goal region rather than commit to it (**49% success, 100 steps to goal**).

### Checkpoint selection

The model is saved **unconditionally at the end of training**, not as "best of N
evaluations". Picking the best of several noisy 50-episode probes reliably
selects the luckiest *measurement* rather than the best *policy* — that exact
mistake once produced a "100%" checkpoint that scored **49%** on a real
300-episode evaluation.

---

## Results

Measured by running `evaluate.py` on the committed checkpoint —
300 episodes, deterministic action selection, randomised start/goal pairs.

| Metric | Value |
|---|---|
| Episodes evaluated | 300 |
| Successful episodes | **300** |
| Failed episodes | 0 |
| **Success rate** | **100.00%** |
| Average return | 12.75 |
| Average steps to goal | **47.74** |

The agent solves every randomised start/goal configuration in the U-maze, and
routes around the dividing wall rather than pressing against it.

### Extended metrics

300 episodes on fixed seeds (`30000 + episode`) so the numbers are
reproducible. Success rates carry a 95% Wilson confidence interval.

| Metric | Value |
|---|---|
| Success rate | **100.00%** (300/300) |
| 95% CI | 98.74 – 100% |

**Steps to goal**

| n | Mean | Std | Min | p25 | Median | p75 | p90 | p95 | Max |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 48.82 | 27.68 | 7 | 24 | 43 | 70 | 91.2 | 101.1 | **121** |

**Episode return**

| n | Mean | Std | Min | p25 | Median | p75 | p95 | Max |
|---|---|---|---|---|---|---|---|---|
| 300 | 12.88 | 3.96 | 4.10 | 9.78 | 11.45 | 15.30 | 21.34 | 24.57 |

**Success as a function of step budget**

| Budget | 25 | 50 | 75 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|
| Solved | 29.7% | 60.3% | 80.3% | 94.3% | **100%** | 100% | 100% |

The spread is the story here. Mean steps-to-goal (48.8) is close to the median
(43), but the distribution has a long right tail: the easiest start/goal pairs
are solved in **7** steps and the hardest need **121**. That 17× range is the
maze geometry, not policy inconsistency — pairs on the same arm of the U are a
straight line apart, while pairs on opposite arms require the full detour
around the dividing block. Note that **5.7% of episodes need more than 100
steps**, so a 100-step evaluation budget would have understated this policy at
94.3%.

The mean return of 12.88 with a std of 3.96 reflects the same geometry: under
the dense `exp(−d)` reward, a short trip accumulates less total reward than a
long one that spends many steps near the goal, so return is a poor proxy for
quality on this task and success rate is the metric that matters.

**Model and runtime**

| Metric | Value |
|---|---|
| Policy parameters | 136,965 (all trainable) |
| Checkpoint size | 1.6 MB |
| Inference latency | 0.136 ms/step |
| Evaluation throughput | ~7,300 env steps/s (MuJoCo + policy) |
| Total steps evaluated | 14,645 |

Measured single-threaded on CPU (Apple M1 Pro), batch size 1.

> The headline table above (47.74 mean steps) comes from `evaluate.py`, which
> uses **unseeded** resets; the extended table uses fixed seeds. The small
> difference between 47.74 and 48.82 is the sampling variance of which
> start/goal pairs get drawn, not a discrepancy.

### Ablations recorded during development

| Configuration | Success rate |
|---|---|
| Final configuration (`γ=0.995`, `ent_coef=0`) | **100%** |
| `γ=0.99`, `ent_coef=0.005` | 99.67% |
| `γ=0.99`, `ent_coef=0` | 93% |
| `reset_target=True` during training | 49% |
| Best-of-N checkpoint selection on 50-episode probes | 49% (measurement artefact) |

---

## Usage

```bash
pip install gymnasium gymnasium-robotics stable-baselines3 torch numpy mujoco

python train.py       # 1M timesteps across 16 envs; runs a 300-episode eval at the end
python evaluate.py    # 300 deterministic episodes; prints success rate and steps to goal
```

Run both from inside this folder — model paths are relative to the working
directory. Training also runs the identical 300-episode evaluation inline
(`RUN_FINAL_EVAL = True`) so a training run reports a real success rate rather
than a proxy metric.

---

## Project structure

```
pointmaze-ppo/
├── train.py                       # PPO training + inline 300-episode evaluation
├── evaluate.py                    # Standalone 300-episode evaluation
├── models/
│   └── ppo_umaze_dense.pt         # Trained SB3 checkpoint
└── README.md
```

---

## Reproducibility

| Component | Version |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.10.0 |
| Gymnasium | 1.3.0 |
| Gymnasium-Robotics | 1.4.2 |
| Stable-Baselines3 | 2.9.0 |
| MuJoCo | 3.11.0 |
| Hardware | Apple M1 Pro (CPU) |

Training is seeded (`SEED = 0`) for both the vectorised environments and the
PPO agent. Evaluation uses unseeded resets, so the 300 episodes are a fresh
random sample of start/goal pairs on every run.

---

## Related

- [`../pointmaze-GFlowNet`](../pointmaze-GFlowNet) — the same maze solved with a
  goal-conditioned GFlowNet under subtrajectory balance.
- [`../cartpole-PPO`](../cartpole-PPO) — the simpler PPO baseline.
