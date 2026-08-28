# PointMaze — GFlowNet (Subtrajectory Balance)

A **goal-conditioned Generative Flow Network** that navigates
`PointMaze_UMaze-v3`, trained with a subtrajectory balance objective, hindsight
goal relabelling, and an edge-flow parameterisation.

This is the most involved project in the repository. Where the
[CartPole GFlowNet](../cartpole-GFlowNet) applies flow matching to a short,
dense, 2-action task, this one scales it to **sparse reward, continuous
dynamics, and a goal that changes every episode** — the setting where a
GFlowNet's flow function has to represent an entire family of policies at once.

---

## Task

| | |
|---|---|
| Environment | `PointMaze_UMaze-v3` (MuJoCo, **sparse** reward) |
| State | 6 values — position (x, y), velocity (vx, vy), goal (gx, gy) |
| Actions | **4 discrete** — right, left, up, down (unit forces, wrapping the continuous action space) |
| Reward | Sparse: `R = 1` on reaching the goal, nothing otherwise |
| Goal radius | **0.45** |
| Training horizon | 100 steps |
| Evaluation horizon | 150 steps |

Velocity is part of the state because the point mass has **momentum** — a
position-only policy physically cannot brake.

The environment is wrapped with `continuing_task=False`. This is required, not
cosmetic: with the default `True`, the episode never terminates when the goal is
reached and success can never be detected at all.

### Why evaluation gets 150 steps, not 100

The point mass tops out at ~0.052 units of displacement per step, and the
longest start/goal pairs sit at opposite ends of the U, about 6 units of
corridor apart. Those episodes need **~115 steps from any policy, however
good**. Scoring them against a 100-step budget marks physically unreachable
episodes as policy failures and censors the metric. Measured over 500 seeds the
trained policy reaches the goal on every one, with a worst case of 112 steps —
so 150 leaves margin, and is still half of the 300-step default the environment
ships with.

---

## Algorithm

### Edge-flow parameterisation — `gflownet.py`

The network outputs **one log-flow per action**, `log F(s, a, g)`. Everything
else is derived from it:

```
P_F(a | s, g) = softmax_a log F(s, a, g)
log F(s, g)   = logsumexp_a log F(s, a, g)
```

Because the policy and the state flow come from the same tensor, they **can
never disagree** — a failure mode that separate policy and flow heads permit.
Both are anchored to the terminal reward through the balance condition.

### Fourier features

A plain MLP on raw coordinates smooths across the maze walls, which is exactly
the mistake that makes the agent drive into the central block instead of going
around it. Position and goal are therefore lifted through sinusoidal features at
frequencies `(1, 2, 4)·π`, letting the network represent that boundary sharply.
Velocities are divided by `VELOCITY_SCALE = 5.0`, since the point mass reaches
speeds an order of magnitude above the position range.

Input features: raw state (6) + displacement to goal (2) + distance to goal (1)
+ sin/cos of every frequency for both position and goal (24) → 3×256 ReLU MLP
→ 4 log-edge-flows.

### Subtrajectory balance

For a run of `n` consecutive steps starting at `s_i`:

```
log F(s_i, a_i) + Σ_{j>i} log P_F(a_j | s_j) + n · STEP_COST  =  log F(s_end)
```

with `log F(s_end) = log R = 0` when the run ends inside the goal radius. The
backward policy is deterministic (`P_B = 1`), so every state along a sampled
trajectory has a single parent.

At `n = 1` this is exactly **detailed balance**. Longer `n` is what makes it
work here: one-step updates move the terminal anchor backwards a single step per
update, far too slow for the 40–70 step trips around the U. Sampling
subtrajectories up to length **8** propagates the anchor in one update.

The loss is a **Huber** loss on the residual, divided by `STEP_COST` so that
constant remains a policy-sharpness knob instead of also rescaling gradients.

### The step cost, and why it must exceed log(4)

Every transition costs `STEP_COST = 4.0` in log space. It **has to exceed
`log(num_actions) = 1.39`**, or the sheer number of wandering trajectories
outweighs the short ones and the flow stops pointing at the goal at all. The
margin above 1.39 is what separates the actions, because the `logsumexp` backup
adds about `log(4)` of path-counting entropy per step.

### Hindsight goal relabelling

Whole **episodes** are stored, padded, rather than loose transitions —
subtrajectory balance needs consecutive steps, and keeping episodes means a
fresh set of hindsight goals can be drawn every time an episode is replayed
instead of freezing a few relabelled copies at collection time.

When a stored episode is replayed, its goal is drawn from three sources:

| Source | Probability | Effect |
|---|---|---|
| The episode's real goal | 0.20 | Grounds the flow in the actual task |
| **Future** — a position from later in that same episode | 0.30 | Teaches short hops |
| **Pool** — any position the agent has ever visited | 0.50 | Makes the flow reach across the whole U to distant goals |

### Half-greedy rollouts

**50% of episodes are collected with pure argmax**, matching how the policy is
evaluated. Without them the buffer never contains the states a greedy run
actually visits, so the policy's mistakes compound off-distribution. The other
half samples from `P_F` under an ε-greedy schedule (1.0 → 0.05 over 600
episodes).

### Stabilisers

- **Polyak target network** (`τ = 0.005`) for the bootstrapped `log F(s_end)`.
- **Cosine-annealed learning rate** down to `LR/20`, so the flow settles instead
  of chasing a moving bootstrap target — which is what makes the greedy policy
  flip between actions in ambiguous states.
- **Validity masking** — a source state already inside the goal radius is
  terminal and has no outgoing flow, so those samples are excluded from the loss.
- **Best-checkpoint selection** on a 100-episode greedy evaluation every 100
  episodes. These evaluation episodes are *not* part of the training budget.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 3,000 |
| Max steps (training / evaluation) | 100 / 150 |
| Learning rate | `1e-3` (Adam, cosine → `5e-5`) |
| Batch size | 256 |
| Updates per episode | 32 |
| Max subtrajectory length | 8 |
| Step cost | 4.0 |
| Target update rate `τ` | 0.005 |
| Real / future / pool goal probability | 0.2 / 0.3 / 0.5 |
| Greedy rollout fraction | 0.5 |
| ε schedule | 1.0 → 0.05 over 600 episodes |
| Hidden dim | 256 (3 layers) |
| Gradient clipping | max-norm 10.0 |
| Seed | 0 |

---

## Results

Measured by running `evaluate.py` on the committed checkpoint —
100 episodes, deterministic (argmax) action selection, randomised start/goal
pairs, 150-step budget.

| Metric | Value |
|---|---|
| Episodes evaluated | 100 |
| Successful episodes | **100** |
| Failed episodes | 0 |
| **Success rate** | **100.00%** |
| Average reward | 1.00 / 1.00 |
| Average steps to goal | **50.19** |

A broader sweep over **500 seeds** during development also reached the goal on
every episode, with a worst case of 112 steps.

### Comparison with PPO on the same maze

| | GFlowNet (SubTB) | [PPO](../pointmaze-ppo) |
|---|---|---|
| Success rate | 100% | 100% |
| Average steps to goal | 50.19 | 47.74 |
| Reward signal | **Sparse** (`R = 1` at goal) | **Dense** (`exp(−d)`) |
| Action space | 4 discrete | 2-D continuous |
| Goal radius | 0.45 | 0.5 |
| Evaluation episodes | 100 | 300 |

**These are not a like-for-like benchmark** — the environments differ in reward
signal, action space and success radius, so the step counts are not directly
comparable. What the pairing does show is that a flow-matching objective reaches
the same perfect success rate on the U-maze **from a sparse reward**, where PPO
was given a dense shaping signal (and needed a lengthened horizon to avoid being
trapped by it).

---

## Usage

```bash
pip install gymnasium gymnasium-robotics torch numpy mujoco

python train.py       # 3,000 episodes; greedy eval every 100; saves best checkpoint
python evaluate.py    # 100 deterministic episodes; prints success rate and steps to goal
```

Run both from inside this folder — model paths are relative to the working
directory. CUDA is used automatically if available; otherwise CPU.

---

## Project structure

```
pointmaze-GFlowNet/
├── environment.py                    # PointMazeWrapper: discrete actions, 6-D state, goal radius
├── gflownet.py                       # Edge-flow network with Fourier features
├── train.py                          # Subtrajectory balance, episode store, hindsight relabelling
├── evaluate.py                       # 100-episode deterministic evaluation
├── models/
│   └── gflownet_pointmaze.pt         # Best greedy-eval checkpoint
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
| MuJoCo | 3.11.0 |
| Hardware | Apple M1 Pro (CPU) |

Training seeds `random`, `numpy` and `torch` with `SEED = 0`. In-training greedy
evaluation uses the disjoint seed range `100000 + index`; `evaluate.py` uses
unseeded resets, so its 100 episodes are a fresh random sample on every run.

---

## Related

- [`../pointmaze-ppo`](../pointmaze-ppo) — PPO on the dense-reward variant of the
  same maze.
- [`../cartpole-GFlowNet`](../cartpole-GFlowNet) — the same flow-matching idea on
  a simpler task, using detailed balance and a separate flow head.
