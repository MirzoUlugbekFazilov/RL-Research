# RL Research — GFlowNets as Control Algorithms

Four self-contained reinforcement learning projects comparing **Generative Flow
Networks** against **PPO** on the same two control tasks.

The question being tested: GFlowNets are designed to *sample* objects in
proportion to a reward, not to maximise return. Can that flow-matching objective
be used as a control algorithm — and how does it compare to a dedicated
policy-gradient method on tasks PPO is known to solve?

---

## Projects

| Project | Task | Algorithm | Success rate |
|---|---|---|---|
| [`cartpole-PPO`](cartpole-PPO) | `CartPole-v1` | PPO (Stable-Baselines3) | **100%** — 500.00/500 avg return |
| [`cartpole-GFlowNet`](cartpole-GFlowNet) | `CartPole-v1` | GFlowNet, **detailed balance** | **100%** — 300/300 episodes |
| [`pointmaze-ppo`](pointmaze-ppo) | `PointMaze_UMazeDense-v3` | PPO (Stable-Baselines3) | **100%** — 300/300, 47.7 steps to goal |
| [`pointmaze-GFlowNet`](pointmaze-GFlowNet) | `PointMaze_UMaze-v3` (sparse) | GFlowNet, **subtrajectory balance** | **100%** — 100/100, 50.2 steps to goal |

All figures are measured by running each project's `evaluate.py` on the
committed checkpoint. Each project's README documents its algorithm,
hyperparameters, design decisions and full results.

---

## Findings

**Both methods saturate both tasks.** The headline is not that one wins — it is
that an objective containing *no reward maximisation anywhere in it* reaches the
same perfect control policy as PPO on both benchmarks.

The differences are in what each method needed to get there:

- **CartPole** — indistinguishable. Both reach 500/500 with zero variance.
- **PointMaze** — the GFlowNet solves the maze from a **sparse** reward
  (`R = 1` at the goal). PPO was given a **dense** shaping signal `exp(−d)`,
  and that shaping is precisely what created its hardest failure mode: the dense
  reward is greedy, escaping one arm of the U means moving *away* from the goal
  first, and a short discount horizon traps the agent against the dividing wall.
  Fixing it required lengthening the horizon to `γ = 0.995` (93% → 100%).

Two parameterisations of the flow objective are used, and the choice matters:

- **Detailed balance** (CartPole) — learns a state flow `log F(s)` and checks
  balance on every transition. Chosen over trajectory balance, whose single
  scalar `log Z` has to travel ~250 nats by gradient descent before anything
  works.
- **Subtrajectory balance** (PointMaze) — carries the terminal anchor up to 8
  steps back per update. One-step updates move it a single step at a time, far
  too slow for the 40–70 step trips around the U.

---

## Repository layout

```
.
├── cartpole-PPO/           # PPO baseline on CartPole-v1
├── cartpole-GFlowNet/      # Detailed-balance GFlowNet on CartPole-v1
├── pointmaze-ppo/          # PPO on the dense-reward U-maze
└── pointmaze-GFlowNet/     # Subtrajectory-balance GFlowNet on the sparse U-maze
```

Every project follows the same structure: `train.py`, `evaluate.py`, a
`models/` directory holding the committed checkpoint, and a `README.md`.

---

## Getting started

```bash
pip install torch numpy gymnasium stable-baselines3 gymnasium-robotics mujoco

cd cartpole-GFlowNet     # or any other project
python evaluate.py       # reproduce the reported numbers from the checkpoint
python train.py          # retrain from scratch
```

Run the scripts from inside a project directory — model paths are relative to
the working directory.

---

## Environment

Results in this repository were produced with:

| Component | Version |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.10.0 |
| Gymnasium | 1.3.0 |
| Gymnasium-Robotics | 1.4.2 |
| Stable-Baselines3 | 2.9.0 |
| MuJoCo | 3.11.0 |
| Hardware | Apple M1 Pro (CPU) |

---

## A note on comparability

The two PointMaze projects do **not** use identical environments — they differ
in reward signal (dense vs sparse), action space (2-D continuous vs 4 discrete)
and success radius (0.5 vs 0.45). Step-to-goal counts are therefore not a
like-for-like benchmark, and the per-project READMEs flag this where the numbers
appear side by side.
