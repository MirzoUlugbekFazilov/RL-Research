# CartPole — PPO

Proximal Policy Optimization baseline on `CartPole-v1`, trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). This is the
reference point the GFlowNet experiments in this repository are compared
against: a standard, well-understood on-policy RL algorithm on a task it is
known to solve.

---

## Task

| | |
|---|---|
| Environment | `CartPole-v1` (Gymnasium) |
| Observation | 4 continuous values — cart position, cart velocity, pole angle, pole angular velocity |
| Actions | 2 discrete — push left, push right |
| Reward | +1 per timestep the pole stays upright |
| Episode ends | Pole falls, cart leaves the track, or 500 steps elapse |
| Solved | Average reward ≥ 475 over 100 consecutive episodes |

The maximum achievable return is **500**, the truncation limit.

---

## Algorithm

**PPO (Proximal Policy Optimization)** — an on-policy, actor-critic policy
gradient method. PPO alternates between collecting a batch of interaction data
under the current policy and taking several gradient steps on that batch. The
key idea is the *clipped surrogate objective*: the ratio between the new and
old action probabilities is clipped to a trust region, so a single update
cannot move the policy far enough to collapse it. This gives PPO much of the
stability of trust-region methods without the second-order machinery.

The agent used here is the SB3 `MlpPolicy` — a shared feed-forward trunk with
two heads:

- **Actor** — outputs a categorical distribution over the two actions.
- **Critic** — outputs a state-value estimate `V(s)`, used to compute
  Generalized Advantage Estimation targets.

Everything except the learning rate and the training budget is left at the SB3
default, deliberately: the point of this baseline is that a stock configuration
solves the task, so any comparison against it is a comparison against a fair
reference rather than a hand-tuned one.

---

## Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Policy | `MlpPolicy` | SB3 default 64×64 actor-critic |
| Learning rate | `3e-4` | Constant |
| Total timesteps | `70,000` | |
| Rollout length (`n_steps`) | 2048 | SB3 default |
| Batch size | 64 | SB3 default |
| Epochs per update | 10 | SB3 default |
| Discount `γ` | 0.99 | SB3 default |
| GAE `λ` | 0.95 | SB3 default |
| Clip range | 0.2 | SB3 default |

### Checkpoint format

`train.py` calls `model.policy.save(...)`, **not** `model.save(...)`. The
latter writes SB3's own `.zip` archive containing the full training state;
the former writes a plain PyTorch checkpoint of the policy network, which is
what `evaluate.py` reloads via `ActorCriticPolicy.load`. If you swap one for
the other, loading will fail.

---

## Results

Measured by running `evaluate.py` on the committed checkpoint —
100 episodes, deterministic (argmax) action selection.

| Metric | Value |
|---|---|
| Episodes evaluated | 100 |
| Average return | **500.00 / 500** |
| Best episode | 500 |
| Worst episode | 500 |
| Success rate (return ≥ 475) | **100%** |

The policy is saturated: every one of the 100 evaluation episodes ran to the
500-step truncation limit without the pole ever falling. The environment is
solved, and the variance is exactly zero.

---

## Usage

```bash
pip install gymnasium stable-baselines3 torch

python train.py       # ~70k timesteps, writes models/cartpole_model.pt
python evaluate.py    # 100 deterministic episodes, prints return statistics
```

`train.py` resolves the model directory relative to the script itself, so it
can be run from any working directory. `evaluate.py` does the same.

---

## Project structure

```
cartpole-PPO/
├── train.py                   # PPO training loop (Stable-Baselines3)
├── evaluate.py                # 100-episode deterministic evaluation
├── models/
│   └── cartpole_model.pt      # Trained policy weights
└── README.md
```

---

## Reproducibility

Results above were produced with:

| Component | Version |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.10.0 |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| Hardware | Apple M1 Pro (CPU) |

`train.py` does not fix a random seed, so retrained checkpoints will differ
slightly. Evaluation of the committed checkpoint is deterministic given the
same seeds and library versions.

---

## Related

- [`../cartpole-GFlowNet`](../cartpole-GFlowNet) — the same task solved with a
  GFlowNet under a Detailed Balance objective.
- [`../pointmaze-ppo`](../pointmaze-ppo) — PPO scaled to a continuous-control
  goal-conditioned maze.
