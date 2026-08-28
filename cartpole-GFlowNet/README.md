# CartPole — GFlowNet (Detailed Balance)

A **Generative Flow Network** trained to control `CartPole-v1`, using a
Detailed Balance objective with a replay buffer and a Polyak-averaged target
network. The aim is to test whether a GFlowNet — a method designed for
*sampling* diverse structured objects in proportion to a reward — can be used
as a control algorithm on a standard RL benchmark, and how it compares to the
PPO baseline in [`../cartpole-PPO`](../cartpole-PPO).

---

## Task

| | |
|---|---|
| Environment | `CartPole-v1` (Gymnasium) |
| State | 5 values — the 4 CartPole observations **+ normalised timestep** `t / 500` |
| Actions | 2 discrete — push left, push right |
| Episode ends | Pole falls, cart leaves the track, or 500 steps elapse |

### Why the timestep is part of the state

This is the design decision the whole method rests on. CartPole-v1 truncates at
500 steps, so without a clock the same observation could occur at two different
depths of the trajectory and the flow through it would be ill-defined.

Adding `t` makes the state graph a proper **DAG**, which buys three things at
once:

1. Truncation at step 500 becomes a genuine terminal state rather than an
   arbitrary cutoff.
2. Every state has **exactly one parent**, so the backward policy is
   deterministic and `log P_B = 0` — it drops out of the objective entirely.
3. The flow function is well-defined everywhere.

---

## Algorithm

### GFlowNets in one paragraph

A GFlowNet treats trajectory generation as **flow through a directed acyclic
graph**. Flow enters at the initial state, is conserved through every
intermediate state, and exits at terminal states in proportion to their reward.
Train the flow to satisfy a conservation condition and the induced forward
policy samples terminal states with probability proportional to `R` — a
maximum-entropy solution by construction, rather than a maximum-return one.

### Detailed Balance, not Trajectory Balance

The obvious parameterisation, **Trajectory Balance**, uses a single scalar
`log Z` for an entire trajectory. On CartPole that scalar has to travel roughly
250 nats by gradient descent before anything works at all — slow and fragile.

**Detailed Balance** instead learns a state flow `log F(s)` and checks the
balance condition on *every transition*:

```
log F(s) + log P_F(a | s)  =  1 / TEMPERATURE + log F(s')
```

with `log F(terminal) = 0`. The loss is the mean squared residual of that
equation. Two consequences follow directly:

- **Temperature stops controlling training time.** Under Trajectory Balance it
  governs how long the run takes to become useful; here it only rescales the
  range of `log F(s)`, making it a pure policy-sharpness knob.
- **Off-policy data stays valid.** The condition holds for *any* transition
  regardless of which policy generated it — so a replay buffer is sound, and
  every transition can be reused many times instead of being discarded after
  one gradient step. This is where the sample efficiency comes from.

### Network — `gflownet.py`

A shared 2×128 ReLU trunk feeding two heads:

| Head | Output |
|---|---|
| `policy_head` | 2 logits → `P_F(a \| s)` |
| `flow_head` | scalar `log F(s)`, multiplied by `FLOW_SCALE = 100` |

The `FLOW_SCALE` factor matters. Flow values reach several hundred at start
states, because `log F(s) ≈ (500 − t) · (log 2 + 1/T)`. Multiplying a small raw
output by a fixed scale keeps the network's own activations at order 1 while
still spanning the required range.

### Training tricks that are load-bearing

- **Target network** (`TARGET_TAU = 0.005`) — Detailed Balance bootstraps on
  `log F(s')`, so without a Polyak-averaged target the flow chases its own
  output and diverges.
- **No entropy bonus** — a GFlowNet policy is maximum-entropy *by construction*;
  adding an explicit entropy term would double-count.
- **Rollouts under `no_grad`** — gradients come only from replayed batches, so
  there is no reason to build a graph during collection.
- **Best-checkpoint selection on greedy eval** — the saved model is the one with
  the best argmax evaluation, not the last one, since the sampling policy and
  the greedy policy are different objects.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 3,000 |
| Learning rate | `1e-3` (Adam) |
| Temperature | 2.0 |
| Batch size | 256 |
| Updates per episode | 32 |
| Replay buffer capacity | 100,000 transitions |
| Warm-up transitions | 1,000 |
| Target network rate `τ` | 0.005 |
| Gradient clipping | max-norm 10.0 |
| Hidden size | 128 |
| Seed | 42 |

---

## Results

Measured by running `evaluate.py` on the committed checkpoint —
300 episodes, deterministic (argmax) action selection, fixed evaluation seeds
`10000 + episode` that are disjoint from the training seeds.

| Metric | Value |
|---|---|
| Episodes evaluated | 300 |
| Successful episodes (return ≥ 475) | **300** |
| Failed episodes | 0 |
| **Success rate** | **100.00%** |
| Average return | **500.00 / 500** |
| Best episode | 500 |
| Worst episode | 500 |

Every single one of the 300 held-out episodes reached the 500-step truncation
limit. **Zero variance** — the worst episode is also the best episode.

### Extended metrics

The same 300 held-out seeds, run under both action-selection rules. For a
GFlowNet this comparison is the meaningful one: `P_F` is a *sampler*, and the
greedy policy is a different object derived from it.

**Return distribution**

| Policy | n | Mean | Std | Min | p25 | Median | p75 | p95 | Max |
|---|---|---|---|---|---|---|---|---|---|
| Greedy (argmax) | 300 | **500.00** | **0.00** | 500 | 500 | 500 | 500 | 500 | 500 |
| Sampled (`P_F`) | 300 | 498.55 | 15.73 | 285 | 500 | 500 | 500 | 500 | 500 |

**Success rate** (return ≥ 475, with 95% Wilson confidence interval)

| Policy | Successes | Rate | 95% CI | Episodes at exactly 500 |
|---|---|---|---|---|
| Greedy | 300/300 | **100.00%** | 98.74 – 100% | 300 |
| Sampled | 297/300 | **99.00%** | 97.10 – 99.66% | 297 |

This is the most informative result in the project. The GFlowNet was never
trained to maximise return — it was trained to make flow consistent — yet
**sampling from `P_F` still balances the pole for the full 500 steps in 99% of
episodes**. The sampler is not a degenerate deterministic policy: it retains
enough entropy that 3 of 300 episodes end early (worst case 285 steps), and
that residual stochasticity is exactly what the maximum-entropy objective is
supposed to preserve. The greedy read-out of the same flow is perfect.

**Model and runtime**

| Metric | Value |
|---|---|
| Parameters | 17,667 (all trainable) |
| Checkpoint size | 74.5 KB |
| Inference latency | 0.041 ms/step (greedy), 0.073 ms/step (sampled) |
| Evaluation throughput | ~24,500 env steps/s (greedy) |
| Total steps evaluated | 150,000 per policy |

**Flow diagnostic**

At an initial state the learned flow is `log F(s₀) = 135.8`. Two analytic
reference points bracket what to expect: `(500 − t)·(log 2 + 1/T) = 596.6`
under a uniform policy, and `(500 − t)·(1/T) = 250` under a fully
deterministic one. The learned value sits below both. This is worth reading as
a diagnostic rather than a defect — the detailed balance residual is only
minimised on transitions that appear in the replay buffer, so the absolute
scale of `log F` is not pinned down away from that distribution. What the
objective actually constrains is the *difference* between connected states,
and that is what the policy reads out via the softmax.

### Comparison with the PPO baseline

| | GFlowNet (DB) | PPO |
|---|---|---|
| Average return (greedy) | 500.00 | 500.00 |
| Std | 0.00 | 0.00 |
| Worst episode | 500 | 500 |
| Success rate | 100% (CI 98.74–100) | 100% (CI 98.74–100) |
| **Return when sampling** | **498.55 (99.0% success)** | 500.00 (100% success) |
| Parameters | 17,667 | 9,155 |
| Inference latency | 0.041 ms/step | 0.090 ms/step |
| Evaluation episodes | 300 | 300 |

The two methods are indistinguishable at the ceiling: CartPole is saturated by
both. The interesting result is not that the GFlowNet wins — it is that a
flow-matching objective with **no reward maximisation anywhere in it** reaches
the same perfect control policy as a dedicated policy-gradient method.

---

## Usage

```bash
pip install gymnasium torch numpy

python train.py       # 3,000 episodes; periodic greedy eval; saves best checkpoint
python evaluate.py    # 300 deterministic episodes on held-out seeds
```

Both scripts use paths relative to the working directory, so run them from
inside this folder.

During training, progress is logged every 100 episodes (sampled reward,
success %, DB loss, buffer size) and a 20-episode greedy evaluation runs every
250 episodes.

---

## Project structure

```
cartpole-GFlowNet/
├── gflownet.py                     # GFlowNetPolicy: shared trunk, policy + flow heads
├── train.py                        # Detailed Balance training loop, replay buffer, target net
├── evaluate.py                     # 300-episode deterministic evaluation
├── models/
│   └── gflownet_cartpole.pt        # Best greedy-eval checkpoint
└── README.md
```

---

## Reproducibility

| Component | Version |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.10.0 |
| Gymnasium | 1.3.0 |
| Hardware | Apple M1 Pro (CPU) |

Training seeds `random`, `numpy` and `torch` with `SEED = 42`; environment
resets use `SEED + episode`. Evaluation uses a disjoint seed range, so the
reported numbers are genuinely held out.

---

## Related

- [`../cartpole-PPO`](../cartpole-PPO) — PPO baseline on the same environment.
- [`../pointmaze-GFlowNet`](../pointmaze-GFlowNet) — the same idea scaled to a
  goal-conditioned continuous-control maze, using subtrajectory balance and an
  edge-flow parameterisation.
