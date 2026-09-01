# RL Research — GFlowNets as Control Algorithms

Six self-contained reinforcement learning projects built around **Generative Flow
Networks** and **PPO**, in two parts.

**Part 1 — can a flow objective control at all?** Four projects put a GFlowNet
head-to-head with PPO on the same two tasks. GFlowNets are designed to *sample*
objects in proportion to a reward, not to maximise return; the question is
whether that objective can be used as a control algorithm, and how it compares
to a dedicated policy-gradient method on tasks PPO is known to solve.

**Part 2 — does what either one learns survive a broken environment?** Two
paired transfer studies train one algorithm on a *corrupted* PointMaze and
fine-tune the *other* algorithm on the original one, in both directions, to test
whether a policy learned under a fault is a useful starting point once the fault
is removed.

---

## Part 1 — Algorithm comparison

| Project | Task | Algorithm | Success rate |
|---|---|---|---|
| [`cartpole-PPO`](cartpole-PPO) | `CartPole-v1` | PPO (Stable-Baselines3) | **100%** — 500.00/500 avg return |
| [`cartpole-GFlowNet`](cartpole-GFlowNet) | `CartPole-v1` | GFlowNet, **detailed balance** | **100%** — 300/300 episodes |
| [`pointmaze-ppo`](pointmaze-ppo) | `PointMaze_UMazeDense-v3` | PPO (Stable-Baselines3) | **100%** — 300/300, 48.8 steps to goal |
| [`pointmaze-GFlowNet`](pointmaze-GFlowNet) | `PointMaze_UMaze-v3` (sparse) | GFlowNet, **subtrajectory balance** | **100%** — 300/300, 47.9 steps to goal |

All figures are measured by running each project's `evaluate.py` on the
committed checkpoint. Each project's README documents its algorithm,
hyperparameters, design decisions and full results.

### Metrics at a glance

All measured over **300 episodes** on fixed held-out seeds, greedy action
selection, single-threaded CPU. Success rates carry a 95% Wilson confidence
interval.

| | cartpole-PPO | cartpole-GFlowNet | pointmaze-ppo | pointmaze-GFlowNet |
|---|---|---|---|---|
| Success rate | 100% | 100% | 100% | 100% |
| 95% CI | 98.74–100% | 98.74–100% | 98.74–100% | 98.74–100% |
| Primary metric | return 500.00 | return 500.00 | 48.82 steps | 47.94 steps |
| Std | 0.00 | 0.00 | 27.68 | 25.18 |
| Median | 500 | 500 | 43 | 41.5 |
| Min / max | 500 / 500 | 500 / 500 | 7 / 121 | 9 / 122 |
| **Success when sampling** | 100% | **99.0%** | — | **100%** |
| Parameters | 9,155 | 17,667 | 136,965 | 141,316 |
| Checkpoint | 43.6 KB | 74.5 KB | 1.6 MB | 556 KB |
| Latency | 0.090 ms/step | 0.041 ms/step | 0.136 ms/step | 0.091 ms/step |

The **"success when sampling"** row is the one worth pausing on. It reports
performance when actions are drawn from the policy distribution instead of
taking the argmax — the mode a GFlowNet is actually designed for. Both
GFlowNets remain near-perfect as *samplers* (99.0% on CartPole, 100% on the
maze), which means the flow is sharp enough to control the system without any
greedy read-out on top.

### Findings

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

## Part 2 — Cross-algorithm transfer from a corrupted environment

| Study | Pretrain (corrupted) | Fine-tune (original) | Result |
|---|---|---|---|
| [`PPO -> GFlowNet`](PPO%20-%3E%20GFlowNet) | PPO | GFlowNet | **null** — *g* = −0.17, Holm *p* = 1.00 |
| [`GFlowNet -> PPO`](GFlowNet%20-%3E%20PPO) | GFlowNet | PPO | **significant harm** — *g* = −1.00, Holm *p* = 6.4 × 10⁻⁴ |

Each study trains one algorithm for 300k environment steps on a corrupted
`PointMaze_UMaze-v3`, copies its policy into the *other* algorithm, and
fine-tunes for a further 300k steps on the uncorrupted environment. Every
result is over **n = 30 seeds**, with a from-scratch control run under the
identical algorithm, budget and hyperparameters — without that control,
"the fine-tuned model reached *X*" says nothing.

The two directories contain **byte-identical code and a byte-identical
`hyperparameters.py`** (SHA-256 `d01ab117…`); only which algorithm occupies
which phase differs. `sanity_checks.py` asserts this, so "both studies used the
same hyperparameters" is checkable rather than claimed.

### The corruption

`negate_both` — both actuators wired backwards, `u ↦ −u`. The maze, observation
space, action space, reward and horizon are untouched. Because this is an
**isometry of the action box**, relabelling world coordinates recovers the
original dynamics: the corrupted MDP is *isomorphic* to the original and
therefore **exactly as hard**. Any transfer gap is a genuine transfer effect
rather than a difficulty confound — the failure mode that makes most "our
corruption hurt performance" results uninterpretable. This is asserted
numerically (identical geodesics, verified bijective isometry), not just argued.

### Results

| | PPO → GFlowNet | GFlowNet → PPO |
|---|---|---|
| Pretrained, on the corrupted env | 0.825 ± 0.030 | 0.666 ± 0.111 |
| Pretrained, **zero-shot** on the original | **0.000** | **0.017** |
| Fine-tuned (300k steps) | 0.701 ± 0.148 | 0.776 ± 0.064 |
| **From scratch (control)** | **0.726 ± 0.134** | **0.827 ± 0.029** |
| Hedges' *g* | −0.17 | **−1.00** |
| Holm-corrected *p* | 1.00 | **6.4 × 10⁻⁴** |
| Jumpstart | −0.217 | −0.220 |

**Transfer never helped in either direction.** It was undetectable for
PPO → GFlowNet and significantly *harmful* for GFlowNet → PPO, where three of
four metrics survive Holm correction — the largest being path efficiency
(0.823 vs 0.899, *g* = −1.87, *p* = 2.0 × 10⁻⁸). Fine-tuned PPO also needs
**9.5× more interaction** to reach 50% success (142,500 vs 15,000 steps), and
only 22 of 30 seeds ever reach 80% against 30 of 30 from scratch.

The mechanism is visible in the zero-shot row. A policy trained under reversed
actuators scores 0.000 on the original environment — far *below* the 0.26
uniform-random floor. It is not merely uninformed, it is confidently wrong, and
adaptation has to unlearn it first. Hence the large negative **jumpstart** in
both directions: the fine-tuned arm starts adaptation well below a randomly
initialised network.

The two studies differ in whether that cost is ever repaid. The GFlowNet's
curves converge with its control by ~150k steps, which is why its final
contrast is null — but a study reporting only final performance would call that
"no effect" and miss a real 3× sample-efficiency penalty over the first third of
training. PPO's curves never converge within the budget.

### How the transfer works

Cross-algorithm initialisation is an **exact parameter copy**, not distillation
— distillation would interpose a second optimisation with its own budget and
failure modes, and any result could then be blamed on it. To make the copy
meaningful the two policies are the same function class: both studies run every
phase on a `Discrete(9)` bang-bang action set, action repeat 5, a `t/T` time
feature, and a `[128, 128]` Tanh trunk, so PPO's actor *is* the GFlowNet's
forward policy and its logits *are* the tensor the GFlowNet softmaxes.

Across all 60 fine-tuning runs the copy was verified on 512 probe observations:
maximum absolute logit difference **0.0**, greedy-action agreement **1.000**.

**One head cannot transfer in either direction.** A GFlowNet has no value
function, so PPO's critic is randomly initialised; PPO has no partition function
(its critic estimates return-to-go, not a log partition function), so the
GFlowNet's `log Z` head is randomly initialised. Each study gives up exactly one
head, which keeps them comparable — and since this can only *disadvantage* the
fine-tuned arm, it cannot explain the null, though it is a candidate partial
explanation for the harm.

### A methodological note

Both studies were first run at **n = 5**, the seed count common in the deep-RL
transfer literature, and gave a **materially wrong** answer: PPO → GFlowNet
appeared to show a large *positive* effect (+0.118, *g* = +1.13). At n = 30 that
reverses sign and vanishes (−0.025, *g* = −0.17). With an across-seed SD of
≈ 0.14, the power to detect *g* = 1.1 at the Holm-corrected α is only **0.15**
at n = 5 — a textbook winner's curse. The GFlowNet → PPO effect, by contrast,
held its size (*g* = −1.06 → −1.00) and simply sharpened as power rose.

---

## Repository layout

```
.
├── cartpole-PPO/           # PPO baseline on CartPole-v1
├── cartpole-GFlowNet/      # Detailed-balance GFlowNet on CartPole-v1
├── pointmaze-ppo/          # PPO on the dense-reward U-maze
├── pointmaze-GFlowNet/     # Subtrajectory-balance GFlowNet on the sparse U-maze
├── PPO -> GFlowNet/        # Transfer study A: PPO (corrupted) -> GFlowNet (original)
└── GFlowNet -> PPO/        # Transfer study B: GFlowNet (corrupted) -> PPO (original)
```

The four Part 1 projects each follow the same structure: `train.py`,
`evaluate.py`, a `models/` directory holding the committed checkpoint, and a
`README.md`.

The two Part 2 studies share a different, common structure:

```
hyperparameters.py     single source of truth; byte-identical across both studies
run_study.py           runs all three arms x 30 seeds
analyze.py             regenerates METRICS.md and metrics_summary.json
sanity_checks.py       six validity assertions (must be 6/6 before trusting a result)
environments/          PointMaze, the corruption family, wrappers, geodesics
algorithms/            PPO, GFlowNet, and the two-way weight transfer
results/<arm>/seed_<n>/    per-episode records, learning curves, transfer proofs
checkpoints/               the corrupted-environment pretrained models
METRICS.md                 the full report
```

---

## Getting started

```bash
pip install torch numpy scipy gymnasium stable-baselines3 gymnasium-robotics mujoco
```

**Part 1** — run from inside a project directory; model paths are relative to
the working directory.

```bash
cd cartpole-GFlowNet     # or any other Part 1 project
python evaluate.py       # reproduce the reported numbers from the checkpoint
python train.py          # retrain from scratch
```

**Part 2** — the folder names contain spaces, so quote them.

```bash
cd "PPO -> GFlowNet"
python3 sanity_checks.py           # 6/6 must pass
python3 analyze.py                 # regenerate METRICS.md from committed results
python3 run_study.py --study A     # retrain all 3 arms x 30 seeds (~13 min)

cd "../GFlowNet -> PPO"
python3 run_study.py --study B     # the mirrored study
```

`run_study.py` caches completed runs, so re-running it is cheap; pass `--force`
to recompute. Only the pretrained (corrupted-environment) checkpoints are
committed — the fine-tuned and from-scratch models are regenerable and were left
out to keep the repository small.

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
| SciPy | 1.18.0 |
| Hardware | Apple M1 Pro (CPU) |

---

## A note on comparability

**Within Part 1**, the two PointMaze projects do **not** use identical
environments — they differ in reward signal (dense vs sparse), action space
(2-D continuous vs 4 discrete) and success radius (0.5 vs 0.45). Step-to-goal
counts are therefore not a like-for-like benchmark, and the per-project READMEs
flag this where the numbers appear side by side.

**Between Part 1 and Part 2**, the success rates are *not* comparable at all,
and the difference is a change of task rather than a regression. Part 1's
PointMaze projects terminate the episode at the goal (`continuing_task=False`)
and evaluate over a 150-step horizon with 4 discrete actions; Part 2 keeps the
registered default (`continuing_task=True`), a 300-step horizon, 9 bang-bang
actions held for 5 steps each, and a `t/T` input — a restricted shared interface
adopted so that weights can be copied between the two algorithms at all. Part 2
also scores every model on the same 100 held-out `(start, goal)` instances after
a fixed 300k-step budget, rather than training to convergence. A Part 2
from-scratch PPO reaching 0.827 and a Part 1 PPO reaching 100% are measurements
of different things.

**Within Part 2**, the comparison that matters is fine-tuned vs. from-scratch
*inside* a single study, which holds the algorithm, budget, hyperparameters and
evaluation set fixed and varies only the initialisation. The *cross-study*
comparison of one direction against the other is weaker: a from-scratch
GFlowNet reaches 0.726 where a from-scratch PPO reaches 0.827, so the two
fine-tuned arms have different amounts of headroom to recover into. Mean returns
are likewise not comparable across algorithms, because an action-repeated
GFlowNet dwells in the goal region differently under the continuing task.
