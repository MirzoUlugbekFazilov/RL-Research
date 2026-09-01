# Study A — PPO on corrupted PointMaze → GFlowNet fine-tuned on the original

Train **only PPO** on a corrupted `PointMaze_UMaze-v3`, then continue from that
policy with **only GFlowNet** on the uncorrupted environment, and measure
whether the corrupted-environment pretraining helps.

Its mirror image lives in `../RL+PPO` (GFlowNet → PPO). The two directories run
**byte-identical code with byte-identical hyperparameters**; only the roles of
the two algorithms are swapped.

```bash
cd "$(dirname "$0")"
python3 run_study.py --study A     # ~6 min: trains all 3 arms x 5 seeds
python3 analyze.py                 # writes METRICS.md + metrics_summary.json
```

---

## What is being measured

| Arm | Algorithm | Environment | Budget | Initialisation |
|---|---|---|---|---|
| `ppo_pretrain_corrupted` | PPO | **corrupted** | 300k steps | random |
| `gfn_finetune_original` | GFlowNet | **original** | 300k steps | ← PPO's policy, exact weight copy |
| `gfn_scratch_original` | GFlowNet | **original** | 300k steps | random |

`gfn_scratch_original` is the control, and it is what makes the study mean
anything. "The fine-tuned GFlowNet reaches success rate *X*" is not a result on
its own — the question is whether starting from PPO's corrupted-environment
policy beats starting from noise, with the same algorithm, the same budget and
the same hyperparameters. Only the **fine-tune − scratch** contrast answers
that, and it is the primary outcome.

The pretrained PPO model is additionally evaluated **zero-shot** on the original
environment, which is the "damage the corruption did" number and the step-0
point of the fine-tuning curve.

---

## The corruption: `negate_both`

Both actuators are wired backwards, `u ↦ −u`. The maze, the observation space,
the action space, the sparse reward and the 300-step horizon are untouched.

This matters for interpretation: `u ↦ −u` is an **isometry of the action box**,
so relabelling the world coordinates turns the corrupted MDP back into the
original one. The corrupted MDP is therefore *isomorphic* to the original and
**exactly as hard**. Any transfer gap the study measures is a genuine transfer
effect, not a difficulty confound — which is the failure mode that makes most
"our corruption hurt performance" results uninterpretable.

It was selected by a rule fixed in advance over a measured family of eight
corruptions (`environments/corruptions.py`): among the difficulty-preserving
candidates, the one with the lowest zero-shot success on the original
environment, required to fall below the uniform-random floor of 0.26. It
measures 0.000 zero-shot against a 0.870 reference. Notably, every *layout*
corruption in the family failed to degrade transfer at all.

---

## Why both algorithms run on one shared interface

A cross-algorithm weight copy is only meaningful if the two policies are the
same function class. Every phase of both studies therefore runs on:

- `Discrete(9)` bang-bang actions, `{−1,0,+1}²` — the extreme points of the
  force box. PointMaze is a linear plant with a box control constraint, so by
  Pontryagin's maximum principle a time-optimal control lies on that boundary
  almost everywhere; this is the smallest discretisation that keeps one
  representable.
- `action_repeat = 5` → 60 decisions per 300-step episode. Trajectory Balance
  sums `log P_F` over every decision, so its gradient variance grows with the
  decision count; this shortens the trajectory without changing the horizon,
  the physics or the reward.
- A `t/T` time feature in the observation (`Box(7,)`). The GFlowNet's forward
  policy is `P_F(a | s, t/T)` — the step index is part of the network input,
  which makes the sampling DAG a tree and lets Trajectory Balance drop its `P_B`
  term. PPO must receive the identical input or the copied first layer would be
  fed a different quantity.
- A `[128, 128]` Tanh trunk into a `Linear(128, 9)` head, so PPO's
  `mlp_extractor.policy_net` **is** the GFlowNet's `pf` hidden stack and PPO's
  categorical logits **are** the tensor the GFlowNet softmaxes.

This is a real restriction relative to the unmodified benchmark, and it is
applied identically to both algorithms in both phases of both studies, so it can
never be a difference between a fine-tuning arm and its control.

Budgets are counted in **environment steps** throughout — never decisions, never
gradient updates — so the two algorithms are directly comparable regardless of
how many updates each performs internally.

---

## The transfer: exact copy, verified

`GFlowNetAgent.from_ppo` copies PPO's actor into the GFlowNet's forward policy
directly. Not behavioural cloning or distillation: those introduce a second
optimisation with its own budget and its own failure modes between the phases,
and any result could then be blamed on the distillation rather than the
transfer. A parameter copy has no free parameters and no budget, so phase 2
provably begins at PPO's own policy.

The copy is **verified numerically rather than trusted** — PPO's and the
GFlowNet's logits are compared on 512 probe observations drawn from the whole
observation box (not from states either policy happens to visit), and the max
absolute difference and greedy-action agreement are recorded in each run's
`transfer.json`. A misaligned layer would show up immediately. Measured: max
|Δlogit| = 0.0, agreement = 1.000.

**What is not transferred:** the `log Z` head is randomly initialised. PPO has
no partition function — its critic is a value function, the expected return-to-go
from the *current* state, whereas `log Z(c)` is the log partition function of
the whole trajectory distribution for the initial (start, goal) instance.
Copying one into the other would be a category error. Study B gives up exactly
one head too (PPO's critic, since a GFlowNet has none), which is what keeps the
two studies comparable. This asymmetry can only ever *disadvantage* the
cross-algorithm arm, so it is stated with the results rather than hidden.

---

## Statistics

The unit of analysis is a **trained model** (one seed), never an evaluation
episode. Each of the 5 seeds contributes one number per metric — its mean over
the shared 100-instance evaluation set — and inference runs across those 5.
Treating the 100 episodes as 500 independent samples would inflate *n* by two
orders of magnitude and manufacture significance.

Every model faces the identical 100 instances (`reset(seed=1_000_000 + i)`), so
evaluation-instance variance is common to all arms and cancels in the contrasts.

Reported per metric: Welch's *t* (no equal-variance assumption), the exact
Mann-Whitney *U* (no distributional assumption, which matters at *n* = 5),
Hedges' *g* (Cohen's *d* is biased upward at small *n*), and the common-language
effect size. The four metrics are Holm-Bonferroni corrected as a family. Effect
sizes are reported unconditionally — at *n* = 5 an absent *p*-value is far more
often low power than a real null.

---

## Layout

```
hyperparameters.py    single source of truth; byte-identical to ../RL+PPO's
run_study.py          runs the 3 arms x 5 seeds
analyze.py            aggregates -> METRICS.md, metrics_summary.json
environments/         PointMaze, the corruption family, wrappers, geodesics
algorithms/           PPO, GFlowNet, and the two-way transfer
evaluation/           the single evaluation protocol, shared by both algorithms
training/             learning-curve callbacks
results/<arm>/seed_<n>/
    config.json       resolved hyperparameters + package versions + fingerprints
    final_eval.json   100-episode frozen-policy metrics
    episodes.csv      raw per-episode records
    curve.csv         learning curve
    zero_shot.json    pretrain arm: performance on the original env, unadapted
    transfer.json     fine-tune arm: the verified weight copy
    train_log.csv     GFlowNet Trajectory-Balance loss trace
checkpoints/<arm>/seed_<n>/model.{zip,pt}
METRICS.md            the report
```

Reproducibility: every run writes its own resolved config, the SHA-256 of
`hyperparameters.py`, exact package versions, and a parameter fingerprint taken
at load and after training.

```bash
# "both studies used the same hyperparameters" is checkable, not a promise:
shasum -a 256 hyperparameters.py ../RL+PPO/hyperparameters.py
diff -r . ../RL+PPO --exclude=results --exclude=checkpoints \
        --exclude='*.md' --exclude='*.log' --exclude=__pycache__ \
        --exclude=metrics_summary.json
```
