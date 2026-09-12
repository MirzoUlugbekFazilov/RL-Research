# GFlowNet → PPO transfer under two new corruptions — metrics

**GFlowNet on corrupted → PPO fine-tuned on original**, run unchanged against two corruptions.

- Environment: `PointMaze_UMaze-v3`, sparse reward, 300-step episodes
- Budgets: 300,000 pretrain + 300,000 adapt (environment steps)
- Seeds: n = 30 independent trained models per arm per corruption
- Evaluation: 100 fixed instances (`seed = 1000000 + i`), frozen policy, deterministic
- `hyperparameters.py` SHA-256: `d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7` (byte-identical to `../RL+PPO`)

`negate_both` is the original study's published result, included as a reference column. It was produced by byte-identical code with the byte-identical `hyperparameters.py` used here, and was **not** re-run.

Significance markers are on the **Holm-corrected** p over the four metrics: `*` < 0.05, `**` < 0.01, `***` < 0.001.

## 1. Headline — does pretraining on the corrupted environment help?

Fine-tuned − scratch on the original environment. Both arms are PPO, same budget, same
hyperparameters, differing only in initialisation. Positive favours fine-tuning.

| Corruption | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| `large_multigoal` | -0.002 (g=-0.04) | -7.3 (g=-0.53) | 6.3 (g=0.80) * | -0.020 (g=-0.72) * |
| `action_gain_jitter` | 0.011 (g=0.35) | 8.2 (g=0.71) * | -2.6 (g=-0.46) | -0.002 (g=-0.08) |
| `negate_both (reference)` | -0.051 (g=-1.00) *** | -40.1 (g=-1.61) *** | 3.8 (g=0.45) | -0.076 (g=-1.87) *** |

Lower *steps to goal* is better; the sign is not flipped, so a negative entry there favours fine-tuning.

## 2. How much did each corruption actually break the policy?

Zero-shot is the pretrained GFlowNet dropped into the **original** environment with no adaptation — it is both the damage measurement and the step-0 point of the fine-tuning curve. The uniform-random floor on this evaluation set is **0.26**.

| Corruption | Native (corrupted env) | Zero-shot (original) | Welch p | vs. random floor |
|---|---|---|---|---|
| `large_multigoal` | 0.178 | 0.311 | 0.0001 | above |
| `action_gain_jitter` | 0.708 | 0.703 | 0.9214 | above |
| `negate_both (reference)` | 0.666 | 0.017 | 0.0000 | **below** — worse than random |

This is the axis the three corruptions differ on most, and it drives everything below:

- `negate_both` produces a policy that is **actively wrong** (0.017, far below the random floor): a good controller under `u ↦ −u` is an anti-optimal controller under `u ↦ u`.
- `large_multigoal` produces a policy that is **mediocre but not wrong** (0.311, just above the floor). Note it scores *higher* zero-shot on UMaze than natively on the large maze — UMaze is simply the easier of the two tasks, which is the difficulty confound this corruption carries and `negate_both` does not.
- `action_gain_jitter` produces a policy that is **already correct** (0.703 vs 0.708 native, p = 0.92). A ±10% gain fault costs essentially nothing, which is the point: `g = 1` is inside the training distribution.

## 3. Jumpstart and sample efficiency

| Corruption | Jumpstart | Steps to 0.80 success (median, fine-tuned) | (scratch) | Speed-up |
|---|---|---|---|---|
| `large_multigoal` | +0.042 | 150,000 (29/30) | 187,500 (30/30) | 1.25x |
| `action_gain_jitter` | +0.467 | 30,000 (30/30) | 187,500 (30/30) | 6.25x |
| `negate_both (reference)` | -0.220 | 232,500 (22/30) | 187,500 (30/30) | 0.81x |

## 4. Learning curves — mean success rate over 30 seeds

The headline contrast is one point on these curves (the last row). The shape is the result.

| Steps | large_multigoal ft | large_multigoal scratch | gap | action_gain_jitter ft | action_gain_jitter scratch | gap |
|---|---|---|---|---|---|---|
| 0 | 0.281 | 0.239 | +0.042 | 0.706 | 0.239 | +0.467 |
| 15,000 | 0.382 | 0.488 | -0.106 | 0.697 | 0.488 | +0.209 |
| 30,000 | 0.420 | 0.541 | -0.121 | 0.711 | 0.541 | +0.170 |
| 45,000 | 0.476 | 0.546 | -0.070 | 0.684 | 0.546 | +0.139 |
| 60,000 | 0.504 | 0.540 | -0.036 | 0.691 | 0.540 | +0.151 |
| 75,000 | 0.586 | 0.576 | +0.010 | 0.701 | 0.576 | +0.126 |
| 90,000 | 0.622 | 0.606 | +0.017 | 0.721 | 0.606 | +0.116 |
| 105,000 | 0.688 | 0.631 | +0.057 | 0.766 | 0.631 | +0.134 |
| 120,000 | 0.703 | 0.627 | +0.077 | 0.788 | 0.627 | +0.161 |
| 135,000 | 0.727 | 0.664 | +0.062 | 0.814 | 0.664 | +0.150 |
| 150,000 | 0.749 | 0.689 | +0.060 | 0.813 | 0.689 | +0.124 |
| 165,000 | 0.788 | 0.728 | +0.060 | 0.837 | 0.728 | +0.109 |
| 180,000 | 0.790 | 0.729 | +0.061 | 0.859 | 0.729 | +0.130 |
| 195,000 | 0.807 | 0.781 | +0.026 | 0.853 | 0.781 | +0.072 |
| 210,000 | 0.813 | 0.802 | +0.011 | 0.868 | 0.802 | +0.066 |
| 225,000 | 0.824 | 0.816 | +0.009 | 0.872 | 0.816 | +0.057 |
| 240,000 | 0.836 | 0.852 | -0.017 | 0.877 | 0.852 | +0.024 |
| 255,000 | 0.834 | 0.857 | -0.022 | 0.869 | 0.857 | +0.012 |
| 270,000 | 0.860 | 0.864 | -0.004 | 0.874 | 0.864 | +0.010 |
| 285,000 | 0.857 | 0.869 | -0.012 | 0.873 | 0.869 | +0.004 |
| 300,000 | 0.858 | 0.868 | -0.009 | 0.877 | 0.868 | +0.009 |

## 5. Verification

### The from-scratch control is shared, and provably so

A corruption may only ever affect `variant='corrupted'`. The from-scratch arm — PPO, randomly initialised, on the **original** environment — therefore has to come out identical in all three studies for a given seed. It does:

- **30/30 seeds bit-identical** across `large_multigoal`, `action_gain_jitter` and `../RL+PPO`'s `negate_both`, on all four metrics (max absolute discrepancy **0**).

This is a strong end-to-end check: it says the new corruption code, the new wrappers and the geodesic change did not perturb the uncorrupted environment by so much as a floating-point bit, and it means the three corruptions are compared against a genuinely common control.

### The cross-algorithm weight copy

| Corruption | Seeds verified | max abs logit diff | Greedy agreement |
|---|---|---|---|
| `large_multigoal` | 30/30 | 0.00e+00 | 1.000 |
| `action_gain_jitter` | 30/30 | 0.00e+00 | 1.000 |

The initialisation is an exact parameter copy, not a distillation, checked on 512 probe observations drawn from the whole observation box rather than from states either policy happens to visit. PPO's critic is randomly initialised in both the fine-tuned and the scratch arm, so it is never a difference between them.

### Corruption-specific assertions

`verify_corruptions.py` — 10/10 passing. The load-bearing ones:

- the `Discrete(9)` / `Box(7,)` interface is identical in all four conditions, so the weight copy is shape- *and* semantics-valid;
- the original environment is step-for-step identical under all three corruption names;
- `large_multigoal` equals `gymnasium_robotics`' own `LARGE_MAZE_DIVERSE_G` (7 goal cells, 1 reset cell) — a named upstream constant, so the map cannot have been drawn by us to produce a result;
- goal/reset markers are free space in the geodesic, so *L\** is not inflated on the large maze;
- the gain is drawn per episode in `[0.9, 1.1]`, held fixed within the episode, reproducible from the evaluation seed, and does not compound over resets;
- **the gain is two-sided under bang-bang actions.** See §6.

## 6. Caveats that change how these numbers should be read

**`large_multigoal` is not difficulty-preserving.** `negate_both` is an isometry of the action box, so its corrupted MDP is provably isomorphic to the original and *exactly as hard* — which is what let the original study attribute its whole zero-shot gap to transfer. Nothing like that holds here: maze, size, start distribution and goal distribution all change together, so the §2 zero-shot number for this corruption mixes *the policy is wrong here* with *the policy never saw a task shaped like this*, and its native score is not comparable across the two environments. **The §1 contrast is unaffected** — both arms there run PPO on the same original environment with the same budget, and differ only in initialisation.

**The gain corruption is applied to `actuator_gear`, not to the action array.** This is a deliberate departure from a literal `u = g*u`. `PointEnv.step` clips actions to `[-1, 1]` and every action in the shared `Discrete(9)` bang-bang table already sits on that boundary, so an `ActionWrapper` returning `g * action` would make `g > 1` a **no-op**: the realised corruption would be `u ↦ min(g, 1)·u`, a one-sided weakening with mean gain ≈0.975. Measured displacement after 25 steps of `u = (1,1)`, relative to `g = 1`:

| implementation | `g = 0.9` | `g = 1.1` |
|---|---|---|
| `g * action` (naive) | 0.066 m | **0.000 m — dead** |
| `actuator_gear` (used here) | 0.066 m | 0.055 m |

A motor's gain sits downstream of the controller's saturation limit, so `actuator_gear` — where delivered force is `gear × ctrl`, evaluated after every clip — is the faithful place for it. Asserted, not assumed, by `check_action_gain_is_two_sided`.

**Power.** n = 30 gives ~0.93 power for |g| = 1.1 at the Holm worst-case α, but only ~0.3–0.5 for |g| ≈ 0.5. Several effects reported above sit in that range, so a non-significant entry in §1 should be read as *not resolved at this n*, not as an established null. Effect sizes are reported unconditionally for that reason.

**One evaluation set, one environment family.** Every number here is PointMaze with a sparse reward under one shared bang-bang interface, and the fine-tuned arm's advantage is measured at a single 300k budget. Nothing here generalises to continuous-action PPO, to dense rewards, or to longer budgets without being re-measured.

