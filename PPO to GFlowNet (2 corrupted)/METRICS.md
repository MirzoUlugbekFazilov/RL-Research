# PPO → GFLOWNET transfer under two new corruptions — metrics

**PPO on corrupted → GFlowNet fine-tuned on original**, run unchanged against two corruptions.

- Environment: `PointMaze_UMaze-v3`, sparse reward, 300-step episodes
- Budgets: 300,000 pretrain + 300,000 adapt (environment steps)
- Seeds: n = 30 independent trained models per arm per corruption
- Evaluation: 100 fixed instances (`seed = 1000000 + i`), frozen policy, deterministic
- `hyperparameters.py` SHA-256: `d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7` (byte-identical to `RL+gflowne`)

`negate_both` is the original study's published result, included as a reference column. It was produced by byte-identical code with the byte-identical `hyperparameters.py` used here, and was **not** re-run.

Significance markers are on the **Holm-corrected** p over the four metrics: `*` < 0.05, `**` < 0.01, `***` < 0.001.

## 1. Headline — does pretraining on the corrupted environment help?

Fine-tuned − scratch on the original environment. Both arms are **gflownet**, same budget, same
hyperparameters, differing only in initialisation. Positive favours fine-tuning.

| Corruption | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| `large_multigoal` | -0.007 (g=-0.05) | 11.1 (g=0.57) | -11.5 (g=-0.92) ** | 0.021 (g=0.44) |
| `action_gain_jitter` | -0.021 (g=-0.13) | 16.8 (g=0.73) * | -7.4 (g=-0.59) | 0.013 (g=0.29) |
| `negate_both (reference)` | -0.025 (g=-0.17) | 4.0 (g=0.22) | -7.2 (g=-0.55) | 0.004 (g=0.08) |

Lower *steps to goal* is better; the sign is not flipped, so a negative entry there favours fine-tuning.

## 2. How much did each corruption actually break the policy?

Zero-shot is the pretrained ppo dropped into the **original** environment with no adaptation — it is both the damage measurement and the step-0 point of the fine-tuning curve. The uniform-random floor on this evaluation set is **0.26**.

| Corruption | Native (corrupted env) | Zero-shot (original) | Welch p | vs. random floor |
|---|---|---|---|---|
| `large_multigoal` | 0.229 | 0.274 | 0.1125 | above |
| `action_gain_jitter` | 0.818 | 0.821 | 0.6778 | above |
| `negate_both (reference)` | 0.825 | 0.000 | 0.0000 | **below** — worse than random |

This is the axis the three corruptions differ on most, and it drives everything below. Each verdict below is derived from the table, not asserted:

- `large_multigoal` produces a policy that is **no better than random** (0.274, indistinguishable from the 0.26 random floor, p = 0.52): the policy is not wrong on UMaze, it was simply trained for a different task.
- `action_gain_jitter` produces a policy that is **essentially undamaged** (0.821 zero-shot vs 0.818 native, p = 0.68; clear of the floor): `g = 1` is inside the training distribution, so a ±10% gain fault costs essentially nothing.
- `negate_both (reference)` produces a policy that is **actively wrong** (0.000, significantly *below* the 0.26 random floor): a good controller under `u ↦ −u` is an anti-optimal controller under `u ↦ u`.

## 3. Jumpstart and sample efficiency

| Corruption | Jumpstart | Steps to 0.80 success (median, fine-tuned) | (scratch) | Speed-up |
|---|---|---|---|---|
| `large_multigoal` | +0.059 | 84,000 (27/30) | 151,200 (28/30) | 1.80x |
| `action_gain_jitter` | +0.641 | 0 (30/30) | 151,200 (28/30) | already there at step 0 |
| `negate_both (reference)` | -0.217 | 151,200 (27/30) | 151,200 (28/30) | 1.00x |

## 4. Learning curves — mean success rate over 30 seeds

The headline contrast is one point on these curves (the last row). The shape is the result.

| Steps | large_multigoal ft | large_multigoal scratch | gap | action_gain_jitter ft | action_gain_jitter scratch | gap |
|---|---|---|---|---|---|---|
| 0 | 0.276 | 0.217 | +0.059 | 0.858 | 0.217 | +0.641 |
| 16,800 | 0.500 | 0.364 | +0.136 | 0.738 | 0.364 | +0.373 |
| 33,600 | 0.566 | 0.448 | +0.118 | 0.722 | 0.448 | +0.274 |
| 50,400 | 0.557 | 0.492 | +0.064 | 0.741 | 0.492 | +0.249 |
| 67,200 | 0.589 | 0.506 | +0.083 | 0.694 | 0.506 | +0.189 |
| 84,000 | 0.593 | 0.596 | -0.002 | 0.689 | 0.596 | +0.093 |
| 100,800 | 0.589 | 0.539 | +0.050 | 0.666 | 0.539 | +0.127 |
| 117,600 | 0.611 | 0.590 | +0.021 | 0.733 | 0.590 | +0.143 |
| 134,400 | 0.669 | 0.629 | +0.040 | 0.616 | 0.629 | -0.013 |
| 151,200 | 0.619 | 0.609 | +0.010 | 0.654 | 0.609 | +0.046 |
| 168,000 | 0.631 | 0.586 | +0.046 | 0.716 | 0.586 | +0.130 |
| 184,800 | 0.666 | 0.638 | +0.028 | 0.686 | 0.638 | +0.048 |
| 201,600 | 0.684 | 0.649 | +0.036 | 0.674 | 0.649 | +0.026 |
| 218,400 | 0.642 | 0.663 | -0.021 | 0.671 | 0.663 | +0.008 |
| 235,200 | 0.656 | 0.679 | -0.023 | 0.723 | 0.679 | +0.044 |
| 252,000 | 0.664 | 0.728 | -0.063 | 0.737 | 0.728 | +0.009 |
| 268,800 | 0.694 | 0.696 | -0.001 | 0.738 | 0.696 | +0.042 |
| 285,600 | 0.700 | 0.686 | +0.014 | 0.712 | 0.686 | +0.027 |
| 300,000 | 0.733 | 0.734 | -0.001 | 0.713 | 0.734 | -0.021 |

## 5. Verification

### The from-scratch control is shared, and provably so

A corruption may only ever affect `variant='corrupted'`. The from-scratch arm — gflownet, randomly initialised, on the **original** environment — therefore has to come out identical in all three studies for a given seed. It does:

- **30/30 seeds bit-identical** across `large_multigoal`, `action_gain_jitter` and `RL+gflowne`'s `negate_both`, on all four metrics (max absolute discrepancy **0**).

This is a strong end-to-end check: it says the new corruption code, the new wrappers and the geodesic change did not perturb the uncorrupted environment by so much as a floating-point bit, and it means the three corruptions are compared against a genuinely common control.

### The cross-algorithm weight copy

| Corruption | Seeds verified | max abs logit diff | Greedy agreement |
|---|---|---|---|
| `large_multigoal` | 30/30 | 0.00e+00 | 1.000 |
| `action_gain_jitter` | 30/30 | 0.00e+00 | 1.000 |

The initialisation is an exact parameter copy, not a distillation, checked on 512 probe observations drawn from the whole observation box rather than from states either policy happens to visit. The GFlowNet's `log Z` head is randomly initialised in both the fine-tuned and the scratch arm, so it is never a difference between them.

### Corruption-specific assertions

`verify_corruptions.py` — 10/10 passing. The load-bearing ones:

- the `Discrete(9)` / `Box(7,)` interface is identical in all four conditions, so the weight copy is shape- *and* semantics-valid;
- the original environment is step-for-step identical under all three corruption names;
- `large_multigoal` equals `gymnasium_robotics`' own `LARGE_MAZE_DIVERSE_G` (7 goal cells, 1 reset cell) — a named upstream constant, so the map cannot have been drawn by us to produce a result;
- goal/reset markers are free space in the geodesic, so *L\** is not inflated on the large maze;
- the gain is drawn per episode in `[0.9, 1.1]`, held fixed within the episode, reproducible from the evaluation seed, and does not compound over resets;
- **the gain is two-sided under bang-bang actions.** See §6.

## 6. Caveats that change how these numbers should be read

**`large_multigoal` is not difficulty-preserving.** `negate_both` is an isometry of the action box, so its corrupted MDP is provably isomorphic to the original and *exactly as hard* — which is what let the original study attribute its whole zero-shot gap to transfer. Nothing like that holds here: maze, size, start distribution and goal distribution all change together, so the §2 zero-shot number for this corruption mixes *the policy is wrong here* with *the policy never saw a task shaped like this*, and its native score is not comparable across the two environments. **The §1 contrast is unaffected** — both arms there run gflownet on the same original environment with the same budget, and differ only in initialisation.

**The gain corruption is applied to `actuator_gear`, not to the action array.** This is a deliberate departure from a literal `u = g*u`. `PointEnv.step` clips actions to `[-1, 1]` and every action in the shared `Discrete(9)` bang-bang table already sits on that boundary, so an `ActionWrapper` returning `g * action` would make `g > 1` a **no-op**: the realised corruption would be `u ↦ min(g, 1)·u`, a one-sided weakening with mean gain ≈0.975. Measured displacement after 25 steps of `u = (1,1)`, relative to `g = 1`:

| implementation | `g = 0.9` | `g = 1.1` |
|---|---|---|
| `g * action` (naive) | 0.066 m | **0.000 m — dead** |
| `actuator_gear` (used here) | 0.066 m | 0.055 m |

A motor's gain sits downstream of the controller's saturation limit, so `actuator_gear` — where delivered force is `gear × ctrl`, evaluated after every clip — is the faithful place for it. Asserted, not assumed, by `check_action_gain_is_two_sided`.

**Power.** n = 30 gives ~0.93 power for |g| = 1.1 at the Holm worst-case α, but only ~0.3–0.5 for |g| ≈ 0.5. Several effects reported above sit in that range, so a non-significant entry in §1 should be read as *not resolved at this n*, not as an established null. Effect sizes are reported unconditionally for that reason.

**One evaluation set, one environment family.** Every number here is PointMaze with a sparse reward under one shared bang-bang interface, and the fine-tuned arm's advantage is measured at a single 300k budget. Nothing here generalises to continuous-action PPO, to dense rewards, or to longer budgets without being re-measured.

