# Study C — metrics

**PPO on corrupted -> PPO fine-tuned on original (actor copy only)**

- Environment: `PointMaze_UMaze-v3`, sparse reward, 300-step episodes
- Corruption: `action_gain_jitter` (per-episode actuator gain `u -> g*u`, `g ~ U(0.9, 1.1)`, drawn at reset)
- Budgets: 300,000 pretrain + 300,000 adapt (environment steps)
- Seeds: [42, 123, 456, 789, 1000, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] (n = 30 independent trained models per arm)
- Evaluation: 100 fixed instances (`seed = 1000000 + i`), frozen policy, deterministic
- `hyperparameters.py` SHA-256: `d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7`

## 1. Results by arm

Mean ± SD across seeds, with a t-based 95% CI on the mean.

| Arm | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| ppo_pretrain_corrupted (native, corrupted env) | 0.818 ± 0.032  [0.806, 0.830] | 183.5 ± 12.1  [178.9, 188.0] | 67.8 ± 6.7  [65.3, 70.3] | 0.898 ± 0.024  [0.889, 0.906] |
| ppo_pretrain_corrupted (ZERO-SHOT on original) | 0.821 ± 0.030  [0.810, 0.833] | 184.1 ± 11.7  [179.8, 188.5] | 68.2 ± 6.0  [66.0, 70.5] | 0.897 ± 0.024  [0.888, 0.906] |
| ppo_finetune_original | 0.820 ± 0.043  [0.804, 0.836] | 199.7 ± 8.5  [196.6, 202.9] | 57.1 ± 4.9  [55.3, 58.9] | 0.928 ± 0.012  [0.924, 0.933] |
| ppo_scratch_original | 0.827 ± 0.029  [0.816, 0.838] | 185.7 ± 13.7  [180.6, 190.8] | 68.6 ± 6.2  [66.3, 71.0] | 0.899 ± 0.024  [0.890, 0.908] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **ppo** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the ppo policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.820 | 0.827 | -0.007 | -0.19 | 0.4632 | 0.4632 | 0.2794 |
| Mean return | 199.7 | 185.7 | 14.0 | 1.22 | 0.0000 | 0.0000 | 0.0000 |
| Steps to goal (successful eps) | 57.1 | 68.6 | -11.5 | -2.03 | 0.0000 | 0.0000 | 0.0000 |
| Path efficiency L*/L | 0.928 | 0.899 | 0.029 | 1.49 | 0.0000 | 0.0000 | 0.0000 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.820 | 0.826 | -0.006 | -0.16 | 0.5509 |
| Mean return | 199.8 | 185.7 | 14.1 | 1.21 | 0.0000 |
| Steps to goal (successful eps) | 57.0 | 68.7 | -11.6 | -2.01 | 0.0000 |
| Path efficiency L*/L | 0.928 | 0.900 | 0.028 | 1.43 | 0.0000 |

## 3. Did the corruption actually break transfer?

The pretrained ppo model scores **0.818** on the corrupted environment it was trained on, and **0.821** zero-shot on the original one (difference 0.003, Welch p = 0.6778).

The support of `g` **contains the original** (`g = 1` is interior to the
sampling range), so this is domain randomisation rather than a conflicting
task: the corrupted environment is a distribution over MDPs of which the
original is a member. Difficulty is near-preserved -- the plant is the same
point mass with a +-10% force scaling and the gain is not observable, so the
only added difficulty is the unresolvable uncertainty in `g`. This is the one
corruption in this codebase for which *positive* transfer is the a-priori
expectation.

## 4. Jumpstart (success rate at step 0 of adaptation)

- Fine-tuned: 0.858 ± 0.037  [0.844, 0.872]
- Scratch:    0.239 ± 0.186  [0.169, 0.308]
- Jumpstart:  0.619

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 0 | 0 |
| 0.25 | scratch | 30/30 | 15,000 | 10,500 |
| 0.50 | finetune | 30/30 | 0 | 0 |
| 0.50 | scratch | 30/30 | 15,000 | 31,500 |
| 0.80 | finetune | 30/30 | 0 | 500 |
| 0.80 | scratch | 30/30 | 187,500 | 168,000 |

## 6. Transfer verification

The same-algorithm initialisation is an **exact parameter copy**, not a
distillation, so phase 2 provably begins at the pretrained policy and spends
no budget on the handover. This is checked numerically on 512 probe
observations drawn from the whole observation box, not from states either
policy happens to visit.

| Seed | Direction | Params | max abs logit diff | Greedy agreement | Verified |
|---|---|---|---|---|---|
| seed_1000 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_123 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2000 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2001 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2002 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2003 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2004 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2005 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2006 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2007 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2008 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2009 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2010 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2011 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2012 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2013 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2014 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2015 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2016 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2017 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2018 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2019 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2020 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2021 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2022 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2023 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2024 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_42 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_456 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_789 | ppo → ppo | 6 | 0.00e+00 | 1.000 | yes |

What is *not* transferred, and deliberately so: the critic and the optimiser
moments. PPO could hand both over — see the `_full` report, which does — but a
GFlowNet cannot, so this arm gives up exactly what study B's arm was forced to,
and the two become a one-variable contrast in *which algorithm pretrained*.
The critic is randomly initialised in this arm and in the scratch control alike,
so it is never a difference between them.
