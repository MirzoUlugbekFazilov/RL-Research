# Study C — metrics

**PPO on corrupted -> PPO fine-tuned on original (actor copy only)**

- Environment: `PointMaze_UMaze-v3`, sparse reward, 300-step episodes
- Corruption: `large_multigoal` (a larger maze carrying several goals (`LARGE_MAZE_DIVERSE_G`, 12x9, 7 goal cells) replaces UMaze (5x5, 9 free cells))
- Budgets: 300,000 pretrain + 300,000 adapt (environment steps)
- Seeds: [42, 123, 456, 789, 1000, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] (n = 30 independent trained models per arm)
- Evaluation: 100 fixed instances (`seed = 1000000 + i`), frozen policy, deterministic
- `hyperparameters.py` SHA-256: `d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7`

## 1. Results by arm

Mean ± SD across seeds, with a t-based 95% CI on the mean.

| Arm | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| ppo_pretrain_corrupted (native, corrupted env) | 0.229 ± 0.094  [0.194, 0.264] | 16.8 ± 8.5  [13.7, 20.0] | 145.7 ± 33.0  [133.2, 158.3] | 0.851 ± 0.092  [0.816, 0.886] |
| ppo_pretrain_corrupted (ZERO-SHOT on original) | 0.274 ± 0.122  [0.229, 0.320] | 22.7 ± 11.8  [18.3, 27.1] | 92.4 ± 24.6  [83.2, 101.6] | 0.690 ± 0.085  [0.658, 0.721] |
| ppo_finetune_original | 0.837 ± 0.055  [0.817, 0.858] | 185.1 ± 18.5  [178.2, 192.0] | 71.2 ± 11.0  [67.1, 75.3] | 0.877 ± 0.036  [0.864, 0.891] |
| ppo_scratch_original | 0.827 ± 0.029  [0.816, 0.838] | 185.7 ± 13.7  [180.6, 190.8] | 68.6 ± 6.2  [66.3, 71.0] | 0.899 ± 0.024  [0.890, 0.908] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **ppo** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the ppo policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.837 | 0.827 | 0.010 | 0.23 | 0.3712 | 0.8270 | 0.3136 |
| Mean return | 185.1 | 185.7 | -0.6 | -0.04 | 0.8900 | 0.8900 | 0.9590 |
| Steps to goal (successful eps) | 71.2 | 68.6 | 2.5 | 0.28 | 0.2757 | 0.8270 | 0.6757 |
| Path efficiency L*/L | 0.877 | 0.899 | -0.022 | -0.71 | 0.0078 | 0.0312 | 0.0092 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.839 | 0.826 | 0.013 | 0.29 | 0.2663 |
| Mean return | 185.2 | 185.7 | -0.6 | -0.03 | 0.8993 |
| Steps to goal (successful eps) | 71.5 | 68.7 | 2.8 | 0.31 | 0.2380 |
| Path efficiency L*/L | 0.876 | 0.900 | -0.024 | -0.76 | 0.0049 |

## 3. Did the corruption actually break transfer?

The pretrained ppo model scores **0.229** on the corrupted environment it was trained on, and **0.274** zero-shot on the original one (difference 0.045, Welch p = 0.1125).

**This corruption is not difficulty-preserving, and the gap below must not be
read as a pure transfer effect.** The corrupted MDP is a different and harder
task, not a relabelling of the original: the maze, its size, the start
distribution (one reset cell vs. nine free cells) and the goal distribution
(seven designated cells vs. any free cell) all change together. The zero-shot
number therefore mixes *the policy is wrong here* with *the policy never saw a
task shaped like this*, and the native score is not comparable across the two
environments at all. The fine-tune-vs-scratch contrast in §2 is unaffected by
this -- both of those arms run the same algorithm on the same original
environment with the same budget, and differ only in initialisation.

## 4. Jumpstart (success rate at step 0 of adaptation)

- Fine-tuned: 0.276 ± 0.146  [0.221, 0.330]
- Scratch:    0.239 ± 0.186  [0.169, 0.308]
- Jumpstart:  0.037

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 0 | 21,500 |
| 0.25 | scratch | 30/30 | 15,000 | 10,500 |
| 0.50 | finetune | 30/30 | 75,000 | 79,500 |
| 0.50 | scratch | 30/30 | 15,000 | 31,500 |
| 0.80 | finetune | 28/30 | 142,500 | 161,786 |
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
