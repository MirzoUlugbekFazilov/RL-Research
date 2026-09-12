# Study B — metrics

**GFlowNet on corrupted -> PPO fine-tuned on original**

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
| gfn_pretrain_corrupted (native, corrupted env) | 0.178 ± 0.080  [0.148, 0.208] | 8.6 ± 6.1  [6.3, 10.9] | 154.5 ± 37.7  [140.4, 168.6] | 0.788 ± 0.085  [0.756, 0.819] |
| gfn_pretrain_corrupted (ZERO-SHOT on original) | 0.311 ± 0.142  [0.258, 0.364] | 16.9 ± 6.9  [14.3, 19.5] | 84.8 ± 21.4  [76.8, 92.8] | 0.666 ± 0.081  [0.636, 0.697] |
| ppo_finetune_original | 0.825 ± 0.046  [0.808, 0.842] | 178.4 ± 13.4  [173.4, 183.4] | 74.9 ± 9.0  [71.5, 78.2] | 0.879 ± 0.030  [0.868, 0.890] |
| ppo_scratch_original | 0.827 ± 0.029  [0.816, 0.838] | 185.7 ± 13.7  [180.6, 190.8] | 68.6 ± 6.2  [66.3, 71.0] | 0.899 ± 0.024  [0.890, 0.908] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **ppo** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the gflownet policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.825 | 0.827 | -0.002 | -0.04 | 0.8675 | 0.8675 | 0.8201 |
| Mean return | 178.4 | 185.7 | -7.3 | -0.53 | 0.0412 | 0.0825 | 0.0048 |
| Steps to goal (successful eps) | 74.9 | 68.6 | 6.3 | 0.80 | 0.0028 | 0.0111 | 0.0096 |
| Path efficiency L*/L | 0.879 | 0.899 | -0.020 | -0.72 | 0.0065 | 0.0194 | 0.0052 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.824 | 0.826 | -0.002 | -0.05 | 0.8399 |
| Mean return | 178.1 | 185.7 | -7.6 | -0.55 | 0.0399 |
| Steps to goal (successful eps) | 75.2 | 68.7 | 6.6 | 0.84 | 0.0022 |
| Path efficiency L*/L | 0.880 | 0.900 | -0.020 | -0.72 | 0.0077 |

## 3. Did the corruption actually break transfer?

The pretrained gflownet model scores **0.178** on the corrupted environment it was trained on, and **0.311** zero-shot on the original one (difference 0.133, Welch p = 0.0001).

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

- Fine-tuned: 0.281 ± 0.129  [0.233, 0.329]
- Scratch:    0.239 ± 0.186  [0.169, 0.308]
- Jumpstart:  0.042

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 0 | 11,000 |
| 0.25 | scratch | 30/30 | 15,000 | 10,500 |
| 0.50 | finetune | 30/30 | 45,000 | 54,000 |
| 0.50 | scratch | 30/30 | 15,000 | 31,500 |
| 0.80 | finetune | 29/30 | 150,000 | 155,172 |
| 0.80 | scratch | 30/30 | 187,500 | 168,000 |

## 6. Transfer verification

The cross-algorithm initialisation is an **exact parameter copy**, not a
distillation, so phase 2 provably begins at the pretrained policy and spends
no budget on the handover. This is checked numerically on 512 probe
observations drawn from the whole observation box, not from states either
policy happens to visit.

| Seed | Direction | Params | max abs logit diff | Greedy agreement | Verified |
|---|---|---|---|---|---|
| seed_1000 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_123 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2000 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2001 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2002 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2003 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2004 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2005 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2006 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2007 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2008 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2009 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2010 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2011 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2012 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2013 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2014 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2015 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2016 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2017 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2018 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2019 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2020 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2021 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2022 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2023 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_2024 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_42 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_456 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |
| seed_789 | gflownet → ppo | 6 | 0.00e+00 | 1.000 | yes |

What is *not* transferred, and cannot be: a GFlowNet has no value function and
PPO has no partition function, so exactly one head is randomly initialised in
each direction (PPO's critic in study B, the GFlowNet's `log Z` head in study A).
The two studies therefore give up the same amount, which is what keeps them
comparable.
