# Study B — metrics

**GFlowNet on corrupted -> PPO fine-tuned on original**

- Environment: `PointMaze_UMaze-v3`, sparse reward, 300-step episodes
- Corruption: `negate_both` (both actuators reversed, `u -> -u`)
- Budgets: 300,000 pretrain + 300,000 adapt (environment steps)
- Seeds: [42, 123, 456, 789, 1000, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] (n = 30 independent trained models per arm)
- Evaluation: 100 fixed instances (`seed = 1000000 + i`), frozen policy, deterministic
- `hyperparameters.py` SHA-256: `d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7`

## 1. Results by arm

Mean ± SD across seeds, with a t-based 95% CI on the mean.

| Arm | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| gfn_pretrain_corrupted (native, corrupted env) | 0.666 ± 0.111  [0.624, 0.707] | 47.6 ± 18.1  [40.8, 54.3] | 79.3 ± 15.5  [73.5, 85.1] | 0.676 ± 0.048  [0.658, 0.694] |
| gfn_pretrain_corrupted (ZERO-SHOT on original) | 0.017 ± 0.023  [0.008, 0.025] | 0.8 ± 1.4  [0.3, 1.3] | 71.2 ± 64.7  [40.0, 102.4] | 0.546 ± 0.175  [0.462, 0.631] |
| ppo_finetune_original | 0.776 ± 0.064  [0.752, 0.800] | 145.6 ± 31.9  [133.7, 157.5] | 72.4 ± 10.0  [68.7, 76.1] | 0.823 ± 0.051  [0.804, 0.842] |
| ppo_scratch_original | 0.827 ± 0.029  [0.816, 0.838] | 185.7 ± 13.7  [180.6, 190.8] | 68.6 ± 6.2  [66.3, 71.0] | 0.899 ± 0.024  [0.890, 0.908] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **ppo** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the gflownet policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.776 | 0.827 | -0.051 | -1.00 | 0.0003 | 0.0006 | 0.0003 |
| Mean return | 145.6 | 185.7 | -40.1 | -1.61 | 0.0000 | 0.0000 | 0.0000 |
| Steps to goal (successful eps) | 72.4 | 68.6 | 3.8 | 0.45 | 0.0857 | 0.0857 | 0.0772 |
| Path efficiency L*/L | 0.823 | 0.899 | -0.076 | -1.87 | 0.0000 | 0.0000 | 0.0000 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.775 | 0.826 | -0.051 | -1.01 | 0.0004 |
| Mean return | 147.0 | 185.7 | -38.7 | -1.57 | 0.0000 |
| Steps to goal (successful eps) | 72.6 | 68.7 | 4.0 | 0.47 | 0.0769 |
| Path efficiency L*/L | 0.827 | 0.900 | -0.073 | -1.89 | 0.0000 |

## 3. Did the corruption actually break transfer?

The pretrained gflownet model scores **0.666** on the corrupted environment it was trained on, and **0.017** zero-shot on the original one (difference -0.649, Welch p = 0.0000).

The corruption is an isometry of the action box, so the corrupted MDP is
isomorphic to the original and *exactly as hard*. Any gap here is therefore a
genuine transfer failure, not a difficulty confound.

## 4. Jumpstart (success rate at step 0 of adaptation)

- Fine-tuned: 0.019 ± 0.030  [0.008, 0.030]
- Scratch:    0.239 ± 0.186  [0.169, 0.308]
- Jumpstart:  -0.220

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 75,000 | 80,500 |
| 0.25 | scratch | 30/30 | 15,000 | 10,500 |
| 0.50 | finetune | 30/30 | 142,500 | 137,500 |
| 0.50 | scratch | 30/30 | 15,000 | 31,500 |
| 0.80 | finetune | 22/30 | 232,500 | 219,545 |
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
