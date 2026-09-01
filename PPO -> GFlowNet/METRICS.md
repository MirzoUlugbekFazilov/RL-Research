# Study A — metrics

**PPO on corrupted -> GFlowNet fine-tuned on original**

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
| ppo_pretrain_corrupted (native, corrupted env) | 0.825 ± 0.030  [0.813, 0.836] | 184.7 ± 11.9  [180.3, 189.2] | 67.7 ± 7.6  [64.9, 70.6] | 0.893 ± 0.029  [0.882, 0.904] |
| ppo_pretrain_corrupted (ZERO-SHOT on original) | 0.000 ± 0.000  [0.000, 0.000] | 0.0 ± 0.0  [0.0, 0.0] | n/a | n/a |
| gfn_finetune_original | 0.701 ± 0.148  [0.646, 0.756] | 55.1 ± 17.2  [48.7, 61.5] | 78.2 ± 13.6  [73.2, 83.3] | 0.683 ± 0.042  [0.667, 0.699] |
| gfn_scratch_original | 0.726 ± 0.134  [0.676, 0.776] | 51.2 ± 18.1  [44.4, 57.9] | 85.4 ± 12.3  [80.9, 90.0] | 0.679 ± 0.046  [0.662, 0.696] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **gflownet** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the ppo policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.701 | 0.726 | -0.025 | -0.17 | 0.5015 | 1.0000 | 0.4492 |
| Mean return | 55.1 | 51.2 | 4.0 | 0.22 | 0.3894 | 1.0000 | 0.3898 |
| Steps to goal (successful eps) | 78.2 | 85.4 | -7.2 | -0.55 | 0.0353 | 0.1412 | 0.0191 |
| Path efficiency L*/L | 0.683 | 0.679 | 0.004 | 0.08 | 0.7541 | 1.0000 | 0.9474 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.702 | 0.728 | -0.025 | -0.17 | 0.5067 |
| Mean return | 55.5 | 51.6 | 4.0 | 0.22 | 0.4025 |
| Steps to goal (successful eps) | 78.5 | 84.8 | -6.4 | -0.49 | 0.0661 |
| Path efficiency L*/L | 0.683 | 0.681 | 0.002 | 0.05 | 0.8532 |

## 3. Did the corruption actually break transfer?

The pretrained ppo model scores **0.825** on the corrupted environment it was trained on, and **0.000** zero-shot on the original one (difference -0.825, Welch p = 0.0000).

The corruption is an isometry of the action box, so the corrupted MDP is
isomorphic to the original and *exactly as hard*. Any gap here is therefore a
genuine transfer failure, not a difficulty confound.

## 4. Jumpstart (success rate at step 0 of adaptation)

- Fine-tuned: 0.000 ± 0.000  [0.000, 0.000]
- Scratch:    0.217 ± 0.146  [0.162, 0.271]
- Jumpstart:  -0.217

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 50,400 | 48,720 |
| 0.25 | scratch | 30/30 | 16,800 | 15,680 |
| 0.50 | finetune | 30/30 | 67,200 | 71,120 |
| 0.50 | scratch | 30/30 | 42,000 | 42,000 |
| 0.80 | finetune | 27/30 | 151,200 | 166,756 |
| 0.80 | scratch | 28/30 | 151,200 | 160,200 |

## 6. Transfer verification

The cross-algorithm initialisation is an **exact parameter copy**, not a
distillation, so phase 2 provably begins at the pretrained policy and spends
no budget on the handover. This is checked numerically on 512 probe
observations drawn from the whole observation box, not from states either
policy happens to visit.

| Seed | Direction | Params | max abs logit diff | Greedy agreement | Verified |
|---|---|---|---|---|---|
| seed_1000 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_123 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2000 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2001 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2002 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2003 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2004 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2005 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2006 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2007 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2008 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2009 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2010 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2011 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2012 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2013 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2014 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2015 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2016 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2017 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2018 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2019 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2020 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2021 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2022 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2023 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_2024 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_42 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_456 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |
| seed_789 | ppo → gflownet | 6 | 0.00e+00 | 1.000 | yes |

What is *not* transferred, and cannot be: a GFlowNet has no value function and
PPO has no partition function, so exactly one head is randomly initialised in
each direction (PPO's critic in study B, the GFlowNet's `log Z` head in study A).
The two studies therefore give up the same amount, which is what keeps them
comparable.
