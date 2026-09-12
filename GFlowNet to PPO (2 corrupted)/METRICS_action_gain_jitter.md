# Study B — metrics

**GFlowNet on corrupted -> PPO fine-tuned on original**

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
| gfn_pretrain_corrupted (native, corrupted env) | 0.708 ± 0.182  [0.640, 0.776] | 48.5 ± 20.6  [40.8, 56.2] | 83.9 ± 15.5  [78.1, 89.6] | 0.673 ± 0.049  [0.655, 0.692] |
| gfn_pretrain_corrupted (ZERO-SHOT on original) | 0.703 ± 0.183  [0.635, 0.772] | 48.6 ± 20.7  [40.9, 56.4] | 83.5 ± 15.2  [77.8, 89.1] | 0.675 ± 0.048  [0.657, 0.693] |
| ppo_finetune_original | 0.838 ± 0.035  [0.825, 0.851] | 193.9 ± 8.7  [190.7, 197.1] | 66.0 ± 5.0  [64.1, 67.9] | 0.897 ± 0.030  [0.886, 0.908] |
| ppo_scratch_original | 0.827 ± 0.029  [0.816, 0.838] | 185.7 ± 13.7  [180.6, 190.8] | 68.6 ± 6.2  [66.3, 71.0] | 0.899 ± 0.024  [0.890, 0.908] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **ppo** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the gflownet policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.838 | 0.827 | 0.011 | 0.35 | 0.1772 | 0.3544 | 0.4232 |
| Mean return | 193.9 | 185.7 | 8.2 | 0.71 | 0.0079 | 0.0317 | 0.0063 |
| Steps to goal (successful eps) | 66.0 | 68.6 | -2.6 | -0.46 | 0.0777 | 0.2331 | 0.0906 |
| Path efficiency L*/L | 0.897 | 0.899 | -0.002 | -0.08 | 0.7687 | 0.7687 | 0.8660 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.836 | 0.826 | 0.010 | 0.31 | 0.2310 |
| Mean return | 193.6 | 185.7 | 7.9 | 0.67 | 0.0127 |
| Steps to goal (successful eps) | 65.6 | 68.7 | -3.0 | -0.53 | 0.0456 |
| Path efficiency L*/L | 0.897 | 0.900 | -0.003 | -0.09 | 0.7312 |

## 3. Did the corruption actually break transfer?

The pretrained gflownet model scores **0.708** on the corrupted environment it was trained on, and **0.703** zero-shot on the original one (difference -0.005, Welch p = 0.9214).

The support of `g` **contains the original** (`g = 1` is interior to the
sampling range), so this is domain randomisation rather than a conflicting
task: the corrupted environment is a distribution over MDPs of which the
original is a member. Difficulty is near-preserved -- the plant is the same
point mass with a +-10% force scaling and the gain is not observable, so the
only added difficulty is the unresolvable uncertainty in `g`. This is the one
corruption in this codebase for which *positive* transfer is the a-priori
expectation.

## 4. Jumpstart (success rate at step 0 of adaptation)

- Fine-tuned: 0.706 ± 0.195  [0.633, 0.778]
- Scratch:    0.239 ± 0.186  [0.169, 0.308]
- Jumpstart:  0.467

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 0 | 0 |
| 0.25 | scratch | 30/30 | 15,000 | 10,500 |
| 0.50 | finetune | 30/30 | 0 | 5,000 |
| 0.50 | scratch | 30/30 | 15,000 | 31,500 |
| 0.80 | finetune | 30/30 | 30,000 | 59,500 |
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
