# Study A — metrics

**PPO on corrupted -> GFlowNet fine-tuned on original**

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
| gfn_finetune_original | 0.719 ± 0.144  [0.665, 0.773] | 62.3 ± 20.5  [54.6, 69.9] | 73.9 ± 12.4  [69.3, 78.5] | 0.701 ± 0.050  [0.682, 0.719] |
| gfn_scratch_original | 0.726 ± 0.134  [0.676, 0.776] | 51.2 ± 18.1  [44.4, 57.9] | 85.4 ± 12.3  [80.9, 90.0] | 0.679 ± 0.046  [0.662, 0.696] |

## 2. Primary contrast — fine-tuned vs. from scratch

Both arms are **gflownet** on the original environment with the
same 300,000-step budget and identical hyperparameters. They differ only in
initialisation: fine-tuned starts from the ppo policy trained on the
corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| Success rate | 0.719 | 0.726 | -0.007 | -0.05 | 0.8532 | 0.8532 | 0.7412 |
| Mean return | 62.3 | 51.2 | 11.1 | 0.57 | 0.0301 | 0.0902 | 0.0243 |
| Steps to goal (successful eps) | 73.9 | 85.4 | -11.5 | -0.92 | 0.0006 | 0.0025 | 0.0006 |
| Path efficiency L*/L | 0.701 | 0.679 | 0.021 | 0.44 | 0.0890 | 0.1780 | 0.0965 |

### Sensitivity — excluding seeds whose training envs touch the eval set

Environment seeds are `seed * 1000 + i`, so seed(s) [1000] start their 8 training envs on instances that also
appear in the 100-instance evaluation set. Those seeds are **retained** in the
headline numbers -- dropping a seed after seeing its result would be the worse
error -- and the contrast is repeated without them here (n = 29).

| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |
|---|---|---|---|---|---|
| Success rate | 0.712 | 0.728 | -0.015 | -0.11 | 0.6787 |
| Mean return | 61.6 | 51.6 | 10.0 | 0.51 | 0.0543 |
| Steps to goal (successful eps) | 73.4 | 84.8 | -11.4 | -0.93 | 0.0007 |
| Path efficiency L*/L | 0.702 | 0.681 | 0.021 | 0.43 | 0.1014 |

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
- Scratch:    0.217 ± 0.146  [0.162, 0.271]
- Jumpstart:  0.059

## 5. Sample efficiency — environment steps to reach a success rate

Measured on the learning curve (30 episodes per point, every 15,000 steps). `n reached` matters as much as the median: a
median over seeds that never crossed the threshold would be meaningless.

| Threshold | Arm | Seeds reaching | Median steps | Mean steps |
|---|---|---|---|---|
| 0.25 | finetune | 30/30 | 0 | 6,720 |
| 0.25 | scratch | 30/30 | 16,800 | 15,680 |
| 0.50 | finetune | 30/30 | 25,200 | 26,880 |
| 0.50 | scratch | 30/30 | 42,000 | 42,000 |
| 0.80 | finetune | 27/30 | 84,000 | 113,867 |
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
