# PPO+PPO vs GFlowNet+PPO vs from scratch

Every arm below is **PPO on the original environment**, 300,000 environment steps, n = 30 seeds, identical hyperparameters (`hyperparameters.py` SHA-256 `d01ab117...`). The arms differ in one thing: what the weights were at step 0.

| Arm | Initialisation | Source |
|---|---|---|
| `scratch` | PPO, random init | shared — the *same* 30 models as Study B (PROVENANCE.md) |
| `gfn_pretrain` | PPO, from GFlowNet-on-corrupted (actor only) | `gflownet to ppo` (Study B) |
| `ppo_pretrain` | PPO, from PPO-on-corrupted (actor only) | this study |
| `ppo_pretrain_full` | PPO, from PPO-on-corrupted (actor + critic + Adam) | this study |

`*` p < 0.05, `**` p < 0.01, `***` p < 0.001, Holm-corrected over the four metrics within each contrast. Lower *steps to goal* is better, so a negative entry there favours the first arm named.

> **Two different measurements appear below, and they do not agree numerically.** The *Final performance* and *Contrasts* tables are the study's headline evaluation: 100 fixed held-out instances, frozen policy. The *learning curves* are the cheap in-training probe: 30 episodes every 15,000 steps, kept for shape, not for level. So the scratch arm finishes its curve at 0.868 while its headline success rate is 0.827 — same models, different sample size, and the curve is the noisier of the two. Compare arms *within* a table, never across the two.

> **The `ppo_pretrain_full` contrast against `scratch` is not initialisation-only.** That arm carries a trained critic and Adam moments; the scratch control starts from a random critic like every other arm here. It is the one comparison in these three studies where the two sides differ by more than their starting weights, and it is included precisely to price that difference — read it against `ppo_pretrain`, not against `scratch`.

## large_multigoal

### Final performance, mean ± SD over 30 seeds

| Arm | Success | Return | Steps | Path eff. |
|---|---|---|---|---|
| `scratch` | 0.827 ± 0.029 | 185.7 ± 13.7 | 68.6 ± 6.2 | 0.899 ± 0.024 |
| `gfn_pretrain` | 0.825 ± 0.046 | 178.4 ± 13.4 | 74.9 ± 9.0 | 0.879 ± 0.030 |
| `ppo_pretrain` | 0.837 ± 0.055 | 185.1 ± 18.5 | 71.2 ± 11.0 | 0.877 ± 0.036 |
| `ppo_pretrain_full` | 0.778 ± 0.138 | 157.8 ± 44.1 | 78.8 ± 18.2 | 0.849 ± 0.062 |

### Contrasts (first arm − second arm)

| Contrast | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| `gfn_pretrain` − `scratch` | -0.002 (g=-0.04) | -7.3 (g=-0.53) | +6.3* (g=0.80) | -0.020* (g=-0.72) |
| `ppo_pretrain` − `scratch` | +0.010 (g=0.23) | -0.6 (g=-0.04) | +2.5 (g=0.28) | -0.022* (g=-0.71) |
| `ppo_pretrain` − `gfn_pretrain` | +0.012 (g=0.23) | +6.7 (g=0.41) | -3.7 (g=-0.37) | -0.002 (g=-0.07) |
| `ppo_pretrain_full` − `ppo_pretrain` | -0.060 (g=-0.56) | -27.3* (g=-0.80) | +7.6 (g=0.50) | -0.028 (g=-0.55) |
| `ppo_pretrain_full` − `scratch` | -0.049 (g=-0.49) | -27.9** (g=-0.84) | +10.1* (g=0.74) | -0.050*** (g=-1.05) |

- `gfn_pretrain` − `scratch`: GFlowNet pretraining vs none — Study B's headline, recomputed. Success rate -0.002, Welch p = 0.868 (Holm 0.868), paired p = 0.880, g = -0.04.
- `ppo_pretrain` − `scratch`: **PPO pretraining vs none — Study C's headline**. Success rate +0.010, Welch p = 0.371 (Holm 0.827), paired p = 0.421, g = 0.23.
- `ppo_pretrain` − `gfn_pretrain`: **PPO vs GFlowNet pretraining — same transfer surface, the one-variable contrast**. Success rate +0.012, Welch p = 0.365 (Holm 0.730), paired p = 0.351, g = 0.23.
- `ppo_pretrain_full` − `ppo_pretrain`: The critic and Adam moments — what a GFlowNet cannot hand over. Success rate -0.060, Welch p = 0.034 (Holm 0.102), paired p = 0.026, g = -0.56.
- `ppo_pretrain_full` − `scratch`: Ordinary PPO fine-tuning vs none. Success rate -0.049, Welch p = 0.065 (Holm 0.065), paired p = 0.068, g = -0.49.

### Sample efficiency — mean curve crossing each threshold

| Arm | step 0 (zero-shot) | to 0.50 | to 0.80 | speed-up to 0.80 |
|---|---|---|---|---|
| `scratch` | 0.239 | 30,000 | 210,000 | 1.00x |
| `gfn_pretrain` | 0.281 | 60,000 | 195,000 | 1.08x |
| `ppo_pretrain` | 0.276 | 75,000 | 210,000 | 1.00x |
| `ppo_pretrain_full` | 0.276 | 120,000 | 300,000 | 0.70x |

Computed on the **mean** 30-episode curve over 30 seeds, at the study's 15,000-step resolution, so the finest speed-up this design can resolve is one grid point. `METRICS_*.md` reports the same quantity differently — per-seed crossings, then the median over seeds — and the two will not match; a mean curve crosses a threshold earlier than the median seed does whenever the seeds are spread out. A budget-limited comparison needs a finer grid than either — see the closing note.

### Learning curves — mean success rate over 30 seeds

| Steps | `scratch` | `gfn_pretrain` | `ppo_pretrain` | `ppo_pretrain_full` |
|---|---|---|---|---|
| 0 | 0.239 | 0.281 | 0.276 | 0.276 |
| 15,000 | 0.488 | 0.382 | 0.288 | 0.271 |
| 30,000 | 0.541 | 0.420 | 0.341 | 0.282 |
| 45,000 | 0.546 | 0.476 | 0.406 | 0.322 |
| 60,000 | 0.540 | 0.504 | 0.414 | 0.356 |
| 75,000 | 0.576 | 0.586 | 0.503 | 0.406 |
| 90,000 | 0.606 | 0.622 | 0.513 | 0.420 |
| 105,000 | 0.631 | 0.688 | 0.594 | 0.489 |
| 120,000 | 0.627 | 0.703 | 0.631 | 0.504 |
| 135,000 | 0.664 | 0.727 | 0.696 | 0.563 |
| 150,000 | 0.689 | 0.749 | 0.722 | 0.586 |
| 165,000 | 0.728 | 0.788 | 0.763 | 0.611 |
| 180,000 | 0.729 | 0.790 | 0.771 | 0.621 |
| 195,000 | 0.781 | 0.807 | 0.798 | 0.680 |
| 210,000 | 0.802 | 0.813 | 0.803 | 0.707 |
| 225,000 | 0.816 | 0.824 | 0.820 | 0.706 |
| 240,000 | 0.852 | 0.836 | 0.834 | 0.746 |
| 255,000 | 0.857 | 0.834 | 0.843 | 0.756 |
| 270,000 | 0.864 | 0.860 | 0.852 | 0.777 |
| 285,000 | 0.869 | 0.857 | 0.852 | 0.787 |
| 300,000 | 0.868 | 0.858 | 0.875 | 0.811 |

### Which algorithm fine-tunes better, from the same PPO checkpoints?

Study A fine-tuned a GFlowNet from `ppo_pretrain_corrupted`; Study C fine-tunes PPO from the **same 30 checkpoints** (PROVENANCE.md), so phase 1 is held fixed and only the fine-tuning algorithm varies.

| Fine-tuner | Success | Return | Steps | Path eff. |
|---|---|---|---|---|
| GFlowNet fine-tuner | 0.719 ± 0.144 | 62.3 ± 20.5 | 73.9 ± 12.4 | 0.701 ± 0.050 |
| PPO fine-tuner (actor only) | 0.837 ± 0.055 | 185.1 ± 18.5 | 71.2 ± 11.0 | 0.877 ± 0.036 |
| PPO fine-tuner (full model) | 0.778 ± 0.138 | 157.8 ± 44.1 | 78.8 ± 18.2 | 0.849 ± 0.062 |

PPO − GFlowNet, same initial weights, same budget: Success rate +0.118*** (g=1.07), Mean return +122.9*** (g=6.22), Steps to goal -2.7 (g=-0.23), Path efficiency +0.176*** (g=3.98).

## action_gain_jitter

### Final performance, mean ± SD over 30 seeds

| Arm | Success | Return | Steps | Path eff. |
|---|---|---|---|---|
| `scratch` | 0.827 ± 0.029 | 185.7 ± 13.7 | 68.6 ± 6.2 | 0.899 ± 0.024 |
| `gfn_pretrain` | 0.838 ± 0.035 | 193.9 ± 8.7 | 66.0 ± 5.0 | 0.897 ± 0.030 |
| `ppo_pretrain` | 0.820 ± 0.043 | 199.7 ± 8.5 | 57.1 ± 4.9 | 0.928 ± 0.012 |
| `ppo_pretrain_full` | 0.866 ± 0.033 | 211.8 ± 6.3 | 56.3 ± 4.1 | 0.926 ± 0.010 |

### Contrasts (first arm − second arm)

| Contrast | Success rate | Mean return | Steps to goal | Path efficiency |
|---|---|---|---|---|
| `gfn_pretrain` − `scratch` | +0.011 (g=0.35) | +8.2* (g=0.71) | -2.6 (g=-0.46) | -0.002 (g=-0.08) |
| `ppo_pretrain` − `scratch` | -0.007 (g=-0.19) | +14.0*** (g=1.22) | -11.5*** (g=-2.03) | +0.029*** (g=1.49) |
| `ppo_pretrain` − `gfn_pretrain` | -0.018 (g=-0.46) | +5.8* (g=0.67) | -8.9*** (g=-1.78) | +0.031*** (g=1.33) |
| `ppo_pretrain_full` − `ppo_pretrain` | +0.046*** (g=1.19) | +12.1*** (g=1.59) | -0.8 (g=-0.17) | -0.002 (g=-0.19) |
| `ppo_pretrain_full` − `scratch` | +0.039*** (g=1.24) | +26.1*** (g=2.42) | -12.3*** (g=-2.31) | +0.027*** (g=1.41) |

- `gfn_pretrain` − `scratch`: GFlowNet pretraining vs none — Study B's headline, recomputed. Success rate +0.011, Welch p = 0.177 (Holm 0.354), paired p = 0.149, g = 0.35.
- `ppo_pretrain` − `scratch`: **PPO pretraining vs none — Study C's headline**. Success rate -0.007, Welch p = 0.463 (Holm 0.463), paired p = 0.470, g = -0.19.
- `ppo_pretrain` − `gfn_pretrain`: **PPO vs GFlowNet pretraining — same transfer surface, the one-variable contrast**. Success rate -0.018, Welch p = 0.074 (Holm 0.074), paired p = 0.033, g = -0.46.
- `ppo_pretrain_full` − `ppo_pretrain`: The critic and Adam moments — what a GFlowNet cannot hand over. Success rate +0.046, Welch p = 0.000 (Holm 0.000), paired p = 0.000, g = 1.19.
- `ppo_pretrain_full` − `scratch`: Ordinary PPO fine-tuning vs none. Success rate +0.039, Welch p = 0.000 (Holm 0.000), paired p = 0.000, g = 1.24.

### Sample efficiency — mean curve crossing each threshold

| Arm | step 0 (zero-shot) | to 0.50 | to 0.80 | speed-up to 0.80 |
|---|---|---|---|---|
| `scratch` | 0.239 | 30,000 | 210,000 | 1.00x |
| `gfn_pretrain` | 0.706 | 0 | 135,000 | 1.56x |
| `ppo_pretrain` | 0.858 | 0 | 0 | at step 0 |
| `ppo_pretrain_full` | 0.858 | 0 | 0 | at step 0 |

Computed on the **mean** 30-episode curve over 30 seeds, at the study's 15,000-step resolution, so the finest speed-up this design can resolve is one grid point. `METRICS_*.md` reports the same quantity differently — per-seed crossings, then the median over seeds — and the two will not match; a mean curve crosses a threshold earlier than the median seed does whenever the seeds are spread out. A budget-limited comparison needs a finer grid than either — see the closing note.

### Learning curves — mean success rate over 30 seeds

| Steps | `scratch` | `gfn_pretrain` | `ppo_pretrain` | `ppo_pretrain_full` |
|---|---|---|---|---|
| 0 | 0.239 | 0.706 | 0.858 | 0.858 |
| 15,000 | 0.488 | 0.697 | 0.867 | 0.861 |
| 30,000 | 0.541 | 0.711 | 0.862 | 0.870 |
| 45,000 | 0.546 | 0.684 | 0.861 | 0.871 |
| 60,000 | 0.540 | 0.691 | 0.860 | 0.870 |
| 75,000 | 0.576 | 0.701 | 0.866 | 0.876 |
| 90,000 | 0.606 | 0.721 | 0.862 | 0.872 |
| 105,000 | 0.631 | 0.766 | 0.863 | 0.871 |
| 120,000 | 0.627 | 0.788 | 0.866 | 0.873 |
| 135,000 | 0.664 | 0.814 | 0.861 | 0.873 |
| 150,000 | 0.689 | 0.813 | 0.859 | 0.876 |
| 165,000 | 0.728 | 0.837 | 0.859 | 0.869 |
| 180,000 | 0.729 | 0.859 | 0.857 | 0.872 |
| 195,000 | 0.781 | 0.853 | 0.858 | 0.878 |
| 210,000 | 0.802 | 0.868 | 0.858 | 0.873 |
| 225,000 | 0.816 | 0.872 | 0.860 | 0.871 |
| 240,000 | 0.852 | 0.877 | 0.864 | 0.880 |
| 255,000 | 0.857 | 0.869 | 0.861 | 0.879 |
| 270,000 | 0.864 | 0.874 | 0.862 | 0.879 |
| 285,000 | 0.869 | 0.873 | 0.860 | 0.880 |
| 300,000 | 0.868 | 0.877 | 0.856 | 0.881 |

### Which algorithm fine-tunes better, from the same PPO checkpoints?

Study A fine-tuned a GFlowNet from `ppo_pretrain_corrupted`; Study C fine-tunes PPO from the **same 30 checkpoints** (PROVENANCE.md), so phase 1 is held fixed and only the fine-tuning algorithm varies.

| Fine-tuner | Success | Return | Steps | Path eff. |
|---|---|---|---|---|
| GFlowNet fine-tuner | 0.704 ± 0.181 | 68.0 ± 26.7 | 78.1 ± 12.5 | 0.693 ± 0.044 |
| PPO fine-tuner (actor only) | 0.820 ± 0.043 | 199.7 ± 8.5 | 57.1 ± 4.9 | 0.928 ± 0.012 |
| PPO fine-tuner (full model) | 0.866 ± 0.033 | 211.8 ± 6.3 | 56.3 ± 4.1 | 0.926 ± 0.010 |

PPO − GFlowNet, same initial weights, same budget: Success rate +0.116** (g=0.87), Mean return +131.7*** (g=6.57), Steps to goal -21.0*** (g=-2.18), Path efficiency +0.235*** (g=7.23).

## Verification

### The initialisation is an exact copy, on every seed

| Corruption | Arm | Seeds equivalent | max abs logit diff | Greedy agreement |
|---|---|---|---|---|
| `large_multigoal` | `ppo_finetune_original` | 30/30 | 0.00e+00 | 1.000 |
| `large_multigoal` | `ppo_finetune_full_original` | 30/30 | 0.00e+00 | 1.000 |
| `action_gain_jitter` | `ppo_finetune_original` | 30/30 | 0.00e+00 | 1.000 |
| `action_gain_jitter` | `ppo_finetune_full_original` | 30/30 | 0.00e+00 | 1.000 |

Checked on 512 probe observations drawn from the whole observation box, not from states either policy happens to visit.

### The from-scratch control is literally shared

- `large_multigoal`: 30/30 seeds, max absolute discrepancy vs Study B's control **0**
- `action_gain_jitter`: 30/30 seeds, max absolute discrepancy vs Study B's control **0**

Zero by construction — the models were imported, not retrained — and `verify_import.py` retrains sample seeds to confirm this directory would have produced the same models anyway (`verify_import.json`).

## What Study C adds

1. **PPO is not a better teacher than a GFlowNet.** With the transfer surface held identical, `ppo_pretrain` − `gfn_pretrain` on success rate is +0.012 and −0.018 on the two corruptions, neither significant after correction. The earlier studies' null was not an artefact of the GFlowNet being a poor source: matched same-algorithm pretraining does not beat it.

2. **PPO is a far better student.** From the *same* 30 pretrained checkpoints, PPO fine-tuning ends ~0.12 success above GFlowNet fine-tuning on both corruptions (g = 1.07 and 0.87, p < 0.001), and the return and path-efficiency gaps are enormous. Study A's ceiling at ~0.72 was the fine-tuning algorithm, not the initialisation — this is the direct test of that claim, with phase 1 held fixed.

3. **The first arm in these studies to beat scratch at convergence.** Under `action_gain_jitter`, full-model PPO→PPO reaches 0.866 against 0.827 (g = 1.24, p < 0.001) and gets there 12 steps faster per episode. Actor-only PPO→PPO ties on success but still wins decisively on route quality (57.1 vs 68.6 steps, g = −2.03). Across Studies A and B not one of six corruption–direction pairs managed a significant gain on the primary metric.

4. **Structural mismatch makes the critic harmful.** The same full transfer that wins under `action_gain_jitter` *loses* under `large_multigoal` (−0.049 success, −27.9 return, p < 0.01 vs scratch): a value function fitted to a 12×9 maze with 7 goals misprices states in the 5×5 UMaze, and PPO spends its budget unlearning it. The critic is worth carrying only when the two tasks share a reward geometry.

## How to read this

At a 300,000-step fine-tuning budget the arms largely converge, because that budget is long enough for PPO to solve the original task from a random start. That is a fact about the budget as much as about transfer: **a converged comparison cannot show a sample-efficiency benefit, because it measures the arms after the benefit has been spent.** What survives convergence is reported above; the mechanism is visible only in the curves, where `ppo_pretrain` under `action_gain_jitter` enters at 0.858 and never has anywhere to climb.

The comparison this design cannot make is the one a robotics setting actually poses: a long pretraining phase and a *short* budget on the target. At 15,000-step curve resolution the earliest measurable crossing is 15,000 steps, so a 10,000-step fine-tuning budget is below this study's resolution entirely. Measuring it needs fine-tuning runs at 5k/10k/25k/50k with a matched from-scratch control at each budget — a separate sweep, not a re-reading of these curves.
