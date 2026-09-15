# Provenance of Study C's arms

Study C (`PPO + PPO`) contains four arms per seed. Two of them already
existed, run by byte-identical code on the same 30 seeds, and were
imported rather than retrained. This file records exactly which.

| Arm | Origin | Role there | Seeds |
|---|---|---|---|
| `large_multigoal` / `ppo_pretrain_corrupted` | `ppo to gflownet` (Study A) | phase 1 of Study A (PPO on corrupted) | 30 |
| `large_multigoal` / `ppo_scratch_original` | `gflownet to ppo` (Study B) | from-scratch control of Study B (PPO on original) | 30 |
| `action_gain_jitter` / `ppo_pretrain_corrupted` | `ppo to gflownet` (Study A) | phase 1 of Study A (PPO on corrupted) | 30 |
| `action_gain_jitter` / `ppo_scratch_original` | `gflownet to ppo` (Study B) | from-scratch control of Study B (PPO on original) | 30 |

`ppo_finetune_original` and `ppo_finetune_full_original` are **not**
imported: they are this study's own runs and exist nowhere else.

## Why importing is the better choice, not the lazy one

- Study A fine-tunes a **GFlowNet** from `ppo_pretrain_corrupted`;
  Study C fine-tunes **PPO** from the very same checkpoints. The
  question "which algorithm fine-tunes better?" is then answered by a
  paired contrast with a common phase 1, seed by seed.
- Study B's fine-tuned arm and Study C's are measured against the
  **same** `ppo_scratch_original` models, so GFlowNet pretraining, PPO
  pretraining and no pretraining are compared to one baseline.

## What was verified before accepting each seed

- `hyperparameters.py` SHA-256 equals `d01ab1172c5f215c...`
- 12 shared modules byte-identical across all three directories
- per seed: arm, algorithm, training variant, 300,000-step budget,
  300,000 steps actually consumed, recorded hyperparameter hash,
  corruption (for the pretrain arm), and a checkpoint on disk

Run `python3 import_shared_arms.py --check` to re-verify, and
`python3 verify_import.py` to retrain sample seeds from scratch and
confirm they reproduce the imported models bit for bit.
