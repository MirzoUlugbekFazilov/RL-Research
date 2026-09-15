# Study C — PPO on corrupted → PPO fine-tuned on original

Same-algorithm transfer learning in PointMaze, comparing actor-only and full-model PPO fine-tuning after pretraining on corrupted environments.

## What this answers, and what the earlier studies could not

Study A (`../PPO to GFlowNet (2 corrupted)`) and Study B (`../GFlowNet to PPO (2 corrupted)`) each asked one
question: *does pretraining on a corrupted environment beat training from
scratch?* Both answered **no at convergence** — pretraining was neutral to
harmful on final success rate in all six corruption × direction pairs.

Neither could ask the obvious follow-up, because neither ever ran the matched
same-algorithm pretraining. "GFlowNet pretraining did not help PPO" leaves open
whether *any* pretraining would have, or whether the GFlowNet specifically was a
poor teacher. Study C settles it: PPO pretrains on the corrupted environment and
PPO fine-tunes on the original, against the identical from-scratch control.

Put beside Study B, the contrast varies exactly one thing — **which algorithm
did the pretraining** — and nothing else.

## Arms

| Arm | Algorithm | Environment | Initialisation |
|---|---|---|---|
| `ppo_pretrain_corrupted` | PPO | corrupted | random |
| `ppo_finetune_original` | PPO | original | actor copied from the pretrain arm; critic random, Adam fresh |
| `ppo_finetune_full_original` | PPO | original | whole model: actor, critic and Adam moments |
| `ppo_scratch_original` | PPO | original | random (control) |

300,000 environment steps per phase, n = 30 seeds, two corruptions
(`large_multigoal`, `action_gain_jitter`), evaluated frozen and deterministically
on the same 100 held-out instances every other arm in every other study sees.

### Why two fine-tuning surfaces

A GFlowNet has no value function, so Study B's transfer could only ever hand PPO
a policy. If Study C handed over a *whole* PPO model, "PPO pretraining beats
GFlowNet pretraining" would be confounded with "a learned critic beats a random
one", and there would be no way to tell which claim the numbers supported.

So both are run over the same pretrained checkpoints:

- **`ppo_finetune_original` (primary).** Actor only — precisely Study B's
  transfer surface. This is the arm the cross-study comparison uses.
- **`ppo_finetune_full_original` (secondary).** Ordinary PPO fine-tuning, what a
  practitioner would actually do. Its gap over the primary arm *prices* the
  critic and the optimiser state, rather than leaving it as an argument.

## Two arms are imported, not retrained

`ppo_pretrain_corrupted` is Study A's phase 1 and `ppo_scratch_original` is
Study B's control. Both already existed, for the same 30 seeds and the same two
corruptions, produced by byte-identical code. They are imported.

This is the better choice, not the cheaper one:

- Study A fine-tunes a **GFlowNet** from those checkpoints and Study C
  fine-tunes **PPO** from the very same ones, so "which algorithm fine-tunes
  better?" is a paired contrast over a common phase 1.
- Studies B and C are measured against the **same** from-scratch models, so
  GFlowNet pretraining, PPO pretraining and no pretraining meet one baseline.

`verify_import.py` retrains sample seeds and compares parameter hashes; the
recorded result (`verify_import.json`) is that retraining reproduces the imported
models exactly, on both arms and every metric. See `PROVENANCE.md`.

## Setup

From the repository root, using Python 3.10 or newer:

```bash
cd "PPO -> PPO"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 sanity_checks.py
```

Dependencies are not pinned to the original experiment environment. The recorded
results describe the original runs; newer dependencies may affect reproducibility.

## Project structure

| Path | Contents |
|---|---|
| `algorithms/` | PPO, GFlowNet, and transfer implementations |
| `environments/` | PointMaze layouts, corruptions, and wrappers |
| `training/`, `evaluation/` | Training callbacks and evaluation metrics |
| `runs/*/results/` | Per-seed configurations, evaluations, and curves |
| `METRICS_*.md`, `COMPARISON.md` | Recorded research reports |
| `*.tex` | Paper tables and manuscript material |

## Analyze recorded results

```bash
./analyze_all.sh
```

Run from a full repository checkout: cross-study analysis uses the sibling
`PPO to GFlowNet (2 corrupted)` and `GFlowNet to PPO (2 corrupted)` folders.
This command regenerates reports in place.

## Reproduce training

The parent repository excludes trained checkpoints from Git. Runtime logs and
Python caches are also omitted; the original local project retains these files.
To import shared arms or fine-tune models, first restore or regenerate the
checkpoints in the two sibling studies. Reading recorded reports does not require
checkpoints. Verification below retrains sample seeds and can take time.


```bash
python3 import_shared_arms.py      # validate + import the two shared arms
python3 verify_import.py           # retrain samples, confirm they reproduce
./run_all.sh 6                     # 120 fine-tuning runs over 6 workers
./analyze_all.sh                   # METRICS_*, COMPARISON.md, overleaf_study_c.tex
```

## Outputs

| File | What it holds |
|---|---|
| `METRICS_<corruption>.md` / `.json` | Study C, per corruption, primary surface |
| `METRICS_<corruption>_full.md` / `.json` | the same for the full-model surface |
| `COMPARISON.md` / `comparison.json` | all four PPO arms side by side, plus GFlowNet-vs-PPO as fine-tuners |
| `overleaf_study_c.tex` | the two Study C tables in the paper's format |
| `PROVENANCE.md`, `verify_import.json` | what was imported, and the proof it reproduces |

`combine_metrics.py` is deliberately absent — it is Study B's report writer,
hard-wired to that study's arms and prose. `compare_studies.py` replaces it and
recomputes every number from the per-seed `final_eval.json` files, so no figure
here can inherit a stale aggregate.

## The limitation to state before anyone reads the numbers

A 300,000-step fine-tuning budget is long enough for PPO to solve the original
task from a random start. **A converged comparison therefore cannot show a
sample-efficiency benefit — it measures the arms after the benefit has been
spent.** What survives convergence is reported; the mechanism is in the curves.

The comparison a robotics setting actually poses — long pretraining, *short*
budget on the target — is below this study's resolution: curves are evaluated
every 15,000 steps, so a 10,000-step fine-tuning budget cannot be read off them
at all. Measuring it needs fine-tuning runs at 5k/10k/25k/50k with a matched
from-scratch control at each budget. That is a separate sweep, not a re-reading
of these curves.
