# PPO → GFlowNet transfer under two new corruptions

Train **only PPO** on a corrupted `PointMaze_UMaze-v3`, then continue from that
policy with **only a GFlowNet** on the uncorrupted environment, and measure
whether the corrupted-environment pretraining helps.

This is the design of `../RL+gflowne` (Study A), re-run unchanged against two
corruptions it never tested:

| Corruption | Mechanism | What changes |
|---|---|---|
| `large_multigoal` | structural | A 12×9 maze carrying **7 goal cells** replaces the 5×5 UMaze |
| `action_gain_jitter` | stochastic actuation | Actuator gain `u ↦ g·u`, **`g ~ U(0.9, 1.1)` drawn once per episode** |

It is the exact mirror of `../gflownet to ppo`, which runs the same two
corruptions in the other direction (GFlowNet → PPO). The two directories run
**byte-identical code with byte-identical hyperparameters**; only the roles of
the two algorithms are swapped, via `run_study.py --study A` vs `--study B`.

```bash
python3 verify_corruptions.py                     # the corruption-specific assertions
./run_all.sh 6                                    # 3 arms x 30 seeds x 2 corruptions
python3 analyze.py --corruption large_multigoal    # -> METRICS_large_multigoal.{md,json}
python3 analyze.py --corruption action_gain_jitter # -> METRICS_action_gain_jitter.{md,json}
python3 combine_metrics.py                        # -> METRICS.md
```

`METRICS.md` is the combined report and the thing to read first; the two
`METRICS_<corruption>.md` files are the full per-corruption output. All three
are generated, so no number in them can drift away from the run it came from.

---

## What is being measured

Per corruption, three arms × 30 seeds:

| Arm | Algorithm | Environment | Budget | Initialisation |
|---|---|---|---|---|
| `ppo_pretrain_corrupted` | PPO | **corrupted** | 300k steps | random |
| `gfn_finetune_original` | GFlowNet | **original** | 300k steps | ← PPO's actor, exact weight copy |
| `gfn_scratch_original` | GFlowNet | **original** | 300k steps | random |

`gfn_scratch_original` is the control, and it is what makes the study mean
anything. "The fine-tuned GFlowNet reaches success rate *X*" is not a result on
its own — the question is whether starting from PPO's corrupted-environment
policy beats starting from noise, with the same algorithm, the same budget and
the same hyperparameters. Only the **fine-tune − scratch** contrast answers
that, and it is the primary outcome.

Because a corruption only ever affects `variant="corrupted"`, the two fine-tune
arms and the two scratch arms train on *the same* original environment. The
scratch arm is therefore a shared control across both corruptions **and** across
the original `negate_both` study — `verify_corruptions.py` asserts the
environments are step-for-step identical, and `METRICS.md` reports the measured
agreement of the three independently-run scratch arms as a cross-check.

### What is not transferred, in this direction

PPO has no partition function, so the GFlowNet's `log Z` head is randomly
initialised. Its critic is not a substitute: `V(s)` is expected return-to-go
from the *current* state, whereas `log Z(c)` is the log partition function of
the whole trajectory distribution for the initial (start, goal) instance, so
copying one into the other would be a category error. The mirror direction gives
up exactly one head too (PPO's critic), which is what keeps the two comparable.
The `log Z` head is fresh in the fine-tuned **and** the scratch arm, so it is
never a difference between them.

---

## Why these two corruptions

The original study's corruption, `negate_both` (`u ↦ −u`), was selected by a
rule that takes the corruption with the **lowest zero-shot success** on the
original environment — i.e. it maximises how wrong the pretrained policy is at
the start of fine-tuning. In this direction that produced a **null** result
(success rate −0.025, Holm *p* = 1.00), unlike the mirror direction where it
produced strong negative transfer. Neither corruption here is selected that way,
and they sit on opposite sides of the interesting question:

**`large_multigoal` — a bigger, differently-shaped task.** Not a relabelling of
UMaze but a genuinely different one: 46 free cells instead of 7, a single reset
cell instead of a free choice, and a goal drawn from 7 designated cells instead
of anywhere. Corridors, junctions and dead ends are things UMaze does not
contain. The pretrained policy is not *wrong* on the original environment the
way a sign-flipped controller is; it is trained for a different distribution.

**`action_gain_jitter` — domain randomisation.** The support of `g` **contains
the original**: `g = 1` is interior to `U(0.9, 1.1)`. The corrupted environment
is a distribution over MDPs of which the original is a member, the plant is the
same point mass, and the gain is not observable, so the only added difficulty is
the unresolvable ±10% uncertainty. This is the one corruption in this codebase
for which *positive* transfer is the a-priori expectation — it is exactly the
setup domain randomisation is normally claimed to help.

### The honest caveat on `large_multigoal`

**It is not difficulty-preserving, and its zero-shot number is not a clean
transfer measurement.** `negate_both` is an isometry of the action box, so the
corrupted MDP is provably isomorphic to the original and *exactly as hard* —
that is what let the original study attribute its whole zero-shot gap to
transfer. Nothing of the kind holds here: the maze, its size, the start
distribution and the goal distribution all change together, so the zero-shot
number mixes *the policy is wrong here* with *the policy never saw a task
shaped like this*, and the pretrain arm's native score on a 12×9 maze is not
comparable to anything measured on UMaze.

This is stated rather than argued away because the **primary contrast is
unaffected by it**: `gfn_finetune_original` and `gfn_scratch_original` both run
the GFlowNet on the same original environment with the same budget and the same
hyperparameters, and differ only in initialisation. A confound in how hard the
*pretraining* task was cannot reach that comparison.

---

## The gain corruption is applied to `actuator_gear`, not to the action

The obvious implementation of `u ↦ g·u` — an `ActionWrapper` returning
`g * action` — is **silently half-broken on this study's interface**, and this
is the one place where the code deliberately departs from a literal reading of
the specification.

`PointEnv.step` begins with `np.clip(action, -1.0, 1.0)`, and MuJoCo clips again
against the actuators' `ctrlrange="-1 1"`. Every action in the shared
`Discrete(9)` bang-bang table is one of `{−1,0,+1}²`, i.e. already on that
boundary. So for a bang-bang agent, **`g > 1` is clipped straight back to the
nominal command and does nothing at all**: the realised corruption would be
`u ↦ min(g, 1)·u`, a one-sided *weakening* of the actuators over half the
sampling range, with a mean gain of ≈0.975 rather than 1.0. Nothing in the
results would reveal this — the runs complete and the numbers look plausible.

Measured, over 25 steps of `u = (1, 1)`, displacement relative to `g = 1`:

| implementation | `g = 0.9` | `g = 1.1` |
|---|---|---|
| `g * action` (naive) | 0.066 m | **0.000 m — dead** |
| `actuator_gear` (used here) | 0.066 m | 0.055 m |

A motor's gain is a property of the motor, downstream of the controller's
saturation limit, so the faithful place to apply it is MuJoCo's
`actuator_gear`, where delivered force is `gear × ctrl`, evaluated after every
clip. Both tails then do something, and they do comparable amounts.
`verify_corruptions.py::check_action_gain_is_two_sided` asserts this rather than
trusting it, and a further check asserts the gear is rescaled from the captured
nominal value on every reset so that thousands of training episodes cannot
compound it.

`g` is drawn once per `reset()` from the wrapper's own generator, re-seeded by
`reset(seed=...)`, so each of the 100 shared evaluation instances pins one gain
and every arm faces the identical sequence of faults.

---

## What is unchanged from the original study

`hyperparameters.py` is **byte-identical** to `../RL+gflowne/hyperparameters.py`
and to `../RL+PPO/hyperparameters.py` (SHA-256 `d01ab117…`) — same budgets, same
30 seeds, same network, same PPO and GFlowNet settings, same evaluation
protocol. The corruption is *not* edited into that file; it is a run-time
argument (`run_study.py --corruption`), recorded in every run's `config.json`,
so "same hyperparameters, only the corruption differs" stays a checkable claim:

```bash
shasum -a 256 hyperparameters.py ../RL+gflowne/hyperparameters.py
```

Also unchanged: the `Discrete(9)` bang-bang interface, `action_repeat = 5`, the
`t/T` time feature, the `[128,128]` Tanh trunk shared by both algorithms, the
exact-parameter-copy transfer with its 512-probe numerical verification, budgets
counted in environment steps, and the statistics (seed as the unit of analysis,
Welch + exact Mann-Whitney, Hedges' *g*, Holm-Bonferroni over the four metrics).

Changed only where a new corruption forced it:

- `environments/layouts.py` — `LARGE_MULTIGOAL` (imported from
  `gymnasium_robotics`, not transcribed) and `wall_mask`.
- `environments/wrappers.py` — `RandomActionGain`.
- `environments/corruptions.py` — the two new family members, plus
  `difficulty_preserving`, `STRUCTURAL_CORRUPTIONS`, `STOCHASTIC_CORRUPTIONS`.
- `environments/shortest_path.py` — geodesics now go through `wall_mask`, so a
  `'g'`/`'r'` marker cell is free space. Previously `get_geodesic` cast every
  cell with `int(v)` and would have raised on `'g'`; the silent version of that
  bug — treating a goal marker as an obstacle — would have inflated *L\** and
  flattered every agent's path efficiency on the large maze.
- `run_study.py` / `analyze.py` — `--corruption`, per-corruption run directories,
  and corruption-specific interpretation notes in the report.
- `combine_metrics.py` — derives its arm names and direction-specific prose from
  `run_study.STUDIES`, so one generator serves both directions.

`sanity_checks.py` is carried over unchanged but is **`negate_both`-specific**:
it asserts that the corruption leaves the maze layout untouched, that the two
geodesic distributions match bit-for-bit, and that the corruption has a signed
permutation `action_matrix`. None of those hold for either corruption here, so
`verify_corruptions.py` states and tests the properties that are load-bearing
for these two instead.

---

## Layout

```
hyperparameters.py         byte-identical to ../RL+gflowne's; SHA in every config.json
run_study.py               --study A --corruption <name>; runs the 3 arms x 30 seeds
run_all.sh                 both corruptions, sharded over worker processes
analyze.py                 --corruption <name> -> METRICS_<name>.{md,json}
combine_metrics.py         the two reports + ../RL+gflowne's -> METRICS.md
verify_corruptions.py      the assertions specific to these two corruptions
environments/              PointMaze, the corruption family, wrappers, geodesics
algorithms/                PPO, GFlowNet, and the two-way transfer
evaluation/                the single evaluation protocol, shared by both algorithms
runs/<corruption>/
    results/<arm>/seed_<n>/
        config.json        resolved hyperparameters + versions + fingerprints
        final_eval.json    100-episode frozen-policy metrics
        episodes.csv       raw per-episode records
        curve.csv          learning curve
        zero_shot.json     pretrain arm: performance on the original env, unadapted
        transfer.json      fine-tune arm: the verified weight copy
        train_log.csv      GFlowNet Trajectory-Balance loss trace
    checkpoints/<arm>/seed_<n>/model.{zip,pt}
    logs/                  per-shard run logs
METRICS.md                 the combined report
METRICS_<corruption>.md    the full per-corruption report
```

**Checkpoints are not tracked in git.** `runs/<corruption>/checkpoints/` holds the
trained weights for all 30 seeds of all three arms (~79 MB per study) and is
excluded by `.gitignore`. Everything needed to verify the reported numbers is
tracked: `results/` carries the per-seed `final_eval.json`, `episodes.csv`,
`curve.csv`, `zero_shot.json`, `transfer.json` and resolved `config.json`, and
the `METRICS*.md` reports are regenerated from those by `analyze.py`. Run
`./run_all.sh` to reproduce the checkpoints.
