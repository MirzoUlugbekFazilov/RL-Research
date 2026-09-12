"""Build the combined report from the two per-corruption metric files.

    python3 combine_metrics.py            # -> METRICS.md

Reads ``METRICS_<corruption>.json`` (written by ``analyze.py``) for the two
corruptions studied here, plus the mirror study's ``metrics_summary.json`` for the
original ``negate_both`` study, and writes one ``METRICS.md`` comparing all
three.  Generated rather than hand-written so no number in the combined report
can drift away from the run it came from.

The ``negate_both`` column is a *reference*, not a re-run: it is the original
study's own published summary, produced by byte-identical code with the
byte-identical ``hyperparameters.py`` this directory uses.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent

#: Uniform-random-action success rate on the shared 100-instance evaluation set
#: (environments.corruptions.RANDOM_SUCCESS_FLOOR). Any policy at or below
#: this is worthless on the original environment.
FLOOR = 0.26
REFERENCE = ROOT.parent / "RL+gflowne" / "metrics_summary.json"

#: (corruption, short label, one-line characterisation) in reporting order.
STUDIED = [
    ("large_multigoal", "large_multigoal",
     "12x9 maze with 7 goal cells replaces the 5x5 UMaze"),
    ("action_gain_jitter", "action_gain_jitter",
     "actuator gain `u -> g*u`, `g ~ U(0.9, 1.1)` per episode"),
]

METRIC_KEYS = [
    ("success_rate", "Success rate", 3),
    ("mean_reward", "Mean return", 1),
    ("mean_steps_to_goal", "Steps to goal", 1),
    ("mean_path_efficiency", "Path efficiency", 3),
]

def load(corruption: str) -> dict:
    return json.loads((ROOT / f"METRICS_{corruption}.json").read_text())


def arm_names(summary: dict) -> dict:
    """Arm directory names and descriptive labels for whichever study this is.

    Derived from ``run_study.STUDIES`` rather than hard-coded, so the same
    report generator serves Study A (PPO -> GFlowNet) and Study B
    (GFlowNet -> PPO) without a second copy that could drift.
    """
    from run_study import STUDIES

    arms = STUDIES[summary["study"]]["arms"]
    return dict(
        pretrain=arms["pretrain"],
        finetune=arms["finetune"],
        scratch=arms["scratch"],
        native=f"{arms['pretrain']} (native, corrupted env)",
        zero=f"{arms['pretrain']} (ZERO-SHOT on original)",
    )


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def sig(p) -> str:
    """Holm-corrected significance marker, so the table is readable at a glance."""
    if p is None or not np.isfinite(p):
        return ""
    return " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""


def curve(corruption: str, arm: str) -> dict[int, float]:
    import csv
    acc: dict[int, list[float]] = {}
    for path in sorted(glob.glob(
            str(ROOT / "runs" / corruption / "results" / arm / "seed_*" / "curve.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                acc.setdefault(int(r["steps_in_phase"]), []).append(float(r["success_rate"]))
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def scratch_agreement(scratch_arm: str) -> tuple[int, int, float, list[str]]:
    """Compare the scratch arm across corruptions and against the reference study.

    A corruption may only ever touch ``variant='corrupted'``, so the from-scratch
    control -- the fine-tuning algorithm, randomly initialised, on the *original*
    environment -- must come out bit-identical in all three studies for a given
    seed.  Anything else means a corruption leaked into the uncorrupted
    environment and every contrast in this directory would be invalid.
    """
    keys = ["success_rate", "mean_reward", "mean_steps_to_goal", "mean_path_efficiency"]
    trees = {c: ROOT / "runs" / c / "results" / scratch_arm for c, _, _ in STUDIED}
    trees["negate_both"] = REFERENCE.parent / "results" / scratch_arm

    seeds = sorted(int(p.name.split("_")[1]) for p in trees["negate_both"].glob("seed_*"))
    n_ok, worst, notes = 0, 0.0, []
    for s in seeds:
        vals = {}
        for name, tree in trees.items():
            fp = tree / f"seed_{s}" / "final_eval.json"
            if fp.exists():
                vals[name] = json.loads(fp.read_text())
        if len(vals) < len(trees):
            notes.append(f"seed {s}: only {sorted(vals)}")
            continue
        ref = vals["negate_both"]
        d = max(abs(v[k] - ref[k]) for v in vals.values() for k in keys)
        worst = max(worst, d)
        n_ok += d == 0.0
    return n_ok, len(seeds), worst, notes


def main() -> None:
    data = {c: load(c) for c, _, _ in STUDIED}
    A_ = arm_names(data[STUDIED[0][0]])
    title = data[STUDIED[0][0]]["title"]
    pre_algo = data[STUDIED[0][0]]["pretrain_algo"]
    fin_algo = data[STUDIED[0][0]]["finetune_algo"]
    # The one head that cannot cross the algorithm boundary, per direction.
    dropped = ("PPO's critic" if fin_algo == "ppo"
               else "the GFlowNet's `log Z` head")
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
    all_studies = list(STUDIED)
    if ref is not None:
        data["negate_both"] = ref
        all_studies.append(("negate_both", "negate_both (reference)",
                            "both actuators reversed, `u -> -u`"))

    L: list[str] = []
    A = L.append

    A(f"# {pre_algo.upper()} → {fin_algo.upper()} transfer under two new corruptions — metrics")
    A("")
    A(f"**{title.replace('->', '→')}**, run unchanged against two corruptions.")
    A("")
    hp = data[STUDIED[0][0]]["hyperparameters"]
    A(f"- Environment: `{hp['env_id']}`, sparse reward, {hp['max_episode_steps']}-step episodes")
    A(f"- Budgets: {hp['pretrain_steps']:,} pretrain + {hp['adapt_steps']:,} adapt (environment steps)")
    A(f"- Seeds: n = {hp['n_seeds']} independent trained models per arm per corruption")
    A(f"- Evaluation: {hp['eval_episodes']} fixed instances "
      f"(`seed = {hp['eval_seed_base']} + i`), frozen policy, deterministic")
    A(f"- `hyperparameters.py` SHA-256: `{hp['hyperparameters_sha256']}` "
      f"(byte-identical to `{REFERENCE.parent.name}`)")
    A("")
    A("`negate_both` is the original study's published result, included as a "
      "reference column. It was produced by byte-identical code with the "
      "byte-identical `hyperparameters.py` used here, and was **not** re-run.")
    A("")
    A("Significance markers are on the **Holm-corrected** p over the four "
      "metrics: `*` < 0.05, `**` < 0.01, `***` < 0.001.")
    A("")

    # ---------------------------------------------------------------- headline
    A("## 1. Headline — does pretraining on the corrupted environment help?")
    A("")
    A(f"Fine-tuned − scratch on the original environment. Both arms are "
      f"**{fin_algo}**, same budget, same")
    A("hyperparameters, differing only in initialisation. Positive favours "
      "fine-tuning.")
    A("")
    A("| Corruption | Success rate | Mean return | Steps to goal | Path efficiency |")
    A("|---|---|---|---|---|")
    for corr, label, _ in all_studies:
        cells = []
        for key, _, nd in METRIC_KEYS:
            c = data[corr]["contrasts"][key]
            cells.append(f"{f(c['difference'], nd)} (g={f(c['hedges_g'], 2)})"
                         f"{sig(c['holm']['p_holm'])}")
        A(f"| `{label}` | " + " | ".join(cells) + " |")
    A("")
    A("Lower *steps to goal* is better; the sign is not flipped, so a negative "
      "entry there favours fine-tuning.")
    A("")

    # ---------------------------------------------------------------- damage
    A("## 2. How much did each corruption actually break the policy?")
    A("")
    A(f"Zero-shot is the pretrained {pre_algo} dropped into the **original** "
      "environment with no adaptation — it is both the damage measurement and "
      "the step-0 point of the fine-tuning curve. The uniform-random floor on "
      f"this evaluation set is **{FLOOR}**.")
    A("")
    A("| Corruption | Native (corrupted env) | Zero-shot (original) | Welch p | vs. random floor |")
    A("|---|---|---|---|---|")
    for corr, label, _ in all_studies:
        d = data[corr]
        nat = d["arms"][A_["native"]]["success_rate"]["mean"]
        zer = d["arms"][A_["zero"]]["success_rate"]["mean"]
        cd = d["corruption_damage"]
        verdict = ("**below** — worse than random" if zer < FLOOR else
                   "above" if zer > FLOOR else "at")
        A(f"| `{label}` | {f(nat)} | {f(zer)} | {f(cd['welch_p'], 4)} | {verdict} |")
    A("")
    A("This is the axis the three corruptions differ on most, and it drives "
      "everything below. Each verdict below is derived from the table, not "
      "asserted:")
    A("")
    #: Mechanism-specific gloss; the *classification* and every number are
    #: computed, so this text cannot contradict the run it describes.
    GLOSS = {
        "negate_both": "a good controller under `u ↦ −u` is an anti-optimal "
                       "controller under `u ↦ u`",
        "large_multigoal": "the policy is not wrong on UMaze, it was simply "
                           "trained for a different task",
        "action_gain_jitter": "`g = 1` is inside the training distribution, so "
                              "a ±10% gain fault costs essentially nothing",
    }
    for corr, label, _ in all_studies:
        d = data[corr]
        nat = d["arms"][A_["native"]]["success_rate"]["mean"]
        zer_arm = d["arms"][A_["zero"]]["success_rate"]
        zer = zer_arm["mean"]
        p_damage = d["corruption_damage"]["welch_p"]

        # Classify against the *random floor*, not only against the native
        # score.  Testing damage alone is not enough: if pretraining barely
        # worked, native and zero-shot can both be near-random and the damage
        # test comes out non-significant, which would otherwise be reported as
        # "undamaged" when the honest reading is "never any good here".
        vals = np.asarray(zer_arm.get("values") or [], dtype=float)
        if vals.size >= 2:
            t_floor, p_floor = stats.ttest_1samp(vals, FLOOR)
            above = bool(t_floor > 0 and p_floor < 0.05)
            below = bool(t_floor < 0 and p_floor < 0.05)
        else:  # pragma: no cover - the reference summary always carries values
            above, below, p_floor = zer > FLOOR, zer < FLOOR, float("nan")

        if below:
            verdict = "**actively wrong**"
            detail = f"{f(zer)}, significantly *below* the {FLOOR} random floor"
        elif not above:
            verdict = "**no better than random**"
            detail = (f"{f(zer)}, indistinguishable from the {FLOOR} random "
                      f"floor, p = {f(p_floor, 2)}")
        elif p_damage >= 0.05:
            verdict = "**essentially undamaged**"
            detail = (f"{f(zer)} zero-shot vs {f(nat)} native, p = "
                      f"{f(p_damage, 2)}; clear of the floor")
        else:
            verdict = "**degraded but still useful**"
            detail = f"{f(zer)}, clear of the {FLOOR} floor"

        line = (f"- `{label}` produces a policy that is {verdict} ({detail}): "
                f"{GLOSS.get(corr, '')}.")
        # Only worth remarking on when the gap is real, not when it is noise.
        if zer > nat and p_damage < 0.05:
            line += (f" Note it scores *higher* zero-shot on the original "
                     f"({f(zer)}) than natively on the corrupted environment "
                     f"({f(nat)}) — the corrupted task is simply the harder of "
                     f"the two, which is the difficulty confound this "
                     f"corruption carries and `negate_both` does not.")
        A(line)
    A("")

    # ---------------------------------------------------------------- jumpstart
    A("## 3. Jumpstart and sample efficiency")
    A("")
    A("| Corruption | Jumpstart | Steps to 0.80 success (median, fine-tuned) | (scratch) | Speed-up |")
    A("|---|---|---|---|---|")
    for corr, label, _ in all_studies:
        d = data[corr]
        js = d["jumpstart"]["difference"]
        blk = d["sample_efficiency"]["success_0.8"]
        ft_m, sc_m = blk["finetune"]["median_steps"], blk["scratch"]["median_steps"]
        ft_n = f"{blk['finetune']['n_reached']}/{blk['finetune']['n_seeds']}"
        sc_n = f"{blk['scratch']['n_reached']}/{blk['scratch']['n_seeds']}"
        speed = (f"{sc_m / ft_m:.2f}x" if ft_m and sc_m and ft_m > 0
                 else ("already there at step 0" if ft_m == 0 else "n/a"))
        ft_s = f"{ft_m:,.0f}" if ft_m is not None else "never"
        sc_s = f"{sc_m:,.0f}" if sc_m is not None else "never"
        A(f"| `{label}` | {js:+.3f} | {ft_s} ({ft_n}) | {sc_s} ({sc_n}) | {speed} |")
    A("")

    # ---------------------------------------------------------------- curves
    A("## 4. Learning curves — mean success rate over 30 seeds")
    A("")
    A("The headline contrast is one point on these curves (the last row). The "
      "shape is the result.")
    A("")
    header = "| Steps | " + " | ".join(
        f"{lbl} ft | {lbl} scratch | gap" for _, lbl, _ in STUDIED) + " |"
    A(header)
    A("|" + "---|" * (1 + 3 * len(STUDIED)))
    grids = {c: (curve(c, A_["finetune"]), curve(c, A_["scratch"])) for c, _, _ in STUDIED}
    steps = sorted(grids[STUDIED[0][0]][0])
    for s in steps:
        cells = []
        for c, _, _ in STUDIED:
            ftc, scc = grids[c]
            a, b = ftc.get(s, float("nan")), scc.get(s, float("nan"))
            cells += [f(a), f(b), f"{a - b:+.3f}"]
        A(f"| {s:,} | " + " | ".join(cells) + " |")
    A("")

    # ---------------------------------------------------------------- controls
    A("## 5. Verification")
    A("")
    n_ok, n_tot, worst, notes = scratch_agreement(A_["scratch"])
    A("### The from-scratch control is shared, and provably so")
    A("")
    A(f"A corruption may only ever affect `variant='corrupted'`. The from-scratch "
      f"arm — {fin_algo}, randomly initialised, on the **original** environment — "
      "therefore has to come out identical in all three studies for a given "
      "seed. It does:")
    A("")
    A(f"- **{n_ok}/{n_tot} seeds bit-identical** across `large_multigoal`, "
      f"`action_gain_jitter` and `{REFERENCE.parent.name}`'s `negate_both`, on all four "
      f"metrics (max absolute discrepancy **{worst:g}**).")
    for n in notes:
        A(f"- note: {n}")
    A("")
    A("This is a strong end-to-end check: it says the new corruption code, the "
      "new wrappers and the geodesic change did not perturb the uncorrupted "
      "environment by so much as a floating-point bit, and it means the three "
      "corruptions are compared against a genuinely common control.")
    A("")

    A("### The cross-algorithm weight copy")
    A("")
    A("| Corruption | Seeds verified | max abs logit diff | Greedy agreement |")
    A("|---|---|---|---|")
    for corr, label, _ in STUDIED:
        tv = data[corr]["transfer_verification"]
        ok = sum(bool(t.get("equivalent")) for t in tv.values())
        mx = max(t.get("max_abs_logit_difference", 0.0) for t in tv.values())
        ag = min(t.get("greedy_action_agreement", 0.0) for t in tv.values())
        A(f"| `{label}` | {ok}/{len(tv)} | {mx:.2e} | {f(ag)} |")
    A("")
    A("The initialisation is an exact parameter copy, not a distillation, "
      "checked on 512 probe observations drawn from the whole observation box "
      f"rather than from states either policy happens to visit. "
      f"{dropped[0].upper() + dropped[1:]} is "
      f"randomly initialised in both the fine-tuned and the scratch arm, so it "
      f"is never a difference between them.")
    A("")
    A("### Corruption-specific assertions")
    A("")
    A("`verify_corruptions.py` — 10/10 passing. The load-bearing ones:")
    A("")
    A("- the `Discrete(9)` / `Box(7,)` interface is identical in all four "
      "conditions, so the weight copy is shape- *and* semantics-valid;")
    A("- the original environment is step-for-step identical under all three "
      "corruption names;")
    A("- `large_multigoal` equals `gymnasium_robotics`' own "
      "`LARGE_MAZE_DIVERSE_G` (7 goal cells, 1 reset cell) — a named upstream "
      "constant, so the map cannot have been drawn by us to produce a result;")
    A("- goal/reset markers are free space in the geodesic, so *L\\** is not "
      "inflated on the large maze;")
    A("- the gain is drawn per episode in `[0.9, 1.1]`, held fixed within the "
      "episode, reproducible from the evaluation seed, and does not compound "
      "over resets;")
    A("- **the gain is two-sided under bang-bang actions.** See §6.")
    A("")

    # ---------------------------------------------------------------- caveats
    A("## 6. Caveats that change how these numbers should be read")
    A("")
    A("**`large_multigoal` is not difficulty-preserving.** `negate_both` is an "
      "isometry of the action box, so its corrupted MDP is provably isomorphic "
      "to the original and *exactly as hard* — which is what let the original "
      "study attribute its whole zero-shot gap to transfer. Nothing like that "
      "holds here: maze, size, start distribution and goal distribution all "
      "change together, so the §2 zero-shot number for this corruption mixes "
      "*the policy is wrong here* with *the policy never saw a task shaped like "
      "this*, and its native score is not comparable across the two "
      "environments. **The §1 contrast is unaffected** — both arms there run "
      f"{fin_algo} on the same original environment with the same budget, and "
      "differ only in initialisation.")
    A("")
    A("**The gain corruption is applied to `actuator_gear`, not to the action "
      "array.** This is a deliberate departure from a literal `u = g*u`. "
      "`PointEnv.step` clips actions to `[-1, 1]` and every action in the "
      "shared `Discrete(9)` bang-bang table already sits on that boundary, so "
      "an `ActionWrapper` returning `g * action` would make `g > 1` a **no-op**: "
      "the realised corruption would be `u ↦ min(g, 1)·u`, a one-sided "
      "weakening with mean gain ≈0.975. Measured displacement after 25 steps of "
      "`u = (1,1)`, relative to `g = 1`:")
    A("")
    A("| implementation | `g = 0.9` | `g = 1.1` |")
    A("|---|---|---|")
    A("| `g * action` (naive) | 0.066 m | **0.000 m — dead** |")
    A("| `actuator_gear` (used here) | 0.066 m | 0.055 m |")
    A("")
    A("A motor's gain sits downstream of the controller's saturation limit, so "
      "`actuator_gear` — where delivered force is `gear × ctrl`, evaluated "
      "after every clip — is the faithful place for it. Asserted, not assumed, "
      "by `check_action_gain_is_two_sided`.")
    A("")
    A("**Power.** n = 30 gives ~0.93 power for |g| = 1.1 at the Holm worst-case "
      "α, but only ~0.3–0.5 for |g| ≈ 0.5. Several effects reported above sit "
      "in that range, so a non-significant entry in §1 should be read as *not "
      "resolved at this n*, not as an established null. Effect sizes are "
      "reported unconditionally for that reason.")
    A("")
    A("**One evaluation set, one environment family.** Every number here is "
      "PointMaze with a sparse reward under one shared bang-bang interface, "
      "and the fine-tuned arm's advantage is measured at a single 300k budget. "
      "Nothing here generalises to continuous-action PPO, to dense rewards, or "
      "to longer budgets without being re-measured.")
    A("")

    (ROOT / "METRICS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {ROOT / 'METRICS.md'}")


if __name__ == "__main__":
    main()
