"""Aggregate one study's runs into the reported metrics.

    python3 analyze.py

Reads ``results/<arm>/seed_*/`` and writes ``metrics_summary.json`` plus a
human-readable ``METRICS.md``.  Byte-identical in both study directories.

Unit of analysis
----------------
The independent experimental unit is a **trained model** (one seed), never an
evaluation episode.  Each model contributes exactly one number per metric --
its mean over the shared 100-instance evaluation set -- and inference is done
across the 5 seeds.  Treating the 100 episodes as 500 independent samples
would inflate n by two orders of magnitude and manufacture significance; the
episodes are one *measurement* of one model, not 100 experiments.

Because every model is evaluated on the identical 100 instances
(``EVAL_SEED_BASE + i``), evaluation-instance variance is common to all arms
and drops out of the contrasts.

Primary outcome
---------------
``finetune`` - ``scratch`` on final success rate: does initialising algorithm 2
from a policy that algorithm 1 learned on the corrupted environment beat
initialising it randomly, given the same algorithm, budget and hyperparameters?
Three secondary metrics (mean reward, steps to goal, path efficiency) are
reported alongside it, and the family of four is Holm-Bonferroni corrected so
that reporting four outcomes does not inflate the false-positive rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import hyperparameters as HP
from run_study import STUDIES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

#: Metrics carried through the full inferential pipeline. ``higher_is_better``
#: only controls how the summary is *worded*; every test is two-sided.
METRICS = [
    ("success_rate", "Success rate", True),
    ("mean_reward", "Mean return", True),
    ("mean_steps_to_goal", "Steps to goal (successful eps)", False),
    ("mean_path_efficiency", "Path efficiency L*/L", True),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def seed_dirs(arm: str) -> list[Path]:
    return sorted((RESULTS / arm).glob("seed_*"),
                  key=lambda p: int(p.name.split("_")[1]))


def load_arm(arm: str, filename: str = "final_eval.json") -> dict[int, dict]:
    out = {}
    for d in seed_dirs(arm):
        f = d / filename
        if f.exists():
            out[int(d.name.split("_")[1])] = json.loads(f.read_text())
    return out


def load_curves(arm: str) -> dict[int, list[dict]]:
    import csv
    out = {}
    for d in seed_dirs(arm):
        f = d / "curve.csv"
        if not f.exists():
            continue
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k, v in r.items():
                if k != "phase":
                    r[k] = float(v) if v not in ("", "nan") else float("nan")
        out[int(d.name.split("_")[1])] = rows
    return out


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe(values: list[float]) -> dict:
    """Mean, SD across seeds, and a t-based 95% CI on the mean."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"),
                    sem=float("nan"), ci95_low=float("nan"), ci95_high=float("nan"))
    mean = float(v.mean())
    if n == 1:
        return dict(n=1, mean=mean, sd=float("nan"), sem=float("nan"),
                    ci95_low=float("nan"), ci95_high=float("nan"), values=v.tolist())
    sd = float(v.std(ddof=1))
    sem = sd / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * sem
    return dict(n=n, mean=mean, sd=sd, sem=sem,
                ci95_low=mean - half, ci95_high=mean + half, values=v.tolist())


# ---------------------------------------------------------------------------
# Inferential statistics
# ---------------------------------------------------------------------------

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardised mean difference (Hedges' g).

    Cohen's d is biased upward at small n; with n = 5 per arm the correction
    factor is material (~5%), so g is reported rather than d.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s_pooled = math.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pooled == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / s_pooled
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def common_language(a: np.ndarray, b: np.ndarray) -> float:
    """P(a random draw from ``a`` exceeds a random draw from ``b``), ties at 0.5."""
    if not len(a) or not len(b):
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def contrast(a_vals: list[float], b_vals: list[float]) -> dict:
    """Compare two arms on one metric.

    Two tests are reported because each answers a different objection:
    Welch's t-test does not assume equal variances (the arms can easily differ
    in spread), and the exact Mann-Whitney U makes no distributional assumption
    at all, which matters at n = 5.  Both are two-sided.  Effect sizes are
    reported unconditionally, since with n = 5 an absent p-value is far more
    often low power than a real null.
    """
    a = np.asarray([x for x in a_vals if x is not None and np.isfinite(x)], float)
    b = np.asarray([x for x in b_vals if x is not None and np.isfinite(x)], float)
    out = dict(n_a=len(a), n_b=len(b),
               mean_a=float(a.mean()) if len(a) else float("nan"),
               mean_b=float(b.mean()) if len(b) else float("nan"))
    out["difference"] = out["mean_a"] - out["mean_b"]
    if len(a) < 2 or len(b) < 2:
        out.update(welch_t=float("nan"), welch_p=float("nan"),
                   mannwhitney_u=float("nan"), mannwhitney_p=float("nan"),
                   hedges_g=float("nan"), common_language=float("nan"))
        return out

    if a.var(ddof=1) == 0 and b.var(ddof=1) == 0:
        # Both arms constant: t is undefined but the comparison is not.
        out.update(welch_t=float("nan"),
                   welch_p=0.0 if a.mean() != b.mean() else 1.0)
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        out.update(welch_t=float(t), welch_p=float(p))

    try:
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    except ValueError:  # exact method rejects ties
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    out.update(mannwhitney_u=float(u), mannwhitney_p=float(pu),
               hedges_g=hedges_g(a, b), common_language=common_language(a, b))
    return out


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, dict]:
    """Holm's step-down correction over a family of contrasts.

    Uniformly more powerful than Bonferroni and, unlike FDR control, bounds the
    probability of *any* false positive in the family -- the right guarantee
    when each metric is reported as a claim in its own right.
    """
    items = [(k, p) for k, p in pvalues.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)   # enforce monotonicity of step-down
        out[k] = dict(p_raw=p, p_holm=running, significant_at_05=bool(running < 0.05))
    for k, p in pvalues.items():
        out.setdefault(k, dict(p_raw=p, p_holm=float("nan"), significant_at_05=False))
    return out


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------

def steps_to_threshold(rows: list[dict], threshold: float) -> float | None:
    """First curve point at which success rate reaches ``threshold``.

    ``None`` (serialised as null) means the threshold was never reached within
    the budget.  Reported as a *count* of seeds reaching it plus the median
    over those that did, because averaging a censored value would silently
    reward an arm for never getting there.
    """
    for r in sorted(rows, key=lambda x: x["steps_in_phase"]):
        if r["success_rate"] >= threshold:
            return float(r["steps_in_phase"])
    return None


def threshold_summary(curves: dict[int, list[dict]], threshold: float) -> dict:
    hit = {s: steps_to_threshold(rows, threshold) for s, rows in curves.items()}
    reached = [v for v in hit.values() if v is not None]
    return dict(
        threshold=threshold,
        n_seeds=len(hit),
        n_reached=len(reached),
        median_steps=float(np.median(reached)) if reached else None,
        mean_steps=float(np.mean(reached)) if reached else None,
        per_seed=hit,
    )


def jumpstart(curves: dict[int, list[dict]]) -> dict:
    """Success rate at step 0 of phase 2 -- before any adaptation.

    For the fine-tuning arm this is the zero-shot transfer of the pretrained
    policy; for the scratch arm it is the random-initialisation baseline. Their
    difference is the classic *jumpstart* transfer metric.
    """
    vals = []
    for rows in curves.values():
        zero = [r for r in rows if r["steps_in_phase"] == 0]
        if zero:
            vals.append(zero[0]["success_rate"])
    return describe(vals)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_ci(d, nd=3):
    if d["n"] < 2:
        return fmt(d["mean"], nd)
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}  [{d['ci95_low']:.{nd}f}, {d['ci95_high']:.{nd}f}]"


def main() -> None:
    study_file = RESULTS / "study.json"
    if not study_file.exists():
        raise SystemExit("no results/study.json -- run run_study.py first")
    meta = json.loads(study_file.read_text())
    study = STUDIES[meta["name"]]
    arms = study["arms"]

    pre = load_arm(arms["pretrain"])                       # native, corrupted
    pre_zero = load_arm(arms["pretrain"], "zero_shot.json")  # zero-shot, original
    fine = load_arm(arms["finetune"])
    scratch = load_arm(arms["scratch"])
    fine_curves = load_curves(arms["finetune"])
    scratch_curves = load_curves(arms["scratch"])

    seeds = sorted(fine)
    summary: dict = dict(
        study=study["name"],
        title=study["title"],
        pretrain_algo=study["pretrain_algo"],
        finetune_algo=study["finetune_algo"],
        corruption=HP.CORRUPTION,
        seeds=seeds,
        hyperparameters=HP.resolved(),
        arms={},
        contrasts={},
        sample_efficiency={},
        transfer_verification={},
    )

    # ---- descriptive -----------------------------------------------------
    named = {
        f"{arms['pretrain']} (native, corrupted env)": pre,
        f"{arms['pretrain']} (ZERO-SHOT on original)": pre_zero,
        arms["finetune"]: fine,
        arms["scratch"]: scratch,
    }
    for label, data in named.items():
        summary["arms"][label] = {
            key: describe([data[s][key] for s in sorted(data)])
            for key, _, _ in METRICS
        }
        summary["arms"][label]["n_seeds"] = len(data)

    # ---- primary contrasts ----------------------------------------------
    raw_p = {}
    for key, label, higher in METRICS:
        c = contrast([fine[s][key] for s in sorted(fine)],
                     [scratch[s][key] for s in sorted(scratch)])
        c["metric"] = label
        c["higher_is_better"] = higher
        summary["contrasts"][key] = c
        raw_p[key] = c["welch_p"]
    holm = holm_bonferroni(raw_p)
    for key in summary["contrasts"]:
        summary["contrasts"][key]["holm"] = holm[key]

    # ---- sensitivity: drop the seeds whose training envs touch the eval set
    keep = [s for s in sorted(fine) if s not in HP.EVAL_OVERLAP_SEEDS]
    if len(keep) < len(fine):
        summary["sensitivity_excluding_eval_overlap_seeds"] = dict(
            excluded=list(HP.EVAL_OVERLAP_SEEDS),
            n_remaining=len(keep),
            **{key: contrast([fine[s][key] for s in keep if s in fine],
                             [scratch[s][key] for s in keep if s in scratch])
               for key, _, _ in METRICS},
        )

    # ---- corruption damage ----------------------------------------------
    summary["corruption_damage"] = contrast(
        [pre_zero[s]["success_rate"] for s in sorted(pre_zero)],
        [pre[s]["success_rate"] for s in sorted(pre)],
    )

    # ---- sample efficiency & jumpstart ----------------------------------
    for thr in HP.SUCCESS_THRESHOLDS:
        summary["sample_efficiency"][f"success_{thr}"] = dict(
            finetune=threshold_summary(fine_curves, thr),
            scratch=threshold_summary(scratch_curves, thr),
        )
    summary["jumpstart"] = dict(
        finetune=jumpstart(fine_curves),
        scratch=jumpstart(scratch_curves),
    )
    summary["jumpstart"]["difference"] = (
        summary["jumpstart"]["finetune"]["mean"] - summary["jumpstart"]["scratch"]["mean"]
    )

    # ---- transfer verification ------------------------------------------
    for d in seed_dirs(arms["finetune"]):
        f = d / "transfer.json"
        if f.exists():
            t = json.loads(f.read_text())
            summary["transfer_verification"][d.name] = {
                k: t[k] for k in (
                    "source_algo", "target_algo", "n_parameters_transferred",
                    "max_abs_logit_difference", "max_abs_probability_difference",
                    "greedy_action_agreement", "equivalent",
                ) if k in t
            }

    (ROOT / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: None if o != o else float(o))
    )

    # ---- markdown --------------------------------------------------------
    L: list[str] = []
    A = L.append
    A(f"# Study {study['name']} — metrics")
    A("")
    A(f"**{study['title']}**")
    A("")
    A(f"- Environment: `{HP.ENV_ID}`, sparse reward, 300-step episodes")
    A(f"- Corruption: `{HP.CORRUPTION}` (both actuators reversed, `u -> -u`)")
    A(f"- Budgets: {HP.PRETRAIN_STEPS:,} pretrain + {HP.ADAPT_STEPS:,} adapt (environment steps)")
    A(f"- Seeds: {seeds} (n = {len(seeds)} independent trained models per arm)")
    A(f"- Evaluation: {HP.EVAL_EPISODES} fixed instances (`seed = {HP.EVAL_SEED_BASE} + i`), frozen policy, deterministic")
    A(f"- `hyperparameters.py` SHA-256: `{HP.hyperparameter_fingerprint()}`")
    A("")

    A("## 1. Results by arm")
    A("")
    A("Mean ± SD across seeds, with a t-based 95% CI on the mean.")
    A("")
    A("| Arm | Success rate | Mean return | Steps to goal | Path efficiency |")
    A("|---|---|---|---|---|")
    for label, data in named.items():
        row = summary["arms"][label]
        A(f"| {label} | {fmt_ci(row['success_rate'])} | {fmt_ci(row['mean_reward'], 1)} "
          f"| {fmt_ci(row['mean_steps_to_goal'], 1)} | {fmt_ci(row['mean_path_efficiency'])} |")
    A("")

    A("## 2. Primary contrast — fine-tuned vs. from scratch")
    A("")
    A(f"Both arms are **{study['finetune_algo']}** on the original environment with the")
    A(f"same {HP.ADAPT_STEPS:,}-step budget and identical hyperparameters. They differ only in")
    A(f"initialisation: fine-tuned starts from the {study['pretrain_algo']} policy trained on the")
    A("corrupted environment, scratch starts randomly. Positive difference favours fine-tuning.")
    A("")
    A("| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p | Holm p | Mann-Whitney p |")
    A("|---|---|---|---|---|---|---|---|")
    for key, label, _ in METRICS:
        c = summary["contrasts"][key]
        nd = 1 if key in ("mean_reward", "mean_steps_to_goal") else 3
        A(f"| {label} | {fmt(c['mean_a'], nd)} | {fmt(c['mean_b'], nd)} | {fmt(c['difference'], nd)} "
          f"| {fmt(c['hedges_g'], 2)} | {fmt(c['welch_p'], 4)} | {fmt(c['holm']['p_holm'], 4)} "
          f"| {fmt(c['mannwhitney_p'], 4)} |")
    A("")

    sens = summary.get("sensitivity_excluding_eval_overlap_seeds")
    if sens:
        A("### Sensitivity — excluding seeds whose training envs touch the eval set")
        A("")
        A(f"Environment seeds are `seed * 1000 + i`, so seed(s) "
          f"{sens['excluded']} start their 8 training envs on instances that also")
        A(f"appear in the 100-instance evaluation set. Those seeds are **retained** in the")
        A(f"headline numbers -- dropping a seed after seeing its result would be the worse")
        A(f"error -- and the contrast is repeated without them here (n = {sens['n_remaining']}).")
        A("")
        A("| Metric | Fine-tuned | Scratch | Difference | Hedges' g | Welch p |")
        A("|---|---|---|---|---|---|")
        for key, label, _ in METRICS:
            c = sens[key]
            nd = 1 if key in ("mean_reward", "mean_steps_to_goal") else 3
            A(f"| {label} | {fmt(c['mean_a'], nd)} | {fmt(c['mean_b'], nd)} "
              f"| {fmt(c['difference'], nd)} | {fmt(c['hedges_g'], 2)} | {fmt(c['welch_p'], 4)} |")
        A("")

    A("## 3. Did the corruption actually break transfer?")
    A("")
    cd = summary["corruption_damage"]
    A(f"The pretrained {study['pretrain_algo']} model scores "
      f"**{fmt(cd['mean_b'])}** on the corrupted environment it was trained on, "
      f"and **{fmt(cd['mean_a'])}** zero-shot on the original one "
      f"(difference {fmt(cd['difference'])}, Welch p = {fmt(cd['welch_p'], 4)}).")
    A("")
    A("The corruption is an isometry of the action box, so the corrupted MDP is")
    A("isomorphic to the original and *exactly as hard*. Any gap here is therefore a")
    A("genuine transfer failure, not a difficulty confound.")
    A("")

    A("## 4. Jumpstart (success rate at step 0 of adaptation)")
    A("")
    js = summary["jumpstart"]
    A(f"- Fine-tuned: {fmt_ci(js['finetune'])}")
    A(f"- Scratch:    {fmt_ci(js['scratch'])}")
    A(f"- Jumpstart:  {fmt(js['difference'])}")
    A("")

    A("## 5. Sample efficiency — environment steps to reach a success rate")
    A("")
    A("Measured on the learning curve (30 episodes per point, every "
      f"{HP.CURVE_EVERY:,} steps). `n reached` matters as much as the median: a")
    A("median over seeds that never crossed the threshold would be meaningless.")
    A("")
    A("| Threshold | Arm | Seeds reaching | Median steps | Mean steps |")
    A("|---|---|---|---|---|")
    for thr in HP.SUCCESS_THRESHOLDS:
        block = summary["sample_efficiency"][f"success_{thr}"]
        for arm_name in ("finetune", "scratch"):
            b = block[arm_name]
            med = f"{b['median_steps']:,.0f}" if b["median_steps"] is not None else "never"
            mn = f"{b['mean_steps']:,.0f}" if b["mean_steps"] is not None else "never"
            A(f"| {thr:.2f} | {arm_name} | {b['n_reached']}/{b['n_seeds']} | {med} | {mn} |")
    A("")

    A("## 6. Transfer verification")
    A("")
    A("The cross-algorithm initialisation is an **exact parameter copy**, not a")
    A("distillation, so phase 2 provably begins at the pretrained policy and spends")
    A("no budget on the handover. This is checked numerically on 512 probe")
    A("observations drawn from the whole observation box, not from states either")
    A("policy happens to visit.")
    A("")
    A("| Seed | Direction | Params | max abs logit diff | Greedy agreement | Verified |")
    A("|---|---|---|---|---|---|")
    for name, t in sorted(summary["transfer_verification"].items()):
        A(f"| {name} | {t.get('source_algo')} → {t.get('target_algo')} "
          f"| {t.get('n_parameters_transferred')} | {t.get('max_abs_logit_difference', float('nan')):.2e} "
          f"| {fmt(t.get('greedy_action_agreement'))} | {'yes' if t.get('equivalent') else 'NO'} |")
    A("")
    A("What is *not* transferred, and cannot be: a GFlowNet has no value function and")
    A("PPO has no partition function, so exactly one head is randomly initialised in")
    A("each direction (PPO's critic in study B, the GFlowNet's `log Z` head in study A).")
    A("The two studies therefore give up the same amount, which is what keeps them")
    A("comparable.")
    A("")

    (ROOT / "METRICS.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {ROOT / 'metrics_summary.json'} and {ROOT / 'METRICS.md'}")


if __name__ == "__main__":
    main()
