"""Retrain imported seeds from scratch and check they reproduce, bit for bit.

    python3 verify_import.py                          # 2 seeds, both arms
    python3 verify_import.py --seeds 42 123 456       # more seeds
    python3 verify_import.py --corruption large_multigoal

``import_shared_arms.py`` takes ``ppo_pretrain_corrupted`` and
``ppo_scratch_original`` from the two earlier studies rather than retraining
them, on the argument that byte-identical code with byte-identical
hyperparameters and the same seed produces the same model.  That argument is
only worth as much as the pipeline's determinism, so this script measures it:
it retrains the named seeds into a throwaway tree and compares the SHA-256 of
every policy parameter against the imported model's, plus the 100-episode
evaluation the study actually reports.

A match means the imported arms are exactly what this directory would have
produced.  A mismatch means they are not, and that the arms must be retrained
here -- which is why this runs before the results are believed, not after.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import run_study as RS

ROOT = Path(__file__).resolve().parent

#: Metrics compared alongside the parameter hash. The hash is the strict test;
#: these say *how far apart* two models are when it fails, which is the
#: difference between "nondeterministic in the last bit" and "a different run".
METRICS = ("success_rate", "mean_reward", "mean_steps_to_goal",
           "mean_path_efficiency")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def compare(label: str, fresh_dir: Path, imported_dir: Path) -> dict:
    """Compare one retrained seed against its imported counterpart."""
    f_cfg, i_cfg = read(fresh_dir / "config.json"), read(imported_dir / "config.json")
    f_ev, i_ev = read(fresh_dir / "final_eval.json"), read(imported_dir / "final_eval.json")

    same_params = f_cfg["parameter_fingerprint"] == i_cfg["parameter_fingerprint"]
    deltas = {m: abs(f_ev[m] - i_ev[m]) for m in METRICS if m in f_ev and m in i_ev}
    same_metrics = all(d == 0.0 for d in deltas.values())

    mark = "OK " if (same_params and same_metrics) else "FAIL"
    print(f"  [{mark}] {label}")
    print(f"         parameters  fresh={f_cfg['parameter_fingerprint']} "
          f"imported={i_cfg['parameter_fingerprint']}"
          f"{'' if same_params else '   <-- DIFFER'}")
    if not same_metrics:
        for m, d in deltas.items():
            if d:
                print(f"         {m}: fresh={f_ev[m]} imported={i_ev[m]} (|d|={d})")
    else:
        print(f"         evaluation  identical on all {len(deltas)} metrics "
              f"(success={f_ev['success_rate']:.3f})")

    return dict(seed_dir=label, parameters_identical=same_params,
                metrics_identical=same_metrics,
                fresh_fingerprint=f_cfg["parameter_fingerprint"],
                imported_fingerprint=i_cfg["parameter_fingerprint"],
                metric_absolute_differences=deltas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corruption", default="action_gain_jitter")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 2000])
    ap.add_argument("--keep", action="store_true",
                    help="keep the throwaway tree instead of deleting it")
    args = ap.parse_args()

    study = RS.STUDIES["C"]
    RS.select_corruption(args.corruption)
    imported_root = RS.RESULTS

    # Retrain into a sibling tree so a verification run can never overwrite,
    # or be mistaken for, a result the study reports.
    sandbox = ROOT / "runs" / "_verify" / args.corruption
    RS.RESULTS = sandbox / "results"
    RS.CHECKPOINTS = sandbox / "checkpoints"

    print(f"verifying {args.corruption}, seeds {args.seeds}")
    print(f"  imported : {imported_root}")
    print(f"  retrained: {RS.RESULTS}\n")

    t0 = time.time()
    results = []
    for seed in args.seeds:
        print(f"seed {seed}: retraining pretrain + scratch (2 x 300k steps)")
        RS.run_pretrain(study, seed, force=True)
        RS.run_phase2(study, seed, "scratch", None, force=True)
        for arm in (study["arms"]["pretrain"], study["arms"]["scratch"]):
            results.append(compare(
                f"{arm}/seed_{seed}",
                RS.RESULTS / arm / f"seed_{seed}",
                imported_root / arm / f"seed_{seed}",
            ))

    ok = all(r["parameters_identical"] and r["metrics_identical"] for r in results)
    report = dict(
        corruption=args.corruption,
        seeds=args.seeds,
        n_comparisons=len(results),
        all_identical=ok,
        comparisons=results,
        wall_clock_seconds=round(time.time() - t0, 1),
        conclusion=("Retraining reproduces the imported models exactly; the "
                    "imported arms are what this directory would have produced."
                    if ok else
                    "Retraining did NOT reproduce the imported models. The "
                    "imported arms must be retrained in this directory before "
                    "any result that uses them is reported."),
    )
    (ROOT / "verify_import.json").write_text(json.dumps(report, indent=2))

    if not args.keep:
        shutil.rmtree(ROOT / "runs" / "_verify", ignore_errors=True)

    print(f"\n{report['conclusion']}")
    print(f"wrote {ROOT / 'verify_import.json'} ({time.time() - t0:.0f}s)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
