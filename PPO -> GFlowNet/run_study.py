"""Run one cross-algorithm transfer study, end to end.

    python3 run_study.py --study A      # PPO(corrupted) -> GFlowNet(original)
    python3 run_study.py --study B      # GFlowNet(corrupted) -> PPO(original)

This file is **byte-identical** in ``RL+gflowne/`` and ``RL+PPO/``; the two
studies are the same code with the two algorithms' roles swapped, so nothing
except that swap can differ between them.

Three arms per seed
-------------------
======================  ====================================================
``pretrain``            algorithm 1, ``PRETRAIN_STEPS`` on the **corrupted**
                        environment.  Evaluated twice: natively (did it learn
                        the corrupted task at all?) and on the **original**
                        environment (**zero-shot** -- the "after corruption,
                        before adaptation" number).
``finetune``            algorithm 2, initialised from ``pretrain``'s policy by
                        exact weight copy, ``ADAPT_STEPS`` on the **original**
                        environment.
``scratch``             algorithm 2, randomly initialised, ``ADAPT_STEPS`` on
                        the **original** environment.
======================  ====================================================

``scratch`` is the control that makes the study interpretable, and it is not
optional.  "The fine-tuned model reaches success rate X" is not a result on its
own: the question is whether starting from a policy trained on the corrupted
environment *helps*, *hurts*, or does *nothing* relative to starting from
scratch with the identical algorithm, the identical budget and the identical
hyperparameters.  Only the ``finetune`` - ``scratch`` contrast answers that,
and it is the study's primary outcome.

Artifacts, per run::

    results/<arm>/seed_<n>/config.json        resolved config + versions + hashes
    results/<arm>/seed_<n>/final_eval.json    100-episode frozen metrics
    results/<arm>/seed_<n>/episodes.csv       raw per-episode records
    results/<arm>/seed_<n>/curve.csv          learning curve (phase 2 arms)
    results/<arm>/seed_<n>/zero_shot.json     pretrain arm only
    results/<arm>/seed_<n>/transfer.json      finetune arm only
    results/<arm>/seed_<n>/train_log.csv      GFlowNet TB-loss trace
    checkpoints/<arm>/seed_<n>/model.{zip,pt}
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import hyperparameters as HP
from algorithms.gflownet.agent import GFlowNetAgent
from algorithms.ppo.agent import PPOAgent
from evaluation.evaluator import evaluate_policy
from training.callbacks import CurveRecorder, PPOCurveCallback, gflownet_curve_callback

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# The two studies, defined as one table so the mirror symmetry is visible
# ---------------------------------------------------------------------------

STUDIES = {
    "A": dict(
        name="A",
        directory="RL+gflowne",
        title="PPO on corrupted -> GFlowNet fine-tuned on original",
        pretrain_algo="ppo",
        finetune_algo="gflownet",
        arms=dict(
            pretrain="ppo_pretrain_corrupted",
            finetune="gfn_finetune_original",
            scratch="gfn_scratch_original",
        ),
    ),
    "B": dict(
        name="B",
        directory="RL+PPO",
        title="GFlowNet on corrupted -> PPO fine-tuned on original",
        pretrain_algo="gflownet",
        finetune_algo="ppo",
        arms=dict(
            pretrain="gfn_pretrain_corrupted",
            finetune="ppo_finetune_original",
            scratch="ppo_scratch_original",
        ),
    ),
}


def package_versions() -> dict:
    import gymnasium
    import gymnasium_robotics
    import mujoco
    import stable_baselines3
    import torch

    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        gymnasium=gymnasium.__version__,
        gymnasium_robotics=gymnasium_robotics.__version__,
        mujoco=mujoco.__version__,
        stable_baselines3=stable_baselines3.__version__,
        torch=torch.__version__,
        numpy=np.__version__,
    )


# ---------------------------------------------------------------------------
# Agent construction -- every hyperparameter comes from hyperparameters.py
# ---------------------------------------------------------------------------

def fresh_ppo(variant: str, seed: int) -> PPOAgent:
    return PPOAgent.fresh(
        variant,
        corruption=HP.CORRUPTION,
        discrete=HP.DISCRETE,
        seed=seed,
        n_envs=HP.PPO_N_ENVS,
        action_repeat=HP.ACTION_REPEAT,
        time_feature=HP.TIME_FEATURE,
        policy_hidden=(HP.HIDDEN, HP.HIDDEN),
        hparams=dict(HP.PPO_HPARAMS, policy_kwargs=HP.ppo_policy_kwargs()),
    )


def fresh_gflownet(variant: str, seed: int) -> GFlowNetAgent:
    return GFlowNetAgent.fresh(
        variant, corruption=HP.CORRUPTION, seed=seed, cfg=HP.gflownet_config()
    )


def fresh_agent(algo: str, variant: str, seed: int):
    return fresh_ppo(variant, seed) if algo == "ppo" else fresh_gflownet(variant, seed)


def crossover_agent(algo: str, checkpoint: Path, seed: int):
    """Algorithm ``algo``, initialised from the *other* algorithm's checkpoint.

    Both branches are exact parameter copies verified numerically inside
    :mod:`algorithms.transfer`; neither performs any optimisation, so the
    fine-tuning phase provably starts at the pretrained policy and phase 2
    spends its whole budget on the original environment.
    """
    if algo == "ppo":
        return PPOAgent.from_gflownet(
            checkpoint,
            "original",
            corruption=HP.CORRUPTION,
            seed=seed,
            n_envs=HP.PPO_N_ENVS,
            action_repeat=HP.ACTION_REPEAT,
            policy_hidden=(HP.HIDDEN, HP.HIDDEN),
            time_feature=HP.TIME_FEATURE,
        )
    return GFlowNetAgent.from_ppo(
        checkpoint, "original", corruption=HP.CORRUPTION, seed=seed,
        cfg=HP.gflownet_config(),
    )


def checkpoint_path(arm: str, seed: int, algo: str) -> Path:
    ext = "zip" if algo == "ppo" else "pt"
    return CHECKPOINTS / arm / f"seed_{seed}" / f"model.{ext}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_eval(out_dir: Path, name: str, metrics, *, variant: str, extra: dict | None = None):
    payload = dict(metrics.summary(), eval_variant=variant, **(extra or {}))
    write_json(out_dir / f"{name}.json", payload)
    return payload


def dump_train_log(agent, out_dir: Path, name: str = "train_log.csv") -> None:
    log = getattr(agent, "train_log", None)
    if log:
        write_csv(out_dir / name, log)


# ---------------------------------------------------------------------------
# The three arms
# ---------------------------------------------------------------------------

def run_pretrain(study: dict, seed: int, force: bool) -> Path:
    """Phase 1: algorithm 1 learns the **corrupted** environment."""
    arm = study["arms"]["pretrain"]
    algo = study["pretrain_algo"]
    out = RESULTS / arm / f"seed_{seed}"
    ckpt = checkpoint_path(arm, seed, algo)
    if ckpt.exists() and (out / "final_eval.json").exists() and not force:
        print(f"  [{arm}/seed_{seed}] cached")
        return ckpt

    t0 = time.time()
    agent = fresh_agent(algo, "corrupted", seed)
    agent.learn(HP.PRETRAIN_STEPS)
    agent.save(ckpt)

    # Native: did phase 1 actually learn the corrupted task? If it did not,
    # any transfer result downstream would be a statement about a failed
    # pretraining run rather than about transfer.
    native = evaluate_policy(agent, "corrupted", corruption=HP.CORRUPTION,
                             n_episodes=HP.EVAL_EPISODES, seed=HP.EVAL_SEED_BASE)
    # Zero-shot: the same frozen model dropped into the ORIGINAL environment,
    # before any adaptation. This is the "damage the corruption did" number and
    # the starting point of the fine-tuning arm's curve.
    zero = evaluate_policy(agent, "original", corruption=HP.CORRUPTION,
                           n_episodes=HP.EVAL_EPISODES, seed=HP.EVAL_SEED_BASE)

    save_eval(out, "final_eval", native, variant="corrupted")
    save_eval(out, "zero_shot", zero, variant="original")
    write_csv(out / "episodes.csv", native.to_records())
    write_csv(out / "episodes_zero_shot.csv", zero.to_records())
    dump_train_log(agent, out)
    write_json(out / "config.json", dict(
        study=study["name"], arm=arm, phase="pretrain", algo=algo,
        train_variant="corrupted", budget_env_steps=HP.PRETRAIN_STEPS,
        seed=seed, consumed_env_steps=int(agent.num_timesteps),
        parameter_fingerprint=agent.parameter_fingerprint(),
        wall_clock_seconds=round(time.time() - t0, 1),
        hyperparameters=HP.resolved(), versions=package_versions(),
    ))
    print(f"  [{arm}/seed_{seed}] native={native.success_rate:.3f} "
          f"zero_shot={zero.success_rate:.3f} ({time.time() - t0:.0f}s)")
    agent.close()
    return ckpt


def run_phase2(study: dict, seed: int, kind: str, pretrained: Path | None,
               force: bool) -> None:
    """Phase 2: algorithm 2 trains on the **original** environment.

    ``kind='finetune'`` starts from ``pretrained`` (the other algorithm's
    weights); ``kind='scratch'`` starts randomly.  Everything else -- budget,
    hyperparameters, environment, evaluation set -- is identical, so the two
    differ only in initialisation.
    """
    arm = study["arms"][kind]
    algo = study["finetune_algo"]
    out = RESULTS / arm / f"seed_{seed}"
    ckpt = checkpoint_path(arm, seed, algo)
    if (out / "final_eval.json").exists() and not force:
        print(f"  [{arm}/seed_{seed}] cached")
        return

    t0 = time.time()
    if kind == "finetune":
        agent = crossover_agent(algo, pretrained, seed)
        write_json(out / "transfer.json", agent.transfer_report)
    else:
        agent = fresh_agent(algo, "original", seed)

    recorder = CurveRecorder(
        agent, "original", HP.CORRUPTION, HP.CURVE_EVERY, phase=kind,
        episodes=HP.CURVE_EPISODES,
    )
    if algo == "ppo":
        agent.learn(HP.ADAPT_STEPS, callback=PPOCurveCallback(recorder, agent))
    else:
        # The PPO callback records its own step-0 point from inside SB3; the
        # GFlowNet's plain-function callback does not, so take it here. Every
        # curve in both studies therefore starts at a measured step-0 value --
        # for a fine-tuning arm that is exactly the zero-shot number.
        recorder.maybe_record(agent, 0, force=True)
        agent.learn(HP.ADAPT_STEPS, callback=gflownet_curve_callback(recorder))
    recorder.maybe_record(agent, HP.ADAPT_STEPS, force=True)

    agent.save(ckpt)
    final = evaluate_policy(agent, "original", corruption=HP.CORRUPTION,
                            n_episodes=HP.EVAL_EPISODES, seed=HP.EVAL_SEED_BASE)
    save_eval(out, "final_eval", final, variant="original")
    write_csv(out / "episodes.csv", final.to_records())
    write_csv(out / "curve.csv", recorder.rows)
    dump_train_log(agent, out)
    write_json(out / "config.json", dict(
        study=study["name"], arm=arm, phase=kind, algo=algo,
        train_variant="original", budget_env_steps=HP.ADAPT_STEPS,
        seed=seed, consumed_env_steps=int(agent.num_timesteps),
        initialised_from=str(pretrained) if kind == "finetune" else "random",
        parameter_fingerprint=agent.parameter_fingerprint(),
        wall_clock_seconds=round(time.time() - t0, 1),
        hyperparameters=HP.resolved(), versions=package_versions(),
    ))
    print(f"  [{arm}/seed_{seed}] success={final.success_rate:.3f} "
          f"reward={final.mean_reward:.1f} ({time.time() - t0:.0f}s)")
    agent.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, choices=sorted(STUDIES))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(HP.SEEDS))
    ap.add_argument("--force", action="store_true", help="recompute cached runs")
    args = ap.parse_args()

    study = STUDIES[args.study]
    print(f"Study {study['name']}: {study['title']}")
    print(f"  corruption      : {HP.CORRUPTION}")
    print(f"  budgets         : {HP.PRETRAIN_STEPS:,} pretrain + "
          f"{HP.ADAPT_STEPS:,} adapt (environment steps)")
    print(f"  seeds           : {args.seeds}")
    print(f"  hparams sha256  : {HP.hyperparameter_fingerprint()[:16]}")

    t0 = time.time()
    for seed in args.seeds:
        print(f"seed {seed}")
        ckpt = run_pretrain(study, seed, args.force)
        run_phase2(study, seed, "finetune", ckpt, args.force)
        run_phase2(study, seed, "scratch", None, args.force)

    write_json(RESULTS / "study.json", dict(
        study, hyperparameters=HP.resolved(), versions=package_versions(),
        seeds_run=args.seeds, wall_clock_seconds=round(time.time() - t0, 1),
    ))
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS}")
    print("now run:  python3 analyze.py")


if __name__ == "__main__":
    main()
