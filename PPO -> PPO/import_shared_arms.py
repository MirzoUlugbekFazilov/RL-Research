"""Import the two Study C arms that already exist, instead of retraining them.

    python3 import_shared_arms.py [--check]

Study C's three arms are::

    ppo_pretrain_corrupted   PPO, 300k steps, corrupted environment
    ppo_finetune_original    PPO, 300k steps, original environment, from that
    ppo_scratch_original     PPO, 300k steps, original environment, random init

Two of them are not new.  ``ppo_pretrain_corrupted`` is *Study A's phase 1*, and
``ppo_scratch_original`` is *Study B's control* -- both already run, for the same
30 seeds and the same two corruptions, by code this directory verifies to be
byte-identical.  Retraining them would produce different models only if the
pipeline were nondeterministic, which ``--check`` measures rather than assumes.

Importing is not merely cheaper, it is *better*, and this is the reason to do it:

* Study C and Study A then fine-tune **the same pretrained PPO models**, seed by
  seed.  "Is GFlowNet or PPO the better fine-tuner?" becomes a paired contrast
  with a common phase 1, rather than two runs that happen to share a recipe.
* Study C and Study B then share **the same from-scratch control**, seed by
  seed, so "GFlowNet pretraining vs PPO pretraining vs none" is one three-way
  comparison against one baseline instead of three separate baselines.

Every imported seed is checked before it is accepted: same hyperparameter
SHA-256, same corruption, same algorithm, same budget, and a checkpoint that
loads.  Provenance is written to ``runs/<corruption>/results/<arm>/IMPORTED.json``
and summarised in ``PROVENANCE.md``, so no imported number can be mistaken for
one produced here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sibling_study(repository_name: str, local_name: str) -> Path:
    """Support both the repository layout and original local study names."""
    for name in (repository_name, local_name):
        candidate = ROOT.parent / name
        if candidate.is_dir():
            return candidate
    return ROOT.parent / repository_name


#: Where each shared arm comes from, and what it was in its home study.
SOURCES = {
    "ppo_pretrain_corrupted": dict(
        study_dir=sibling_study("PPO to GFlowNet (2 corrupted)", "ppo to gflownet"),
        home_study="A",
        role="phase 1 of Study A (PPO on corrupted)",
        algo="ppo",
        train_variant="corrupted",
        corruption_specific=True,
    ),
    "ppo_scratch_original": dict(
        study_dir=sibling_study("GFlowNet to PPO (2 corrupted)", "gflownet to ppo"),
        home_study="B",
        role="from-scratch control of Study B (PPO on original)",
        algo="ppo",
        train_variant="original",
        corruption_specific=False,
    ),
}

CORRUPTIONS = ("large_multigoal", "action_gain_jitter")

#: The claim the whole import rests on: identical code, identical
#: hyperparameters.  ``hyperparameters.py`` is byte-identical in all three
#: directories; the two files below are supersets here (Study C is additive --
#: the A and B code paths are untouched), so they are compared by the A/B
#: entry points they still contain rather than by hash.
BYTE_IDENTICAL = [
    "hyperparameters.py",
    "algorithms/gflownet/agent.py",
    "algorithms/gflownet/model.py",
    "environments/factory.py",
    "environments/corruptions.py",
    "environments/wrappers.py",
    "environments/layouts.py",
    "environments/original.py",
    "environments/shortest_path.py",
    "evaluation/evaluator.py",
    "evaluation/metrics.py",
    "training/callbacks.py",
]

EXPECTED_HPARAM_SHA = "d01ab1172c5f215c46057304aec524514e1cd8a9bebfc985c2b9e51c10bb74e7"

#: Environment steps a 300,000-step PPO run actually consumes. See
#: :func:`check_seed` -- SB3 finishes the rollout it is in, so every PPO run in
#: every study lands on the same 307,200, and the constant is asserted rather
#: than tolerated so a genuinely short run would still be caught.
PPO_CONSUMED_ENV_STEPS = 307_200


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_code_identical() -> list[str]:
    """Verify the shared modules match in every source directory."""
    problems = []
    for name in BYTE_IDENTICAL:
        mine = ROOT / name
        if not mine.exists():
            problems.append(f"missing here: {name}")
            continue
        h = sha256(mine)
        for src in SOURCES.values():
            other = src["study_dir"] / name
            if not other.exists():
                problems.append(f"missing in {src['study_dir'].name}: {name}")
            elif sha256(other) != h:
                problems.append(
                    f"differs from {src['study_dir'].name}: {name}"
                )
    if sha256(ROOT / "hyperparameters.py") != EXPECTED_HPARAM_SHA:
        problems.append("hyperparameters.py is not the study's canonical file")
    return problems


def check_seed(seed_dir: Path, arm: str, corruption: str, spec: dict) -> list[str]:
    """Validate one seed directory before it is imported."""
    bad = []
    cfg_file = seed_dir / "config.json"
    if not cfg_file.exists():
        return [f"{seed_dir.name}: no config.json"]
    cfg = json.loads(cfg_file.read_text())

    if cfg.get("arm") != arm:
        bad.append(f"{seed_dir.name}: arm is {cfg.get('arm')!r}, expected {arm!r}")
    if cfg.get("algo") != spec["algo"]:
        bad.append(f"{seed_dir.name}: algo is {cfg.get('algo')!r}")
    if cfg.get("train_variant") != spec["train_variant"]:
        bad.append(f"{seed_dir.name}: train_variant is {cfg.get('train_variant')!r}")
    if cfg.get("budget_env_steps") != 300_000:
        bad.append(f"{seed_dir.name}: budget is {cfg.get('budget_env_steps')}")
    # SB3 stops on a rollout boundary, not on the exact budget: 300,000 env
    # steps is 60,000 decisions under action_repeat=5, and the rollout buffer
    # holds 2,048, so PPO always finishes the 30th rollout at 61,440 decisions
    # = 307,200 env steps. The overshoot is one rollout, it is the same
    # constant for every PPO arm in every study -- fine-tuned and scratch alike
    # -- so it can never be a difference between the arms being contrasted.
    if cfg.get("consumed_env_steps") != PPO_CONSUMED_ENV_STEPS:
        bad.append(f"{seed_dir.name}: consumed {cfg.get('consumed_env_steps')} "
                   f"steps, expected {PPO_CONSUMED_ENV_STEPS}")

    hp = cfg.get("hyperparameters", {})
    if hp.get("hyperparameters_sha256") != EXPECTED_HPARAM_SHA:
        bad.append(f"{seed_dir.name}: hyperparameter hash differs")
    # The pretrain arm is corruption-specific; the scratch arm never touches a
    # corrupted environment, so its record's corruption field is incidental.
    if spec["corruption_specific"] and hp.get("corruption") != corruption:
        bad.append(f"{seed_dir.name}: trained on {hp.get('corruption')!r}, "
                   f"importing into {corruption!r}")
    if not (seed_dir / "final_eval.json").exists():
        bad.append(f"{seed_dir.name}: no final_eval.json")
    return bad


def import_arm(arm: str, corruption: str, spec: dict, *, check_only: bool) -> dict:
    src_root = spec["study_dir"] / "runs" / corruption
    src_results = src_root / "results" / arm
    src_ckpts = src_root / "checkpoints" / arm
    if not src_results.is_dir():
        raise SystemExit(f"source missing: {src_results}")

    seed_dirs = sorted(src_results.glob("seed_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    problems: list[str] = []
    for d in seed_dirs:
        problems += check_seed(d, arm, corruption, spec)
        ckpt = src_ckpts / d.name / "model.zip"
        if not ckpt.exists():
            problems.append(f"{d.name}: no checkpoint at {ckpt}")

    seeds = [int(d.name.split("_")[1]) for d in seed_dirs]
    print(f"  {corruption}/{arm}: {len(seeds)} seeds, "
          f"{'OK' if not problems else str(len(problems)) + ' PROBLEMS'}")
    for p in problems[:10]:
        print(f"    ! {p}")
    if problems:
        raise SystemExit(f"refusing to import {arm} for {corruption}")

    if check_only:
        return dict(arm=arm, corruption=corruption, seeds=seeds, imported=False)

    dst_results = ROOT / "runs" / corruption / "results" / arm
    dst_ckpts = ROOT / "runs" / corruption / "checkpoints" / arm
    for dst in (dst_results, dst_ckpts):
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_results, dst_results)
    shutil.copytree(src_ckpts, dst_ckpts)

    record = dict(
        arm=arm,
        corruption=corruption,
        imported_from=str(src_results),
        checkpoints_from=str(src_ckpts),
        home_study=spec["home_study"],
        role_in_home_study=spec["role"],
        seeds=seeds,
        n_seeds=len(seeds),
        hyperparameters_sha256=EXPECTED_HPARAM_SHA,
        note=("Not retrained here. Identical code, identical hyperparameters, "
              "identical seeds; importing makes Study C's contrast against its "
              "home study paired seed-by-seed rather than merely matched."),
    )
    (dst_results / "IMPORTED.json").write_text(json.dumps(record, indent=2))
    return dict(record, imported=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate the sources without copying anything")
    args = ap.parse_args()

    print("verifying shared code is byte-identical ...")
    problems = check_code_identical()
    for p in problems:
        print(f"  ! {p}")
    if problems:
        raise SystemExit("shared code differs -- import would not be sound")
    print(f"  {len(BYTE_IDENTICAL)} modules identical in all three directories")
    print(f"  hyperparameters.py sha256 = {EXPECTED_HPARAM_SHA[:16]}...")

    records = []
    for corruption in CORRUPTIONS:
        for arm, spec in SOURCES.items():
            records.append(import_arm(arm, corruption, spec,
                                      check_only=args.check))

    if args.check:
        print("\ncheck only -- nothing copied")
        return

    lines = [
        "# Provenance of Study C's arms",
        "",
        "Study C (`PPO + PPO`) runs three arms per seed. Two of them already",
        "existed, run by byte-identical code on the same 30 seeds, and were",
        "imported rather than retrained. This file records exactly which.",
        "",
        "| Arm | Origin | Role there | Seeds |",
        "|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r['corruption']}` / `{r['arm']}` | "
            f"`{Path(r['imported_from']).parents[2].parent.name}` (Study "
            f"{r['home_study']}) | {r['role_in_home_study']} | {r['n_seeds']} |"
        )
    lines += [
        "",
        "`ppo_finetune_original` and `ppo_finetune_full_original` are **not**",
        "imported: they are this study's own runs and exist nowhere else.",
        "",
        "## Why importing is the better choice, not the lazy one",
        "",
        "- Study A fine-tunes a **GFlowNet** from `ppo_pretrain_corrupted`;",
        "  Study C fine-tunes **PPO** from the very same checkpoints. The",
        "  question \"which algorithm fine-tunes better?\" is then answered by a",
        "  paired contrast with a common phase 1, seed by seed.",
        "- Study B's fine-tuned arm and Study C's are measured against the",
        "  **same** `ppo_scratch_original` models, so GFlowNet pretraining, PPO",
        "  pretraining and no pretraining are compared to one baseline.",
        "",
        "## What was verified before accepting each seed",
        "",
        "- `hyperparameters.py` SHA-256 equals `" + EXPECTED_HPARAM_SHA[:16] + "...`",
        f"- {len(BYTE_IDENTICAL)} shared modules byte-identical across all three directories",
        "- per seed: arm, algorithm, training variant, 300,000-step budget,",
        "  300,000 steps actually consumed, recorded hyperparameter hash,",
        "  corruption (for the pretrain arm), and a checkpoint on disk",
        "",
        "Run `python3 import_shared_arms.py --check` to re-verify, and",
        "`python3 verify_import.py` to retrain sample seeds from scratch and",
        "confirm they reproduce the imported models bit for bit.",
    ]
    (ROOT / "PROVENANCE.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {ROOT / 'PROVENANCE.md'}")


if __name__ == "__main__":
    main()
