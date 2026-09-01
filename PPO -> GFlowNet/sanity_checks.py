"""Assertions that must hold for either study's results to mean anything.

    python3 sanity_checks.py

Each check targets a specific way the study could be silently wrong. Byte-identical
in both study directories. Exits non-zero on the first failure.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

import hyperparameters as HP
from algorithms.gflownet.model import GFlowNetPolicy
from algorithms.ppo.agent import PPOAgent
from algorithms.transfer import (
    PF_TO_PPO,
    PPO_TO_PF,
    transfer_gflownet_policy_to_ppo,
    transfer_ppo_policy_to_gflownet,
)
from environments.factory import make_env
from environments.shortest_path import get_geodesic

ROOT = Path(__file__).resolve().parent
CHECKS: list[tuple[str, callable]] = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------


@check("t/T convention is identical for both algorithms")
def _time_feature():
    """The GFlowNet appends ``t/T`` itself; PPO gets it from ``TimeFeature``.

    If these two disagree by even one decision, the copied first layer is fed a
    different quantity in its last column and every cross-algorithm transfer in
    both studies is meaningless. This is the load-bearing assumption of the
    whole design, so it is asserted over a *full* episode rather than at reset.
    """
    e_tf = make_env("original", corruption=HP.CORRUPTION, discrete=True,
                    action_repeat=HP.ACTION_REPEAT, time_feature=True)
    e_no = make_env("original", corruption=HP.CORRUPTION, discrete=True,
                    action_repeat=HP.ACTION_REPEAT, time_feature=False)
    try:
        assert e_tf.observation_space.shape == (HP.OBS_DIM + 1,)
        o1, _ = e_tf.reset(seed=7)
        o2, _ = e_no.reset(seed=7)
        D = HP.DECISIONS_PER_EPISODE
        worst = 0.0
        for d in range(D):
            gfn_input = np.concatenate([o2, [d / D]])
            worst = max(worst, float(np.abs(gfn_input - o1).max()))
            a = (d * 7) % HP.N_ACTIONS
            o1, *_ = e_tf.step(a)
            o2, *_ = e_no.step(a)
        assert worst == 0.0, f"time-feature mismatch, max diff {worst}"
        return f"bit-identical over all {D} decisions"
    finally:
        e_tf.close()
        e_no.close()


@check("weight transfer is exact in both directions")
def _transfer_both_ways():
    """A round trip PPO -> GFlowNet -> PPO must be the identity.

    Checks the two mappings are genuine inverses, so neither study is quietly
    permuting a layer relative to the other.
    """
    assert PPO_TO_PF == {v: k for k, v in PF_TO_PPO.items()}, "mappings not inverse"

    agent = PPOAgent.fresh(
        "original", corruption=HP.CORRUPTION, discrete=HP.DISCRETE, seed=3,
        n_envs=1, action_repeat=HP.ACTION_REPEAT, time_feature=HP.TIME_FEATURE,
        policy_hidden=(HP.HIDDEN, HP.HIDDEN),
        hparams=dict(HP.PPO_HPARAMS, policy_kwargs=HP.ppo_policy_kwargs()),
    )
    try:
        obs_space = agent.model.observation_space
        net = GFlowNetPolicy(obs_dim=HP.OBS_DIM, n_actions=HP.N_ACTIONS, hidden=HP.HIDDEN)

        # Study A's direction, strict=True raises unless logits match exactly.
        fwd = transfer_ppo_policy_to_gflownet(net, agent.model.policy,
                                              obs_space=obs_space, seed=1)
        assert fwd["equivalent"], fwd

        # Study B's direction, back into a *different* fresh PPO.
        other = PPOAgent.fresh(
            "original", corruption=HP.CORRUPTION, discrete=HP.DISCRETE, seed=99,
            n_envs=1, action_repeat=HP.ACTION_REPEAT, time_feature=HP.TIME_FEATURE,
            policy_hidden=(HP.HIDDEN, HP.HIDDEN),
            hparams=dict(HP.PPO_HPARAMS, policy_kwargs=HP.ppo_policy_kwargs()),
        )
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                ck = Path(td) / "gfn.pt"
                torch.save({"state_dict": net.state_dict(),
                            "config": dict(hidden=HP.HIDDEN)}, ck)
                back = transfer_gflownet_policy_to_ppo(
                    other.model.policy, ck, obs_space=obs_space, seed=1)
            assert back["equivalent"], back

            src = dict(agent.model.policy.named_parameters())
            dst = dict(other.model.policy.named_parameters())
            worst = max(float((src[k] - dst[k]).abs().max())
                        for k in PF_TO_PPO.values())
            assert worst == 0.0, f"round trip not identity, max diff {worst}"
            return f"6 params each way; round-trip max diff {worst}"
        finally:
            other.close()
    finally:
        agent.close()


@check("the corruption preserves difficulty")
def _difficulty_preserved():
    """The corrupted MDP must be *exactly as hard* as the original.

    Otherwise a measured transfer gap is just a difficulty confound -- the
    failure mode that makes most "our corruption hurt performance" results
    uninterpretable. Three things are asserted, not merely argued:

    1. The corruption leaves the **maze** untouched, so no geometry changed.
    2. Every start->goal **geodesic** is bit-identical between the variants,
       measured on the real (start, goal) pairs the evaluator actually draws
       rather than on uniform points (most of which are inside walls).
    3. The action map is a **signed permutation**, hence an isometry and a
       bijection of the action box: it permutes the 9 bang-bang actions among
       themselves and preserves every action's norm. The corrupted MDP is
       therefore isomorphic to the original up to relabelling coordinates.
    """
    from environments.corruptions import get
    from environments.factory import layout_for
    from environments.wrappers import BANG_BANG_ACTIONS

    lay_o = layout_for("original", HP.CORRUPTION)
    lay_c = layout_for("corrupted", HP.CORRUPTION)
    assert lay_o == lay_c, "the corruption altered the maze layout"

    # Real reachable instances, not uniform points in the bounding box.
    env = make_env("original", corruption=HP.CORRUPTION, discrete=True,
                   action_repeat=HP.ACTION_REPEAT, time_feature=True)
    try:
        pairs = []
        for i in range(25):
            o, _ = env.reset(seed=HP.EVAL_SEED_BASE + i)
            pairs.append((np.asarray(o[:2], float), np.asarray(o[4:6], float)))
    finally:
        env.close()

    a, b = get_geodesic(lay_o), get_geodesic(lay_c)
    worst = max(abs(a.distance_to_goal_region(s, g) - b.distance_to_goal_region(s, g))
                for s, g in pairs)
    assert worst == 0.0, f"geodesics differ by {worst}; difficulty not preserved"

    m = get(HP.CORRUPTION).action_matrix
    assert m is not None, f"{HP.CORRUPTION} is not an actuation corruption"
    mapped = BANG_BANG_ACTIONS @ np.asarray(m, dtype=np.float32).T
    orig = {tuple(np.round(v, 6)) for v in BANG_BANG_ACTIONS}
    assert {tuple(np.round(v, 6)) for v in mapped} == orig, "not a bijection of the action set"
    norm_err = float(np.abs(np.linalg.norm(mapped, axis=1)
                            - np.linalg.norm(BANG_BANG_ACTIONS, axis=1)).max())
    assert norm_err == 0.0, f"not an isometry, norm error {norm_err}"
    return (f"layouts identical; {len(pairs)} geodesics bit-identical; "
            f"action map is a bijective isometry of the {len(BANG_BANG_ACTIONS)}-action set")


@check("both studies share one hyperparameter file")
def _same_hparams():
    """The user's requirement: the two researches use the same hyperparameters."""
    mine = hashlib.sha256((ROOT / "hyperparameters.py").read_bytes()).hexdigest()
    assert mine == HP.hyperparameter_fingerprint()
    sibling = None
    for cand in (ROOT.parent / "RL+PPO", ROOT.parent / "RL+gflowne"):
        if cand.resolve() != ROOT and (cand / "hyperparameters.py").exists():
            sibling = cand
    if sibling is None:
        return f"{mine[:16]} (sibling study not found; skipped comparison)"
    theirs = hashlib.sha256((sibling / "hyperparameters.py").read_bytes()).hexdigest()
    assert mine == theirs, f"hyperparameters differ: {mine} vs {theirs}"
    return f"{mine[:16]} == {sibling.name}'s"


@check("evaluation is on a fixed, shared instance set")
def _shared_eval_instances():
    """Every model in both studies must face the identical 100 instances, or
    contrasts are unpaired and evaluation variance leaks into the comparison."""
    env = make_env("original", corruption=HP.CORRUPTION, discrete=True,
                   action_repeat=HP.ACTION_REPEAT, time_feature=True)
    try:
        first = [env.reset(seed=HP.EVAL_SEED_BASE + i)[0][:6].copy() for i in range(10)]
        second = [env.reset(seed=HP.EVAL_SEED_BASE + i)[0][:6].copy() for i in range(10)]
        worst = max(float(np.abs(a - b).max()) for a, b in zip(first, second))
        assert worst == 0.0, f"eval instances not reproducible, diff {worst}"
        starts = {tuple(np.round(o[:2], 6)) for o in first}
        goals = {tuple(np.round(o[4:6], 6)) for o in first}
        assert len(starts) > 1 and len(goals) > 1, "eval set is degenerate"
        return f"reproducible; {len(starts)} distinct starts / {len(goals)} goals in first 10"
    finally:
        env.close()


@check("budget accounting is in environment steps")
def _budget_currency():
    """Under ``action_repeat=5`` SB3 counts decisions, not environment steps.

    If the conversion were wrong, a fine-tuning arm and its control would not
    receive the same interaction and the primary contrast would be confounded.
    """
    agent = PPOAgent.fresh(
        "original", corruption=HP.CORRUPTION, discrete=HP.DISCRETE, seed=5,
        n_envs=HP.PPO_N_ENVS, action_repeat=HP.ACTION_REPEAT,
        time_feature=HP.TIME_FEATURE, policy_hidden=(HP.HIDDEN, HP.HIDDEN),
        hparams=dict(HP.PPO_HPARAMS, policy_kwargs=HP.ppo_policy_kwargs()),
    )
    try:
        budget = 8192
        agent.learn(budget)
        got = int(agent.num_timesteps)
        # SB3 rounds up to a whole rollout; allow one buffer of slack.
        buf = HP.PPO_HPARAMS["n_steps"] * HP.PPO_N_ENVS * HP.ACTION_REPEAT
        assert budget <= got <= budget + buf, f"consumed {got}, asked {budget}"
        return f"asked {budget}, consumed {got} env steps"
    finally:
        agent.close()


def main() -> int:
    print(f"sanity checks for {ROOT.name}\n")
    failed = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {name}\n          {detail}")
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
