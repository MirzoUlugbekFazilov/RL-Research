"""Assertions specific to this study's two corruptions.

    python3 verify_corruptions.py

``sanity_checks.py`` carries the original study's checks, and several of them
are ``negate_both``-specific by construction -- they assert that the maze
layout is *unchanged*, that the two geodesic distributions match bit-for-bit,
and that the corruption has a signed-permutation ``action_matrix``.  Neither
corruption studied here satisfies those: ``large_multigoal`` deliberately
replaces the maze, and ``action_gain_jitter`` is stochastic and has no matrix.
Running those checks against these corruptions would be meaningless, so this
module states and tests the properties that *are* load-bearing here.

The one that matters most is :func:`check_action_gain_is_two_sided`.  The
natural implementation of ``u -> g*u`` -- an ``ActionWrapper`` returning
``g * action`` -- is silently half-dead on this study's ``Discrete(9)``
bang-bang interface, because ``PointEnv.step`` clips actions to ``[-1, 1]`` and
every bang-bang action already sits on that boundary.  That failure is
invisible in the results (the runs complete, the numbers look plausible) and
would turn "gain jitter" into "gain *reduction* half the time and nothing the
other half".  It is therefore asserted, not assumed.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from environments.corruptions import (
    STOCHASTIC_CORRUPTIONS,
    STRUCTURAL_CORRUPTIONS,
    get,
)
from environments.factory import layout_for, make_env
from environments.layouts import LARGE_MULTIGOAL, UMAZE, free_cells, wall_mask
from environments.shortest_path import get_geodesic

CORRUPTIONS = ("large_multigoal", "action_gain_jitter")

CHECKS: list[tuple[str, callable]] = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _env(variant, corruption, seed=1):
    return make_env(variant, corruption=corruption, discrete=True,
                    action_repeat=5, time_feature=True, seed=seed)


# ---------------------------------------------------------------------------
# Shared interface: the transfer is only meaningful if it survives both
# ---------------------------------------------------------------------------

@check("interface is identical across variants and corruptions")
def check_interface():
    """The weight copy requires one function class everywhere.

    If a corruption changed the observation dimension or the action count, the
    GFlowNet -> PPO parameter copy would be shape-invalid (or, worse, silently
    valid but semantically wrong).  Both corruptions must leave ``Box(7,)`` and
    ``Discrete(9)`` exactly as they are.
    """
    ref = None
    for corr in CORRUPTIONS:
        for variant in ("original", "corrupted"):
            e = _env(variant, corr)
            got = (e.observation_space.shape, int(e.action_space.n))
            e.close()
            assert got == ((7,), 9), f"{corr}/{variant} has interface {got}"
            ref = ref or got
    return "Box(7,) / Discrete(9) in all 4 conditions"


@check("the ORIGINAL environment is untouched by either corruption")
def check_original_is_shared():
    """``variant='original'`` must not depend on which corruption is named.

    Both fine-tuning and both from-scratch arms train on the original
    environment.  If a corruption leaked into it, the two corruption studies
    would not share a control and could not be compared to each other or to the
    ``negate_both`` study.  ``make_env`` gates every corruption branch on
    ``variant == 'corrupted'``; this asserts the consequence.
    """
    assert layout_for("original", "large_multigoal") == [list(r) for r in UMAZE]
    assert layout_for("original", "action_gain_jitter") == [list(r) for r in UMAZE]

    traj = {}
    for corr in CORRUPTIONS + ("negate_both",):
        e = _env("original", corr, seed=3)
        obs, _ = e.reset(seed=1_000_000)
        rows = [obs.copy()]
        for a in (0, 4, 8, 2, 6, 1, 7):
            obs, r, te, tr, info = e.step(a)
            rows.append(obs.copy())
        e.close()
        traj[corr] = np.array(rows)

    base = traj["negate_both"]
    for corr in CORRUPTIONS:
        assert np.array_equal(traj[corr], base), (
            f"the original environment differs under {corr}: a corruption leaked "
            f"into the uncorrupted variant"
        )
    return "identical trajectories under all 3 corruption names"


# ---------------------------------------------------------------------------
# large_multigoal
# ---------------------------------------------------------------------------

@check("large_multigoal uses the upstream benchmark map verbatim")
def check_large_map_is_upstream():
    """The corrupted layout must be gymnasium-robotics' own constant.

    A hand-drawn maze would make "the corruption was chosen after seeing a
    result" unfalsifiable.  Equality with a named upstream constant cannot have
    been tuned by us.
    """
    from gymnasium_robotics.envs.maze.maps import LARGE_MAZE_DIVERSE_G

    lay = layout_for("corrupted", "large_multigoal")
    assert lay == [list(r) for r in LARGE_MAZE_DIVERSE_G], "map is not the upstream one"
    assert lay == [list(r) for r in LARGE_MULTIGOAL]
    assert get("large_multigoal").difficulty_preserving is False
    assert "large_multigoal" in STRUCTURAL_CORRUPTIONS
    return f"{len(lay)}x{len(lay[0])}, equals LARGE_MAZE_DIVERSE_G"


@check("large_multigoal really has several goals")
def check_multiple_goals():
    """Several ``'g'`` cells, and the goal actually varies over resets.

    "Adding a few other goals" is the substance of this corruption, so it is
    measured rather than assumed: the map must declare >1 goal cell and the
    environment must actually draw from them.
    """
    lay = layout_for("corrupted", "large_multigoal")
    g_cells = [(r, c) for r, row in enumerate(lay)
               for c, v in enumerate(row) if v == "g"]
    r_cells = [(r, c) for r, row in enumerate(lay)
               for c, v in enumerate(row) if v == "r"]
    assert len(g_cells) >= 2, f"only {len(g_cells)} goal cells"

    e = _env("corrupted", "large_multigoal")
    goals = set()
    for i in range(60):
        obs, _ = e.reset(seed=1_000_000 + i)
        goals.add(tuple(np.round(obs[4:6], 0)))
    e.close()
    assert len(goals) >= len(g_cells) - 1, (
        f"only {len(goals)} distinct goal regions over 60 resets, expected ~{len(g_cells)}"
    )

    umaze_free = len(free_cells(UMAZE))
    large_free = len(free_cells(lay))
    assert large_free > umaze_free
    return (f"{len(g_cells)} goal cells, {len(r_cells)} reset cell(s), "
            f"{len(goals)} distinct goals over 60 resets; "
            f"free cells {umaze_free} -> {large_free}")


@check("goal and reset markers are free space in the geodesic")
def check_markers_are_not_walls():
    """``'g'`` / ``'r'`` must not be mistaken for walls.

    ``MazeGeodesic`` treats a cell as a wall iff it equals ``1``.  Before
    :func:`environments.layouts.wall_mask` existed, ``get_geodesic`` cast every
    cell with ``int(v)``, which raises on ``'g'``; a *silent* variant of the
    same bug (treating a marker as an obstacle) would inflate L* and flatter
    every agent's path efficiency on this layout.
    """
    lay = layout_for("corrupted", "large_multigoal")
    mask = wall_mask(lay)
    n_wall_map = sum(v == 1 for row in lay for v in row)
    n_wall_mask = sum(row.count(1) for row in mask)
    assert n_wall_map == n_wall_mask, "wall_mask changed the wall count"

    geo = get_geodesic(lay)
    # every declared goal cell centre must be reachable free space
    from environments.layouts import cell_to_xy
    for r, row in enumerate(lay):
        for c, v in enumerate(row):
            if v in ("g", "r", "c"):
                p = cell_to_xy(r, c, lay)
                assert geo.clearance(p[None])[0] > geo.ball_radius, (
                    f"marker cell {(r, c)} was treated as an obstacle"
                )
    d = geo.distance_to_goal_region(np.array([-4.5, 3.0]), np.array([3.5, -3.0]))
    assert np.isfinite(d) and d > 0
    return f"{n_wall_map} walls, all markers free, sample geodesic {d:.2f} m"


# ---------------------------------------------------------------------------
# action_gain_jitter
# ---------------------------------------------------------------------------

@check("action_gain_jitter leaves the maze alone")
def check_gain_layout_untouched():
    lay_o = layout_for("original", "action_gain_jitter")
    lay_c = layout_for("corrupted", "action_gain_jitter")
    assert lay_o == lay_c, "the gain corruption altered the maze"
    assert get("action_gain_jitter").gain_range == (0.9, 1.1)
    assert "action_gain_jitter" in STOCHASTIC_CORRUPTIONS
    return "layout identical, g ~ U(0.9, 1.1)"


@check("gain is drawn per episode, in range, and held fixed within it")
def check_gain_sampling():
    e = _env("corrupted", "action_gain_jitter")
    gains, within_ok = [], True
    for i in range(200):
        obs, info = e.reset(seed=1_000_000 + i)
        g = info["action_gain"]
        gains.append(g)
        seen = {g}
        for a in (4, 0, 8):
            obs, r, te, tr, info = e.step(a)
            seen.add(info["action_gain"])
        within_ok &= len(seen) == 1
    e.close()
    gains = np.array(gains)
    assert within_ok, "the gain changed during an episode"
    assert gains.min() >= 0.9 and gains.max() <= 1.1, "gain outside U(0.9, 1.1)"
    assert len(set(gains)) > 150, "the gain is not actually being resampled"
    assert abs(gains.mean() - 1.0) < 0.01, f"mean gain {gains.mean():.4f} is off-centre"
    return (f"n=200 draws in [{gains.min():.4f}, {gains.max():.4f}], "
            f"mean {gains.mean():.4f}, constant within every episode")


@check("gain is reproducible from the evaluation seed")
def check_gain_reproducible():
    """Every arm must face the identical fault on evaluation instance ``i``.

    The 100 shared evaluation instances are what make the contrasts paired; if
    the gain were drawn from an unseeded stream, arms would face different
    faults and that pairing would be lost.
    """
    a, b = _env("corrupted", "action_gain_jitter", seed=1), \
           _env("corrupted", "action_gain_jitter", seed=999)
    ga = [a.reset(seed=1_000_000 + i)[1]["action_gain"] for i in range(50)]
    gb = [b.reset(seed=1_000_000 + i)[1]["action_gain"] for i in range(50)]
    a.close(); b.close()
    assert ga == gb, "same reset seed gave different gains in two envs"
    assert len(set(ga)) > 40, "the gains are degenerate across instances"
    return "50 eval instances reproduce their gain exactly across two envs"


@check("gain is TWO-SIDED under bang-bang actions (the clipping trap)")
def check_action_gain_is_two_sided():
    r"""Both tails of ``U(0.9, 1.1)`` must actually change the dynamics.

    ``PointEnv.step`` starts with ``np.clip(action, -1, 1)`` and MuJoCo clips
    again against ``ctrlrange="-1 1"``.  Every action in the shared
    ``Discrete(9)`` table lies on that boundary, so an ``ActionWrapper`` that
    returned ``g * action`` would have **no effect whatsoever** for ``g > 1``.
    The realised corruption would be a one-sided weakening, and nothing in the
    results would reveal it.

    This runs the same open-loop action sequence under g = 0.9, 1.0 and 1.1 and
    requires both off-nominal gains to move the ball measurably, by comparable
    amounts.
    """
    from environments.wrappers import RandomActionGain

    def rollout(gain):
        e = make_env("corrupted", corruption="action_gain_jitter", discrete=True,
                     action_repeat=5, time_feature=True, seed=1)
        e.reset(seed=1_000_000)
        w = e
        while not isinstance(w, RandomActionGain):
            w = w.env
        w._apply(gain)                      # pin the gain, bypassing the draw
        for a in (8, 8, 8, 8, 5, 5, 5, 5):
            obs, r, te, tr, info = e.step(a)
            assert abs(info["action_gain"] - gain) < 1e-12, "pinned gain drifted"
        e.close()
        return np.asarray(obs[:4], dtype=float)

    nominal = rollout(1.0)
    lo = float(np.linalg.norm(rollout(0.9) - nominal))
    hi = float(np.linalg.norm(rollout(1.1) - nominal))
    assert lo > 1e-3, f"g=0.9 had no effect (delta {lo:.2e})"
    assert hi > 1e-3, (
        f"g=1.1 had NO EFFECT (delta {hi:.2e}) -- the gain is being clipped away; "
        f"it must be applied to actuator_gear, not to the action array"
    )
    ratio = max(lo, hi) / min(lo, hi)
    assert ratio < 3.0, f"the two tails are wildly asymmetric (ratio {ratio:.2f})"
    return f"|dx| at g=0.9: {lo:.4f}, at g=1.1: {hi:.4f} (ratio {ratio:.2f})"


@check("repeated resets do not compound the gain")
def check_gain_does_not_compound():
    """The gear is rescaled from the *nominal* value every reset.

    Scaling ``actuator_gear`` in place is stateful; multiplying the current
    gear instead of the captured base would drift the plant over thousands of
    training episodes and quietly invalidate the whole run.
    """
    from environments.wrappers import RandomActionGain

    e = _env("corrupted", "action_gain_jitter")
    w = e
    while not isinstance(w, RandomActionGain):
        w = w.env
    base = w._base_gear[:, 0].copy()
    for i in range(300):
        _, info = e.reset(seed=None if i % 3 else 1_000_000 + i)
        ratio = w._model().actuator_gear[:, 0] / base
        assert np.allclose(ratio, info["action_gain"]), (
            f"gear/base = {ratio} but reported gain = {info['action_gain']}"
        )
    e.close()
    return "300 resets, gear always equals base * reported gain"


def main() -> int:
    print("Corruption-specific verification\n" + "=" * 60)
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {name}\n          {detail}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}\n          {exc}")
            traceback.print_exc()
    print("=" * 60)
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
