r"""The corruption family, and the rule used to pick the primary corruption.

Motivation
----------
The brief (§3) requires a corruption that is (a) defined programmatically,
(b) meaningful, and (c) *verified* to cause measurable transfer degradation.
Our first candidate -- reflecting the maze layout (``mirror_x``) -- satisfies
(a) and (b) but **fails (c)**.  Measured (``results/corruption_sweep/``), a PPO
policy trained on the mirrored maze scores **0.845** success back on the
original, against a **0.870** reference.  The corruption cost it 0.025.

This turned out to hold for *every* layout corruption in UMaze's D4 orbit
(``mirror_x`` 0.845, ``transpose`` 0.780, ``anti_transpose``), and it is a
result worth reporting rather than a nuisance: because the observation
contains both absolute position and the goal, PPO learns a *reactive*
"steer toward the goal, slide along walls" controller that is largely
indifferent to where the corridor is, and a 5x5 maze is small enough for that
controller to solve every layout in the orbit.  **Rearranging this maze does
not create a transfer problem at all.**

Actuation faults do: rewiring the actuators drops zero-shot success to
0.600 (``rotate_actions_90``), 0.065 (``negate_x``), 0.025 (``swap_axes``) and
0.000 (``negate_both``), while native performance stays at reference level in
every case -- so the difficulty is untouched and the gap is pure transfer.

Rather than quietly swap in whichever corruption produced a big number -- the
exact failure mode §3 and §17 warn against -- we define a **family** of
corruptions spanning two mechanisms, measure the zero-shot degradation of
every member (``scripts/corruption_sweep.py``), and select the primary
corruption by a rule fixed *in advance*:

    **Selection rule.**  Among corruptions that are provably
    difficulty-preserving (below), take the one whose zero-shot success rate on
    the original environment is lowest, provided it falls below the
    measured random-action floor of 0.26 (:data:`RANDOM_SUCCESS_FLOOR`).
    Ties break toward the *milder* mechanism.

The rule depends only on **zero-shot degradation**, which is a property of the
corruption alone.  It never looks at the from-scratch-vs-fine-tuning contrast,
so it cannot bias the study's actual research question.  The full sweep is
reported, not just the winner, and the four primary conditions are additionally
re-run at three severity levels (§ "severity ladder" in ``METHODOLOGY.md``) so
that "the answer depends on corruption severity" remains a detectable outcome.

Two mechanisms
--------------
**Layout corruptions** rearrange the maze via ``maze_map``.  **Actuation
corruptions** rewire the actuators, :math:`u \mapsto M u`, leaving the maze
untouched -- "the motors were reconnected wrongly".  Both keep the task
("drive the ball from start to goal"), the observation space, the action
space, the sparse reward and the 300-step horizon exactly as they are.

Why every member is provably difficulty-preserving
--------------------------------------------------
We only admit transforms drawn from :math:`D_4`, the symmetry group of the
square, acting either on the layout or on the actuators.

* A **layout** transform :math:`g \in D_4` maps the maze to a congruent maze.
  Since :math:`g` is an isometry of the plane, it preserves all geodesic
  distances and maps the free-cell set bijectively, so the start/goal
  distribution and every shortest path carry over exactly.
* An **actuation** transform :math:`u \mapsto Mu` with :math:`M` a signed
  permutation is an isometry of the action box :math:`[-1,1]^2`.  Relabelling
  the world coordinates by :math:`M^{-1}` turns the corrupted MDP into the
  *uncorrupted* dynamics on the layout :math:`M^{-1}` applied to the maze --
  which is congruent to the original.  So the corrupted MDP is isomorphic to
  the original up to a fixed permutation/sign of observation and action
  coordinates.

In both cases the corrupted MDP is isomorphic to the original, hence **exactly
as hard**.  Any transfer gap is therefore a genuine transfer effect and not a
difficulty confound -- and this is asserted numerically, not just argued, by
``sanity_checks.py`` (geodesic distributions must match bit-for-bit, and
natively-trained performance must match within seed noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np

from .layouts import UMAZE

# ---------------------------------------------------------------------------
# Layout transforms (elements of D4 acting on the maze matrix)
# ---------------------------------------------------------------------------


def identity_layout(m):
    return [list(r) for r in m]


def mirror_x(m):
    r""":math:`(x, y) \mapsto (-x, y)`. Reflect about the vertical axis."""
    return [list(r[::-1]) for r in m]


def mirror_y(m):
    r""":math:`(x, y) \mapsto (x, -y)`. Reflect about the horizontal axis."""
    return [list(r) for r in m[::-1]]


def rotate_180(m):
    r""":math:`(x, y) \mapsto (-x, -y)`."""
    return [list(r[::-1]) for r in m[::-1]]


def transpose(m):
    r""":math:`(x, y) \mapsto (y, x)`. The U now opens downward, not rightward."""
    return [list(col) for col in zip(*m)]


def anti_transpose(m):
    r""":math:`(x, y) \mapsto (-y, -x)`."""
    return [list(col) for col in zip(*[list(r[::-1]) for r in m[::-1]])]


# ---------------------------------------------------------------------------
# Actuation transforms (signed permutation matrices -- isometries of the box)
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.float32)
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
NEG_X = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
ROT90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
NEG_BOTH = -I2


class LinearActionCorruption(gym.ActionWrapper):
    r"""Rewire the actuators: :math:`u \mapsto M u`.

    ``M`` must be a signed permutation matrix so that the transform is a
    bijection of :math:`[-1, 1]^2` onto itself; the advertised ``action_space``
    is therefore unchanged and every algorithm remains free to emit any legal
    action.  The wrapper sits *inside* any discretisation wrapper so that the
    discrete action table is corrupted consistently with the continuous one.
    """

    def __init__(self, env: gym.Env, matrix: np.ndarray):
        super().__init__(env)
        matrix = np.asarray(matrix, dtype=np.float32)
        nz = np.count_nonzero(matrix, axis=0)
        if matrix.shape != (2, 2) or not np.all(nz == 1) or not np.all(
            np.isin(matrix[matrix != 0], (-1.0, 1.0))
        ):
            raise ValueError("matrix must be a 2x2 signed permutation matrix")
        self.matrix = matrix

    def action(self, action):
        return (self.matrix @ np.asarray(action, dtype=np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corruption:
    name: str
    mechanism: str  # "layout" | "actuation" | "none"
    layout_fn: Callable = identity_layout
    action_matrix: np.ndarray | None = None
    rationale: str = ""

    def layout(self, base=UMAZE) -> list[list[int]]:
        return self.layout_fn(base)


CORRUPTION_FAMILY: dict[str, Corruption] = {
    "none": Corruption(
        "none", "none",
        rationale="Control: the original environment itself.",
    ),
    # -- layout mechanism ------------------------------------------------
    "mirror_x": Corruption(
        "mirror_x", "layout", layout_fn=mirror_x,
        rationale="Corridor moves from x=+1 to x=-1; flips exactly 2 cells. "
                  "Mildest layout change that still moves the passage.",
    ),
    # NOTE: rotate_180 and mirror_y are deliberately absent from SWEEP_ORDER.
    # UMaze is already symmetric about its horizontal axis (row 0 == row 4,
    # row 1 == row 3), so mirror_y is in its stabiliser and
    # rotate_180(UMAZE) == mirror_x(UMAZE) *exactly*. The first sweep
    # confirmed this empirically -- the two produced bit-identical numbers on
    # both seeds. The D4 orbit of UMaze therefore has only 4 members, and the
    # three distinct non-identity layouts are mirror_x, transpose and
    # anti_transpose. They are kept here only so the equivalence is testable.
    "rotate_180": Corruption(
        "rotate_180", "layout", layout_fn=rotate_180,
        rationale="Redundant on UMaze: identical to mirror_x because UMaze is "
                  "symmetric about its horizontal axis. Retained so the "
                  "sanity checks can assert the equivalence.",
    ),
    "anti_transpose": Corruption(
        "anti_transpose", "layout", layout_fn=anti_transpose,
        rationale="The U opens upward. Completes the set of 3 distinct "
                  "non-identity layouts in UMaze's D4 orbit.",
    ),
    "transpose": Corruption(
        "transpose", "layout", layout_fn=transpose,
        rationale="The U opens downward instead of rightward: the barrier "
                  "turns from horizontal to vertical, so the required detour "
                  "axis changes rather than merely its sign.",
    ),
    # -- actuation mechanism ---------------------------------------------
    "negate_x": Corruption(
        "negate_x", "actuation", action_matrix=NEG_X,
        rationale="The x actuator is wired backwards. Pure sign error, the "
                  "mildest actuation fault.",
    ),
    "swap_axes": Corruption(
        "swap_axes", "actuation", action_matrix=SWAP,
        rationale="The two actuators are exchanged: commanding +x pushes +y. "
                  "Equivalent to the original dynamics on the transposed maze.",
    ),
    "rotate_actions_90": Corruption(
        "rotate_actions_90", "actuation", action_matrix=ROT90,
        rationale="Every commanded force is rotated a quarter turn.",
    ),
    "negate_both": Corruption(
        "negate_both", "actuation", action_matrix=NEG_BOTH,
        rationale="Both actuators reversed: every command pushes the "
                  "opposite way. Maximal actuation fault in D4.",
    ),
}

#: Members considered by the pre-registered selection rule. Covers all three
#: distinct non-identity layouts in UMaze's D4 orbit and four actuation faults.
SWEEP_ORDER = [
    "mirror_x", "transpose", "anti_transpose",
    "rotate_actions_90", "negate_x", "swap_axes", "negate_both",
]

#: Floor for "the corruption really broke something": the success rate of a
#: uniform-random policy on the standard 100-instance evaluation set, measured
#: by ``evaluation.evaluator.evaluate_random_baseline`` (0.26). A corrupted
#: policy that scores below this on the original environment has been rendered
#: worse than useless there, which is the degradation we want to study.
RANDOM_SUCCESS_FLOOR = 0.26

#: Primary corruption, chosen by the selection rule above from the measured
#: sweep in ``results/corruption_sweep/``. ``negate_both`` has the lowest
#: zero-shot success (0.000 vs a 0.870 reference) among the corruptions that
#: preserved native difficulty. See ``METHODOLOGY.md`` §2.
DEFAULT_CORRUPTION = "negate_both"

#: Severity ladder for the secondary "does the answer depend on corruption
#: severity?" analysis (§17). Ordered by *measured* zero-shot degradation and
#: held to a single mechanism (actuation) so severity is the only thing that
#: varies along it. Measured zero-shot success: 0.600 / 0.025 / 0.000 against a
#: 0.870 reference.
SEVERITY_LADDER = ("rotate_actions_90", "swap_axes", "negate_both")

#: The layout corruptions, none of which degraded transfer at all. Reported as
#: a negative result rather than discarded.
LAYOUT_NULL_RESULTS = ("mirror_x", "transpose", "anti_transpose")


def get(name: str) -> Corruption:
    if name not in CORRUPTION_FAMILY:
        raise KeyError(f"unknown corruption {name!r}; have {sorted(CORRUPTION_FAMILY)}")
    return CORRUPTION_FAMILY[name]
