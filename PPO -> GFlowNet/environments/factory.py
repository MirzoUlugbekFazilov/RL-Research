"""One entry point for every environment used in the study.

Keeping construction in a single function is what makes "same task, different
condition" auditable: the only things a condition may vary are ``variant``
(original vs corrupted), which ``corruption`` is in force, and whether actions
are ``discrete``.  Nothing else about the environment is ever touched.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics

from .corruptions import DEFAULT_CORRUPTION, LinearActionCorruption, get
from .layouts import UMAZE
from .original import ENV_ID, MAX_EPISODE_STEPS
from .wrappers import (
    ActionRepeat,
    BangBangDiscreteActions,
    GoalFlattenObservation,
    TimeFeature,
)

gym.register_envs(gymnasium_robotics)

VARIANTS = ("original", "corrupted")


def layout_for(variant: str, corruption: str = DEFAULT_CORRUPTION) -> list[list[int]]:
    """The ``maze_map`` a given condition actually runs on."""
    if variant == "original":
        return [row[:] for row in UMAZE]
    if variant == "corrupted":
        return get(corruption).layout(UMAZE)
    raise KeyError(f"unknown variant {variant!r}; have {VARIANTS}")


def make_env(
    variant: str = "original",
    *,
    corruption: str = DEFAULT_CORRUPTION,
    discrete: bool = False,
    action_repeat: int = 1,
    time_feature: bool = False,
    render_mode: str | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Build a wrapped PointMaze env.

    Wrapper order (innermost first) is fixed and load-bearing:

    1. ``gym.make(PointMaze_UMaze-v3, maze_map=...)`` -- layout corruption, if any.
    2. :class:`LinearActionCorruption` -- actuation corruption, if any.
    3. :class:`BangBangDiscreteActions` -- optional ``Discrete(9)`` action space.
    4. :class:`GoalFlattenObservation` -- ``Dict`` -> ``Box(6,)``.
    5. :class:`ActionRepeat` -- optional ``k`` env steps per decision.
    6. :class:`TimeFeature` -- optional ``t/D`` appended -> ``Box(7,)``.

    Putting the actuation corruption *inside* the discretiser means the
    discrete action table is corrupted exactly as the continuous one is, so
    PPO-continuous, PPO-discrete and the GFlowNet all experience the identical
    fault.  ``ActionRepeat`` sits outermost so that the 300-step ``TimeLimit``
    it wraps still measures true environment steps.

    ``TimeFeature`` sits outside ``ActionRepeat`` so its counter advances once
    per *decision*, reproducing exactly the ``t/T`` input the GFlowNet's
    forward policy takes.  It is used only by the cross-algorithm conditions
    and their architecture-matched controls; every other condition leaves it
    off and is unaffected.
    """
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; have {VARIANTS}")
    corr = get(corruption)
    active = variant == "corrupted"

    kwargs = {"render_mode": render_mode}
    if active and corr.mechanism == "layout":
        kwargs["maze_map"] = corr.layout(UMAZE)
    env = gym.make(ENV_ID, **kwargs)

    if active and corr.mechanism == "actuation":
        env = LinearActionCorruption(env, corr.action_matrix)
    if discrete:
        env = BangBangDiscreteActions(env)
    env = GoalFlattenObservation(env)
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    if time_feature:
        env = TimeFeature(env, MAX_EPISODE_STEPS // max(1, action_repeat))

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


__all__ = [
    "ENV_ID",
    "MAX_EPISODE_STEPS",
    "VARIANTS",
    "layout_for",
    "make_env",
]
