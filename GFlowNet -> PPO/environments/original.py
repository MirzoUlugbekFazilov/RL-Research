"""The *target* environment: unmodified ``PointMaze_UMaze-v3``.

Nothing here overrides the benchmark.  The only kwarg we ever pass is
``render_mode``.  ``reward_type`` stays at the registered default (``sparse``)
and ``continuing_task`` stays at the registered default (``True``).
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics

from .layouts import UMAZE

gym.register_envs(gymnasium_robotics)

ENV_ID = "PointMaze_UMaze-v3"

#: Registered ``max_episode_steps`` for ``PointMaze_UMaze-v3``.
MAX_EPISODE_STEPS = 300


def make_original(render_mode: str | None = None, **kwargs) -> gym.Env:
    """``gym.make("PointMaze_UMaze-v3")`` with no task-altering overrides."""
    return gym.make(ENV_ID, render_mode=render_mode, **kwargs)


def original_layout() -> list[list[int]]:
    return [row[:] for row in UMAZE]
