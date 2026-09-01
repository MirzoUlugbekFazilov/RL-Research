"""Maze layouts and the corruption that defines this study.

Every fact in this module was verified against the installed
``gymnasium_robotics`` (1.4.2) by ``scripts/verify_env.py`` rather than taken
from documentation.  See ``METHODOLOGY.md`` §2 for the write-up.

Coordinate convention (verified via ``maze.cell_rowcol_to_xy``)
--------------------------------------------------------------
A maze map is a ``(R, C)`` matrix of 0 (free) / 1 (wall).  Cell ``(r, c)`` has
its centre at::

    x = (c - C//2) * maze_size_scaling
    y = (R//2 - r) * maze_size_scaling

with ``maze_size_scaling == 1.0`` for PointMaze.  So *columns run +x to the
right* and *rows run -y downward*.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

#: The layout that ``gym.make("PointMaze_UMaze-v3")`` uses by default.  Read
#: straight off ``env.spec.kwargs["maze_map"]``; do not edit by hand.
UMAZE: list[list[int]] = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

#: Physical size of one maze cell, in metres (PointMaze ``maze_size_scaling``).
MAZE_SIZE_SCALING = 1.0

#: Radius of the point-mass geom, in metres (``particle_geom`` half-size).
BALL_RADIUS = 0.1

#: Goal tolerance actually implemented in ``MazeEnv.compute_reward`` /
#: ``compute_terminated``.  NOTE: the Gymnasium-Robotics *documentation* says
#: 0.5 m; the *code* in 1.4.2 uses 0.45 m.  We follow the code.
GOAL_RADIUS = 0.45


# The corruption family lives in `environments/corruptions.py`; this module is
# deliberately limited to facts about the *uncorrupted* maze so that the
# geometry helpers below cannot silently depend on which corruption is active.


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cell_to_xy(row: int, col: int, maze_map: list[list[int]]) -> np.ndarray:
    """Centre of cell ``(row, col)`` in world coordinates."""
    n_rows, n_cols = len(maze_map), len(maze_map[0])
    x = (col - n_cols // 2) * MAZE_SIZE_SCALING
    y = (n_rows // 2 - row) * MAZE_SIZE_SCALING
    return np.array([x, y], dtype=np.float64)


def free_cells(maze_map: list[list[int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, row in enumerate(maze_map)
        for c, v in enumerate(row)
        if v == 0
    ]


def free_cell_xy(maze_map: list[list[int]]) -> np.ndarray:
    """``(n_free, 2)`` array of free-cell centres."""
    return np.array([cell_to_xy(r, c, maze_map) for r, c in free_cells(maze_map)])


def adjacency(maze_map: list[list[int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """4-connected adjacency graph over free cells."""
    fc = set(free_cells(maze_map))
    return {
        (r, c): [n for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if n in fc]
        for (r, c) in fc
    }


def maze_bounds(maze_map: list[list[int]]) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` of the maze footprint, in metres."""
    n_rows, n_cols = len(maze_map), len(maze_map[0])
    half = 0.5 * MAZE_SIZE_SCALING
    x_min = (0 - n_cols // 2) * MAZE_SIZE_SCALING - half
    x_max = (n_cols - 1 - n_cols // 2) * MAZE_SIZE_SCALING + half
    y_min = (n_rows // 2 - (n_rows - 1)) * MAZE_SIZE_SCALING - half
    y_max = (n_rows // 2 - 0) * MAZE_SIZE_SCALING + half
    return x_min, x_max, y_min, y_max
