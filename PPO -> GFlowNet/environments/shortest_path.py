r"""Exact-as-practical geodesic distances inside a PointMaze layout.

This is the denominator-free half of the *path efficiency* metric: we need
:math:`L^\ast(s_0, g)`, the length of the shortest collision-free path the
ball's **centre** can take from its start position to the goal *region*.

Method (documented because §9D of the brief forbids undocumented approximations)
--------------------------------------------------------------------------------
1. **Configuration space.**  The ball is a disc of radius
   :data:`~environments.layouts.BALL_RADIUS` (0.1 m, read from the MuJoCo
   model).  Each wall cell is an axis-aligned 1 x 1 m square.  A centre
   position :math:`p` is feasible iff its distance to *every* wall square
   exceeds the ball radius, using the exact point-to-box distance

   .. math:: d(p, \text{box}_c) = \big\| \max(|p - c| - \tfrac{1}{2},\, 0) \big\|_2

   which is the exact Minkowski inflation (rounded corners), not the cruder
   axis-aligned inflation.

2. **Discretisation.**  Feasible space is sampled on a uniform grid of pitch
   ``resolution`` (default 0.025 m).

3. **Graph.**  A **32-neighbour** stencil -- all *primitive* integer offsets
   :math:`(dx, dy)` with :math:`\max(|dx|,|dy|) \le 3` and
   :math:`\gcd(|dx|,|dy|) = 1` -- with Euclidean edge weights.  Any such
   stencil overestimates a straight line by at most
   :math:`\sec(\theta_{\max}/2) - 1`, where :math:`\theta_{\max}` is the
   widest angular gap between adjacent stencil directions.  Computed by
   :func:`stencil_error_bound`, that is **8.24%** for the 8-neighbour grid,
   **2.75%** for 16 neighbours and **1.31%** for the 32 neighbours used here.
   Multi-cell edges additionally require the lattice points along the segment
   to be feasible, which prevents clipping a convex wall corner (walls are
   1 m thick -- 13x the stencil reach -- so tunnelling is impossible).
   :meth:`~MazeGeodesic.validate_against_free_space` measures the realised
   error at **1.27%**, consistent with the bound.

4. **Goal region.**  PointMaze counts success when
   :math:`\|p - g\| \le 0.45`, so the target is the whole disc, not the point
   :math:`g`.  We run a single-source Dijkstra from the start node and take
   the minimum over all feasible nodes inside the disc.  This matches the
   success criterion exactly.

**Known bias:** the reported :math:`L^\ast` is an *upper* bound on the true
continuous geodesic (by <=1.31% plus up to one grid pitch from snapping the
endpoints).  Path efficiency :math:`L^\ast / L_\text{agent}` is therefore
biased *very slightly upward*, identically for every condition, so it cannot
favour one condition over another.

**Exact mirror symmetry.**  Because the corruption is the isometry
:math:`(x,y) \mapsto (-x,y)`, this module must return *bit-identical*
distances for a query and its mirror image, or the "corruption preserves
difficulty" claim would rest on a numerical artefact.  It does: the lattice is
built by rounding rather than by accumulating ``arange`` steps, and
feasibility is tested boundary-inclusively, because grid points land exactly
on the clearance boundary (0.6 m from a wall centre is on the 0.025 m
lattice).  ``sanity_checks.py`` asserts a max mirror-pair discrepancy of 0.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .layouts import BALL_RADIUS, GOAL_RADIUS, MAZE_SIZE_SCALING, cell_to_xy, maze_bounds

DEFAULT_RESOLUTION = 0.025


STENCIL_REACH = 3


def _stencil(reach: int = STENCIL_REACH) -> list[tuple[int, int]]:
    """All primitive integer offsets within a Chebyshev radius of ``reach``.

    ``reach=1`` -> 8 neighbours, ``reach=2`` -> 16, ``reach=3`` -> 32.
    """
    offs = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx == 0 and dy == 0:
                continue
            if math.gcd(abs(dx), abs(dy)) != 1:
                continue
            offs.append((dx, dy))
    return offs


def stencil_error_bound(reach: int = STENCIL_REACH) -> float:
    r"""Worst-case relative overestimate of a straight line by this stencil.

    A straight segment at angle :math:`\phi` must be approximated by mixing
    the two stencil directions that bracket it; the worst case is the bisector
    of the widest angular gap :math:`\theta`, giving
    :math:`\sec(\theta/2) - 1`.
    """
    angles = sorted(math.atan2(dy, dx) % (2 * math.pi) for dx, dy in _stencil(reach))
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    gaps.append(angles[0] + 2 * math.pi - angles[-1])
    return 1.0 / math.cos(max(gaps) / 2.0) - 1.0


class MazeGeodesic:
    """Geodesic distance queries for one maze layout.  Build once, query often."""

    def __init__(self, maze_map, resolution: float = DEFAULT_RESOLUTION,
                 ball_radius: float = BALL_RADIUS,
                 stencil_reach: int = STENCIL_REACH):
        self.maze_map = [list(r) for r in maze_map]
        self.resolution = float(resolution)
        self.ball_radius = float(ball_radius)
        self.stencil_reach = int(stencil_reach)

        x_min, x_max, y_min, y_max = maze_bounds(self.maze_map)
        # Build the lattice by rounding rather than by accumulating `arange`
        # steps: grid points land *exactly* on the clearance boundary (0.6 m
        # from a wall centre is on the 0.025 m lattice), so 1-ulp asymmetries
        # in the coordinates would otherwise flip feasibility tests and break
        # the mirror symmetry the corruption depends on.
        self.nx = int(round((x_max - x_min) / self.resolution)) + 1
        self.ny = int(round((y_max - y_min) / self.resolution)) + 1
        self.xs = np.round(x_min + np.arange(self.nx) * self.resolution, 12)
        self.ys = np.round(y_min + np.arange(self.ny) * self.resolution, 12)

        self.feasible = self._build_feasible()
        self._graph = self._build_graph()

    # -- geometry ---------------------------------------------------------
    def _wall_centres(self) -> np.ndarray:
        return np.array(
            [
                cell_to_xy(r, c, self.maze_map)
                for r, row in enumerate(self.maze_map)
                for c, v in enumerate(row)
                if v == 1
            ]
        )

    def clearance(self, pts: np.ndarray) -> np.ndarray:
        """Distance from each point to the nearest wall square (exact)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        walls = self._wall_centres()
        half = 0.5 * MAZE_SIZE_SCALING
        # (n_pts, n_walls, 2)
        delta = np.abs(pts[:, None, :] - walls[None, :, :]) - half
        delta = np.maximum(delta, 0.0)
        return np.linalg.norm(delta, axis=2).min(axis=1)

    def _build_feasible(self) -> np.ndarray:
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        # Boundary-inclusive (`>=` up to a tolerance): the infimum of path
        # lengths over the open free set equals the minimum over its closure,
        # so including grazing contact gives the correct geodesic and, unlike
        # a strict `>`, is not decided by floating-point noise.
        ok = self.clearance(pts) >= self.ball_radius - 1e-9
        return ok.reshape(self.nx, self.ny)

    # -- graph ------------------------------------------------------------
    def _build_graph(self):
        n = self.nx * self.ny
        idx = np.arange(n).reshape(self.nx, self.ny)
        feas = self.feasible
        rows, cols, data = [], [], []
        for dx, dy in _stencil(self.stencil_reach):
            xs_a = slice(max(0, -dx), self.nx - max(0, dx))
            ys_a = slice(max(0, -dy), self.ny - max(0, dy))
            xs_b = slice(max(0, dx), self.nx - max(0, -dx))
            ys_b = slice(max(0, dy), self.ny - max(0, -dy))
            ok = feas[xs_a, ys_a] & feas[xs_b, ys_b]
            # Every lattice point the segment passes near must also be free.
            # Walls are 1 m thick (40x the stencil reach) so tunnelling is
            # impossible; this guards against clipping a convex wall corner.
            span = max(abs(dx), abs(dy))
            for k in range(1, span):
                mx, my = round(k * dx / span), round(k * dy / span)
                xs_m = slice(max(0, -dx) + mx, self.nx - max(0, dx) + mx)
                ys_m = slice(max(0, -dy) + my, self.ny - max(0, dy) + my)
                ok = ok & feas[xs_m, ys_m]
            a = idx[xs_a, ys_a][ok]
            b = idx[xs_b, ys_b][ok]
            w = self.resolution * math.hypot(dx, dy)
            rows.append(a)
            cols.append(b)
            data.append(np.full(a.size, w))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # -- queries ----------------------------------------------------------
    def _nearest_feasible_node(self, p) -> int:
        p = np.asarray(p, dtype=np.float64)
        ix = int(np.clip(round((p[0] - self.xs[0]) / self.resolution), 0, self.nx - 1))
        iy = int(np.clip(round((p[1] - self.ys[0]) / self.resolution), 0, self.ny - 1))
        if self.feasible[ix, iy]:
            return ix * self.ny + iy
        # spiral outward -- only needed if MuJoCo lets the ball touch a wall
        for r in range(1, 12):
            best, best_d = None, np.inf
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < self.nx and 0 <= jy < self.ny and self.feasible[jx, jy]:
                        d = math.hypot(self.xs[jx] - p[0], self.ys[jy] - p[1])
                        if d < best_d:
                            best, best_d = jx * self.ny + jy, d
            if best is not None:
                return best
        raise RuntimeError(f"no feasible grid node near {p}")

    def distance_to_goal_region(self, start, goal, goal_radius: float = GOAL_RADIUS) -> float:
        """Shortest feasible path length from ``start`` to the goal disc."""
        src = self._nearest_feasible_node(start)
        dist = dijkstra(self._graph, directed=False, indices=src)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        inside = ((gx - goal[0]) ** 2 + (gy - goal[1]) ** 2) <= goal_radius**2
        mask = (inside & self.feasible).ravel()
        if not mask.any():
            return float("nan")
        d = dist[mask]
        d = d[np.isfinite(d)]
        return float(d.min()) if d.size else float("nan")

    def validate_against_free_space(self) -> dict:
        """Realised discretisation error on straight, unobstructed segments.

        Picks pairs of points in the same corridor whose true geodesic is the
        straight line, and reports the relative overestimate.
        """
        cases = [
            ((-1.0, 1.0), (1.0, 1.0)),   # top corridor, axis-aligned
            ((-1.0, -1.0), (1.0, -1.0)),  # bottom corridor, axis-aligned
            ((-1.3, 1.3), (1.3, 0.8)),   # oblique inside the top corridor
        ]
        errs = []
        for a, b in cases:
            a, b = np.array(a), np.array(b)
            if self.clearance(a[None])[0] <= self.ball_radius:
                continue
            if self.clearance(b[None])[0] <= self.ball_radius:
                continue
            truth = float(np.linalg.norm(b - a))
            got = self.distance_to_goal_region(a, b, goal_radius=1e-6)
            errs.append((truth, got, (got - truth) / truth))
        return {
            "cases": errs,
            "max_rel_error": max((abs(e[2]) for e in errs), default=float("nan")),
        }


@lru_cache(maxsize=8)
def _cached(maze_key: tuple, resolution: float, ball_radius: float) -> MazeGeodesic:
    maze_map = [list(r) for r in maze_key]
    return MazeGeodesic(maze_map, resolution=resolution, ball_radius=ball_radius)


def get_geodesic(maze_map, resolution: float = DEFAULT_RESOLUTION,
                 ball_radius: float = BALL_RADIUS) -> MazeGeodesic:
    """Memoised :class:`MazeGeodesic` -- the graph build is the expensive part."""
    key = tuple(tuple(int(v) for v in row) for row in maze_map)
    return _cached(key, float(resolution), float(ball_radius))
