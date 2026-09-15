"""Per-episode records and the aggregate PointMaze metrics of §9.

Every metric here is computed from the environment's *own* success signal
(``info["success"]``, i.e. ``||achieved - desired|| <= 0.45``).  We never
invent a success criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class EpisodeRecord:
    """Everything measured about one frozen-policy evaluation episode."""

    episode: int
    seed: int
    ret: float                  # sum of sparse rewards over the episode
    success: bool               # env's own criterion, reached at any timestep
    length: int                 # steps actually run (300 under TimeLimit)
    steps_to_goal: int | None   # first timestep at which success became true
    path_length: float          # metres travelled up to first goal contact
    optimal_path: float         # geodesic start -> goal region
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (0.0, 0.0)

    @property
    def path_efficiency(self) -> float | None:
        """``L* / L_agent`` -- undefined (None) for failed episodes."""
        if not self.success or self.path_length <= 0:
            return None
        if not np.isfinite(self.optimal_path):
            return None
        return float(self.optimal_path / self.path_length)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path_efficiency"] = self.path_efficiency
        return d


@dataclass
class EvalMetrics:
    """Aggregate over the evaluation episodes of a single frozen model."""

    episodes: list[EpisodeRecord] = field(default_factory=list)

    # -- §9A ---------------------------------------------------------------
    @property
    def returns(self) -> np.ndarray:
        return np.array([e.ret for e in self.episodes], dtype=float)

    @property
    def mean_reward(self) -> float:
        return float(self.returns.mean()) if self.episodes else float("nan")

    # -- §9B ---------------------------------------------------------------
    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return float("nan")
        return float(np.mean([e.success for e in self.episodes]))

    @property
    def n_success(self) -> int:
        return int(sum(e.success for e in self.episodes))

    # -- §9C ---------------------------------------------------------------
    @property
    def mean_steps_to_goal(self) -> float:
        """Mean over **successful episodes only**.

        Failed episodes are *excluded*, not counted as 300 -- averaging a
        censored value in would confound "slow" with "never arrived".  The
        censored variant is reported separately as
        :attr:`mean_steps_to_goal_censored`, and :attr:`n_success` says how
        many episodes the uncensored mean is based on, so a low mean from a
        handful of lucky episodes cannot masquerade as good performance.
        """
        v = [e.steps_to_goal for e in self.episodes if e.success and e.steps_to_goal is not None]
        return float(np.mean(v)) if v else float("nan")

    @property
    def mean_steps_to_goal_censored(self) -> float:
        """Failures charged the full horizon. Comparable across success rates."""
        if not self.episodes:
            return float("nan")
        v = [
            e.steps_to_goal if (e.success and e.steps_to_goal is not None) else e.length
            for e in self.episodes
        ]
        return float(np.mean(v))

    # -- §9D ---------------------------------------------------------------
    @property
    def mean_path_efficiency(self) -> float:
        """Mean over successful episodes only (undefined otherwise)."""
        v = [e.path_efficiency for e in self.episodes if e.path_efficiency is not None]
        return float(np.mean(v)) if v else float("nan")

    # -- §9E ---------------------------------------------------------------
    @property
    def return_std(self) -> float:
        """SD of returns *across evaluation episodes* -- consistency, not
        uncertainty.  The across-seed SD used for inference is computed in
        ``analysis/stats.py`` and is a different quantity."""
        return float(self.returns.std(ddof=1)) if len(self.episodes) > 1 else float("nan")

    @property
    def return_sem(self) -> float:
        n = len(self.episodes)
        return float(self.returns.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    def summary(self) -> dict:
        return dict(
            n_episodes=len(self.episodes),
            mean_reward=self.mean_reward,
            success_rate=self.success_rate,
            n_success=self.n_success,
            mean_steps_to_goal=self.mean_steps_to_goal,
            mean_steps_to_goal_censored=self.mean_steps_to_goal_censored,
            mean_path_efficiency=self.mean_path_efficiency,
            return_std=self.return_std,
            return_sem=self.return_sem,
        )

    def to_records(self) -> list[dict]:
        return [e.to_dict() for e in self.episodes]
