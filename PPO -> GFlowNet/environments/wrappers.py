"""Observation / action wrappers shared by every algorithm.

Both wrappers are applied identically to PPO and to the GFlowNet so that the
two algorithms see the *same* task.  Any asymmetry is called out explicitly.
"""

from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GoalFlattenObservation(gym.ObservationWrapper):
    """``Dict(observation, achieved_goal, desired_goal)`` -> ``Box(6,)``.

    The concatenation is ``[x, y, vx, vy, goal_x, goal_y]``.

    ``achieved_goal`` is deliberately dropped: ``verify_env.py`` confirms it is
    bit-identical to ``observation[:2]`` for PointMaze, so including it would
    only duplicate two inputs.  No information is lost.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space["observation"]
        goal_space = env.observation_space["desired_goal"]
        low = np.concatenate([obs_space.low, goal_space.low])
        high = np.concatenate([obs_space.high, goal_space.high])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def observation(self, observation):
        return np.concatenate(
            [observation["observation"], observation["desired_goal"]]
        ).astype(np.float64)


#: Bang-bang discretisation of ``Box(-1, 1, (2,))``: the 9 points of
#: ``{-1, 0, +1}^2``.  See :class:`BangBangDiscreteActions` for the rationale.
BANG_BANG_ACTIONS = np.array(
    list(itertools.product((-1.0, 0.0, 1.0), repeat=2)), dtype=np.float32
)


class BangBangDiscreteActions(gym.ActionWrapper):
    r"""Expose ``Discrete(9)`` over the extreme points of the force box.

    Why this discretisation and not an arbitrary grid
    -------------------------------------------------
    PointMaze's action is a force :math:`u \in [-1, 1]^2` applied to a point
    mass with linear damping -- a *linear* plant with a *box* control
    constraint.  For such systems Pontryagin's maximum principle gives the
    **bang-bang principle**: a time-optimal control can be taken to lie on the
    boundary of the constraint set almost everywhere.  The vertex/edge set
    :math:`\{-1, 0, +1\}^2` is precisely the set of maximal-magnitude forces
    in each of the 8 axis-and-diagonal directions, plus the null action -- the
    smallest discretisation that keeps a time-optimal control representable up
    to angular quantisation.

    This makes the reduction *scientifically motivated* rather than chosen for
    convenience.  It is nevertheless a real restriction of the action set, so:

    * it is applied **identically** to every algorithm that uses it;
    * a control condition (``ppo_discrete_*``) runs PPO on this same
      ``Discrete(9)`` space, so PPO-vs-GFlowNet can be compared on an
      identical action space rather than across a continuous/discrete gap;
    * the four *primary* conditions keep PPO continuous, matching the
      unmodified benchmark.

    ``scripts/sanity_checks.py::check_discretisation_is_not_crippling``
    measures the cost of the restriction empirically.
    """

    def __init__(self, env: gym.Env, action_table: np.ndarray = BANG_BANG_ACTIONS):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        assert action_table.shape[1] == env.action_space.shape[0]
        self.action_table = np.asarray(action_table, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_table))

    def action(self, action):
        return self.action_table[int(action)]


class ActionRepeat(gym.Wrapper):
    """Hold each chosen action for ``k`` environment steps.

    Long trajectories are Trajectory Balance's known weak point: the TB loss
    sums ``log P_F`` over every decision, so its gradient variance grows with
    the decision count.  Repeating actions shortens a 300-step episode to
    300/k decisions without changing the horizon, the physics or the reward.

    **Budget accounting is unaffected**: ``k`` inner steps are ``k``
    environment steps, and that is what every budget in this study is counted
    in.  A repeat agent therefore gets *fewer decisions* for the same number of
    environment steps, never more interaction.

    To keep the §9 metrics measured at true environment-step resolution rather
    than at decision resolution -- which would flatter a repeat agent's path
    length and steps-to-goal -- the wrapper reports what happened *inside* the
    repeat:

    ``info["repeat_steps"]``      inner steps actually taken (< k if the
                                  episode ended mid-repeat)
    ``info["repeat_positions"]``  ball position after each inner step
    ``info["success_offset"]``    1-based inner step at which success first
                                  occurred during this repeat, else ``None``

    Applied identically to the GFlowNet and to the PPO-discrete control, so it
    is never a difference between those two.
    """

    def __init__(self, env: gym.Env, k: int = 1):
        super().__init__(env)
        if k < 1:
            raise ValueError("action repeat must be >= 1")
        self.action_repeat = int(k)

    def step(self, action):
        total_reward = 0.0
        positions: list[np.ndarray] = []
        success_offset = None
        terminated = truncated = False
        obs = None
        info: dict = {}
        n = 0

        for j in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            n += 1
            total_reward += float(reward)
            positions.append(np.asarray(obs[:2], dtype=float).copy())
            if success_offset is None and bool(info.get("success", False)):
                success_offset = j + 1
            if terminated or truncated:
                break

        info = dict(info)
        info["repeat_steps"] = n
        info["repeat_positions"] = positions
        info["success_offset"] = success_offset
        return obs, total_reward, terminated, truncated, info


class TimeFeature(gym.ObservationWrapper):
    r"""Append the normalised decision index ``t / D`` to the observation.

    ``Box(6,)`` -> ``Box(7,)``, the appended entry running from ``0`` at reset
    to ``1`` at truncation.

    Why this exists
    ---------------
    The GFlowNet's forward policy is :math:`P_F(a \mid s, t/T)`: the step index
    is *part of the network input*, which is what makes the sampling DAG a tree
    and lets the Trajectory Balance loss drop its :math:`P_B` term (see
    ``algorithms/gflownet/formulation.md``).  The GFlowNet agent appends that
    feature itself, inside ``act`` and ``_collect``.

    A PPO agent that is to be **initialised from a GFlowNet's weights** must
    therefore receive the identical 7-dimensional input, or the transferred
    first layer would be fed a different quantity in its last column and the
    transfer would be meaningless.  Rather than teach ``PPOAgent`` to carry a
    per-episode counter -- which SB3's vectorised rollout collection makes
    error-prone -- the feature is produced by the environment, where the
    episode boundary is already known exactly.

    ``t`` counts **decisions**, not environment steps, so this wrapper must sit
    *outside* :class:`ActionRepeat`.  With the study's ``k = 5`` an episode is
    ``300 / 5 = 60`` decisions, matching
    ``GFlowNetConfig.decisions_per_episode``.

    The feature is a function of elapsed time only.  It adds no information
    about the maze, the goal or the corruption, and every condition that uses
    it uses it in both phases, so it is never a difference between a
    fine-tuning arm and its from-scratch control.
    """

    def __init__(self, env: gym.Env, decisions_per_episode: int):
        super().__init__(env)
        if decisions_per_episode < 1:
            raise ValueError("decisions_per_episode must be >= 1")
        self.decisions_per_episode = int(decisions_per_episode)
        base = env.observation_space
        assert isinstance(base, spaces.Box) and len(base.shape) == 1
        self.observation_space = spaces.Box(
            np.concatenate([base.low, [0.0]]),
            np.concatenate([base.high, [1.0]]),
            dtype=np.float64,
        )
        self._t = 0

    def observation(self, observation):
        frac = min(1.0, self._t / self.decisions_per_episode)
        return np.concatenate([observation, [frac]]).astype(np.float64)

    def reset(self, **kwargs):
        self._t = 0
        return super().reset(**kwargs)

    def step(self, action):
        # ObservationWrapper.step would apply `observation` before we could
        # advance the counter, so the transition is written out explicitly.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        return self.observation(obs), reward, terminated, truncated, info
