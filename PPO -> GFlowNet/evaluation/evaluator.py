"""The single evaluation protocol, used identically by PPO and the GFlowNet.

Protocol (§8)
-------------
* The model is **frozen**.  :func:`evaluate_policy` never touches an optimiser,
  never calls ``learn``, and puts torch modules in ``eval()`` under
  ``torch.no_grad()``.  ``sanity_checks.py`` asserts parameter hashes are
  unchanged across an evaluation.
* **Fixed instance set.**  Episode *i* always runs ``env.reset(seed=
  EVAL_SEED_BASE + i)``, independent of the training seed.  Every model in the
  study therefore faces the *same* 100 (start, goal) instances, which makes
  condition-vs-condition contrasts paired and removes evaluation-instance
  variance from the comparison.  The independent experimental units remain the
  training seeds (§6) -- these 100 episodes are one measurement of one model,
  not 100 experiments.
* **Deterministic action selection** for both algorithms: PPO takes the mean
  of its Gaussian, the GFlowNet takes ``argmax`` of its forward policy.  The
  stochastic variant is also recorded for the GFlowNet, since sampling
  proportional to reward is what a GFlowNet is actually trained to do.
"""

from __future__ import annotations

import numpy as np

from environments.corruptions import DEFAULT_CORRUPTION
from environments.factory import layout_for, make_env
from environments.shortest_path import get_geodesic

from .metrics import EpisodeRecord, EvalMetrics

#: Base seed for evaluation instances. Deliberately far from any training seed.
EVAL_SEED_BASE = 1_000_000

#: Default number of evaluation episodes per frozen model (§8).
N_EVAL_EPISODES = 100


def evaluate_policy(
    agent,
    variant: str = "original",
    *,
    corruption: str = DEFAULT_CORRUPTION,
    n_episodes: int = N_EVAL_EPISODES,
    deterministic: bool = True,
    seed: int = EVAL_SEED_BASE,
    discrete: bool | None = None,
    action_repeat: int | None = None,
    time_feature: bool | None = None,
    keep_episodes: bool = True,
) -> EvalMetrics:
    """Run ``n_episodes`` frozen-policy episodes and compute the §9 metrics.

    ``discrete``, ``action_repeat`` and ``time_feature`` default to the agent's
    own settings, so an agent is always evaluated on the interface it was
    trained on.

    All step counts and path lengths are recorded at **true environment-step
    resolution** even under action repeat: the wrapper reports each inner
    step's position and the exact inner step at which success occurred, so a
    repeat agent's "steps to goal" and path length are directly comparable to
    a continuous agent's rather than being quantised to its decision rate.
    """
    if discrete is None:
        discrete = bool(getattr(agent, "uses_discrete_actions", False))
    if action_repeat is None:
        action_repeat = int(getattr(agent, "action_repeat", 1))
    if time_feature is None:
        time_feature = bool(getattr(agent, "uses_time_feature", False))

    env = make_env(variant, corruption=corruption, discrete=discrete,
                   action_repeat=action_repeat, time_feature=time_feature)
    geo = get_geodesic(layout_for(variant, corruption))
    metrics = EvalMetrics()

    freeze = getattr(agent, "freeze", None)
    if callable(freeze):
        freeze()
    # Agents that need their own per-episode state (the GFlowNet tracks the
    # step index for its t/T input and holds an action across action-repeat)
    # get told where episode boundaries are. PPO has no such state.
    on_start = getattr(agent, "on_episode_start", None)

    try:
        for i in range(n_episodes):
            if callable(on_start):
                on_start()
            ep_seed = seed + i
            obs, _ = env.reset(seed=ep_seed)
            pos = np.asarray(obs[:2], dtype=float)
            goal = np.asarray(obs[4:6], dtype=float)
            start = pos.copy()

            ret = 0.0
            steps = 0
            success = False
            steps_to_goal: int | None = None
            path_len = 0.0
            path_len_to_goal = 0.0

            done = False
            while not done:
                action = agent.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ret += float(reward)

                # Positions after every *environment* step of this decision.
                # Without action repeat this is just the single new position.
                sub_positions = info.get("repeat_positions")
                if sub_positions is None:
                    sub_positions = [np.asarray(obs[:2], dtype=float)]
                    n_inner = 1
                    success_offset = 1 if info.get("success", False) else None
                else:
                    n_inner = int(info["repeat_steps"])
                    success_offset = info.get("success_offset")

                for j, p in enumerate(sub_positions):
                    path_len += float(np.linalg.norm(p - pos))
                    pos = p
                    if not success and success_offset is not None and j + 1 == success_offset:
                        success = True
                        steps_to_goal = steps + j + 1
                        path_len_to_goal = path_len
                steps += n_inner

                done = bool(terminated or truncated)

            optimal = geo.distance_to_goal_region(start, goal)
            rec = EpisodeRecord(
                episode=i,
                seed=ep_seed,
                ret=ret,
                success=success,
                length=steps,
                steps_to_goal=steps_to_goal,
                path_length=path_len_to_goal if success else path_len,
                optimal_path=float(optimal),
                start=(float(start[0]), float(start[1])),
                goal=(float(goal[0]), float(goal[1])),
            )
            if keep_episodes:
                metrics.episodes.append(rec)
            else:
                metrics.episodes.append(rec)
    finally:
        env.close()

    return metrics


def evaluate_random_baseline(
    variant: str = "original",
    *,
    corruption: str = DEFAULT_CORRUPTION,
    n_episodes: int = N_EVAL_EPISODES,
    seed: int = EVAL_SEED_BASE,
    rng_seed: int = 0,
) -> EvalMetrics:
    """Uniform-random-action reference, on the same instances.

    This is the floor every learned result must clear, and it is what defines
    :data:`environments.corruptions.RANDOM_SUCCESS_FLOOR`.
    """

    class _Random:
        uses_discrete_actions = False

        def __init__(self):
            self.space = make_env(variant, corruption=corruption).action_space
            self.space.seed(rng_seed)

        def act(self, obs, deterministic=True):
            return self.space.sample()

    return evaluate_policy(
        _Random(), variant, corruption=corruption, n_episodes=n_episodes, seed=seed
    )
