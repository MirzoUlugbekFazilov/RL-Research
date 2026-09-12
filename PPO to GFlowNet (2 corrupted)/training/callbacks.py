"""Periodic frozen-policy evaluation during training (learning curves, §11).

The same schedule and the same evaluation function are used for both
algorithms; only the plumbing differs, because SB3 wants a ``BaseCallback``
while the GFlowNet takes a plain function.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from evaluation.evaluator import evaluate_policy

#: Episodes per learning-curve point. Fewer than the 100 used for the final
#: frozen evaluation, because a curve has many points and its job is to locate
#: *when* a threshold is crossed, not to be the headline number.
CURVE_EPISODES = 30


class CurveRecorder:
    """Collects ``(steps, metrics)`` points on a fixed environment-step grid."""

    def __init__(self, agent_ref, eval_variant: str, corruption: str,
                 every: int, phase: str, episodes: int = CURVE_EPISODES,
                 step_offset: int = 0):
        self.agent_ref = agent_ref
        self.eval_variant = eval_variant
        self.corruption = corruption
        self.every = every
        self.phase = phase
        self.episodes = episodes
        self.step_offset = step_offset
        self.rows: list[dict] = []
        self._next_at = 0

    def maybe_record(self, agent, steps_in_phase: int, force: bool = False) -> None:
        if not force and steps_in_phase < self._next_at:
            return
        self._next_at = steps_in_phase + self.every
        m = evaluate_policy(
            agent, self.eval_variant, corruption=self.corruption,
            n_episodes=self.episodes,
        )
        self.rows.append(
            dict(
                phase=self.phase,
                steps_in_phase=int(steps_in_phase),
                total_steps=int(self.step_offset + steps_in_phase),
                mean_reward=m.mean_reward,
                success_rate=m.success_rate,
                mean_steps_to_goal=m.mean_steps_to_goal,
                mean_path_efficiency=m.mean_path_efficiency,
                return_std=m.return_std,
                n_episodes=self.episodes,
            )
        )


class PPOCurveCallback(BaseCallback):
    """Adapter that drives a :class:`CurveRecorder` from inside SB3."""

    def __init__(self, recorder: CurveRecorder, agent):
        super().__init__(verbose=0)
        self.recorder = recorder
        self.agent = agent

    def _on_training_start(self) -> None:
        # Point at step 0 so every curve has its own pre-training baseline --
        # for a fine-tuning run this is the zero-shot performance of the
        # corrupted checkpoint, which is exactly the "after corruption" number
        # §10 asks for.
        self.recorder.maybe_record(self.agent, 0, force=True)

    def _on_step(self) -> bool:
        # SB3 counts decisions; the study's currency is environment steps.
        steps = int(self.num_timesteps) * self.agent.action_repeat
        self.recorder.maybe_record(self.agent, steps)
        return True


def gflownet_curve_callback(recorder: CurveRecorder):
    def _cb(agent, steps_in_phase: int) -> None:
        recorder.maybe_record(agent, steps_in_phase)

    return _cb
