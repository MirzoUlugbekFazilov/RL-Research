"""PPO (Stable-Baselines3) behind the shared :class:`~algorithms.base.Agent`.

Fine-tuning contract
--------------------
:meth:`PPOAgent.finetune_from` uses ``PPO.load`` on the pretrained ``.zip`` and
then *swaps the environment*.  It never constructs a fresh ``PPO``, so the
policy, value function and optimiser state all carry over from the corrupted
run -- which is what §4 Condition 2 demands.  ``parameter_fingerprint()`` is
recorded immediately after the load and again at the start of fine-tuning so
the sanity checks can prove the weights really are the pretrained ones.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.corruptions import DEFAULT_CORRUPTION
from environments.factory import make_env

#: Hyperparameters. Re-exported from :mod:`hyperparameters`, which is the
#: single source of truth shared byte-for-byte by both studies -- they are not
#: duplicated here, so the two studies cannot drift apart. Identical for every
#: condition, both phases and both environments; only the *environment* changes
#: between conditions.
from hyperparameters import PPO_HPARAMS as _SHARED_PPO_HPARAMS
from hyperparameters import PPO_N_ENVS, ppo_policy_kwargs

PPO_HPARAMS = dict(_SHARED_PPO_HPARAMS, policy_kwargs=ppo_policy_kwargs())

N_ENVS = PPO_N_ENVS


def _make_vec(variant, corruption, discrete, seed, n_envs=N_ENVS, action_repeat=1,
              time_feature=False):
    return DummyVecEnv(
        [
            (lambda i=i: make_env(variant, corruption=corruption, discrete=discrete,
                                  action_repeat=action_repeat,
                                  time_feature=time_feature, seed=seed * 1000 + i))
            for i in range(n_envs)
        ]
    )


def _policy_kwargs(policy_hidden) -> dict:
    """``policy_kwargs`` for an actor matching the GFlowNet's forward-policy trunk.

    ``policy_hidden=()`` keeps the study's default ``net_arch=[64, 64]``, so
    every pre-existing condition is untouched.  A non-empty value builds both
    heads at that width with ``Tanh`` activations -- the architecture
    :mod:`algorithms.transfer` requires in order to copy GFlowNet weights in
    without any projection.  The critic is given the same width as the actor
    purely for symmetry; it is randomly initialised in every condition.
    """
    if not policy_hidden:
        return dict(PPO_HPARAMS["policy_kwargs"])
    h = [int(x) for x in policy_hidden]
    return dict(net_arch=dict(pi=list(h), vf=list(h)), activation_fn=torch.nn.Tanh)


class PPOAgent:
    algo = "ppo"

    def __init__(self, model: PPO, variant: str, corruption: str,
                 discrete: bool, seed: int, action_repeat: int = 1,
                 time_feature: bool = False):
        self.model = model
        self.variant = variant
        self.corruption = corruption
        self.uses_discrete_actions = discrete
        self.seed = seed
        #: Env steps per decision. 1 for the primary continuous conditions; set
        #: to the GFlowNet's value for the PPO-discrete control so that the
        #: PPO-vs-GFlowNet contrast is read off an identical interface.
        self.action_repeat = int(action_repeat)
        #: Whether the observation carries the GFlowNet's ``t/T`` input. The
        #: evaluator reads this so an agent is always evaluated on the exact
        #: interface it was trained on.
        self.uses_time_feature = bool(time_feature)

    # -- construction ------------------------------------------------------
    @classmethod
    def fresh(cls, variant: str = "original", *, corruption: str = DEFAULT_CORRUPTION,
              discrete: bool = False, seed: int = 0, n_envs: int = N_ENVS,
              action_repeat: int = 1, time_feature: bool = False,
              policy_hidden: tuple[int, ...] = (),
              hparams: dict | None = None) -> "PPOAgent":
        """A brand-new, randomly initialised PPO (Condition 1's start)."""
        venv = _make_vec(variant, corruption, discrete, seed, n_envs, action_repeat,
                         time_feature)
        hp = dict(PPO_HPARAMS)
        hp["policy_kwargs"] = _policy_kwargs(policy_hidden)
        hp.update(hparams or {})
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **hp)
        return cls(model, variant, corruption, discrete, seed, action_repeat,
                   time_feature)

    @classmethod
    def finetune_from(cls, checkpoint, variant: str = "original", *,
                      corruption: str = DEFAULT_CORRUPTION, discrete: bool = False,
                      seed: int = 0, n_envs: int = N_ENVS,
                      action_repeat: int = 1, time_feature: bool = False,
                      reset_exploration: bool = False) -> "PPOAgent":
        """Load an existing checkpoint and point it at a new environment.

        Deliberately *not* a fresh model: this is the whole point of §4
        Condition 2.  ``reset_num_timesteps`` is left to the caller of
        :meth:`learn`; we reset the counter here so that "environment steps of
        adaptation" is measured from zero.

        ``reset_exploration`` restores the Gaussian policy's ``log_std`` to its
        initialisation value (std = 1) while keeping every *mean* weight from
        the checkpoint.  This exists purely as a control: the GFlowNet's
        training procedure includes an eps-uniform behaviour policy whose
        schedule restarts at the beginning of fine-tuning, whereas PPO's
        exploration is a *learned* parameter that carries over from the
        corrupted run.  If the GFlowNet adapts and PPO does not, this control
        distinguishes "the algorithm differs" from "only the exploration
        schedule differs".  It is **not** used by any primary condition.
        """
        venv = _make_vec(variant, corruption, discrete, seed, n_envs, action_repeat,
                         time_feature)
        model = PPO.load(checkpoint, env=venv, device="cpu")
        model.set_random_seed(seed)
        if reset_exploration and hasattr(model.policy, "log_std"):
            with torch.no_grad():
                model.policy.log_std.fill_(0.0)  # std = 1, as at initialisation
            # Adam moments for that tensor would otherwise fight the reset.
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(), lr=model.learning_rate
                if not callable(model.learning_rate) else model.learning_rate(1.0),
                **model.policy.optimizer_kwargs
            )
        agent = cls(model, variant, corruption, discrete, seed, action_repeat,
                    time_feature)
        agent._loaded_from = str(checkpoint)
        agent._fingerprint_at_load = agent.parameter_fingerprint()
        return agent

    @classmethod
    def from_gflownet(cls, checkpoint, variant: str = "original", *,
                      corruption: str = DEFAULT_CORRUPTION, seed: int = 0,
                      n_envs: int = N_ENVS, action_repeat: int = 5,
                      policy_hidden: tuple[int, ...] = (128, 128),
                      time_feature: bool = True) -> "PPOAgent":
        """Start PPO from a **GFlowNet** checkpoint and point it at ``variant``.

        This is the cross-algorithm fine-tuning contract: the GFlowNet trains
        on the corrupted environment, and PPO continues from its forward policy
        on the original one.  The actor is overwritten with the GFlowNet's
        weights and the copy is verified numerically; the critic is
        necessarily fresh, because a GFlowNet has none.  See
        :mod:`algorithms.transfer` for why an exact copy is used rather than
        distillation, and for what the verification proves.

        The resulting agent carries ``transfer_report``, which
        ``run_condition`` writes into the run's ``transfer.json``.
        """
        from algorithms.transfer import transfer_gflownet_policy_to_ppo

        agent = cls.fresh(
            variant, corruption=corruption, discrete=True, seed=seed,
            n_envs=n_envs, action_repeat=action_repeat,
            time_feature=time_feature, policy_hidden=policy_hidden,
        )
        obs_space = agent.model.observation_space
        agent.transfer_report = transfer_gflownet_policy_to_ppo(
            agent.model.policy, checkpoint, obs_space=obs_space, seed=seed,
        )
        agent._loaded_from = str(checkpoint)
        agent._fingerprint_at_load = agent.parameter_fingerprint()
        return agent

    @classmethod
    def load(cls, checkpoint, variant: str = "original", *,
             corruption: str = DEFAULT_CORRUPTION, discrete: bool = False,
             seed: int = 0, action_repeat: int = 1,
             time_feature: bool = False) -> "PPOAgent":
        """Load for *evaluation only* (no environment interaction planned)."""
        venv = _make_vec(variant, corruption, discrete, seed, 1, action_repeat,
                         time_feature)
        model = PPO.load(checkpoint, env=venv, device="cpu")
        return cls(model, variant, corruption, discrete, seed, action_repeat,
                   time_feature)

    # -- Agent protocol ----------------------------------------------------
    @property
    def num_timesteps(self) -> int:
        """Consumed budget in **environment steps**.

        SB3 counts calls to ``env.step``, which under action repeat is one per
        *decision*, not per environment step. We scale so that every agent in
        this study reports the same currency.
        """
        return int(self.model.num_timesteps) * self.action_repeat

    def act(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int, callback=None, reset_num_timesteps: bool = True):
        """``total_timesteps`` is a budget in **environment steps**.

        Converted to SB3's decision-step currency here, so that a repeat agent
        receives exactly the same amount of environment interaction as a
        non-repeat one rather than ``action_repeat`` times as much.
        """
        self.model.learn(
            total_timesteps=max(1, total_timesteps // self.action_repeat),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        return self

    def freeze(self) -> None:
        self.model.policy.set_training_mode(False)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def parameter_fingerprint(self, exclude: tuple[str, ...] = ()) -> str:
        """SHA-256 over the policy's parameters.

        ``exclude`` skips parameters whose name contains any of the given
        substrings.  The ``ppo_finetune_reset_std`` control deliberately
        overwrites ``log_std`` after loading, so its full fingerprint *must*
        differ from the checkpoint's; excluding ``log_std`` still lets the
        sanity checks prove every *mean* weight came from the checkpoint.
        """
        h = hashlib.sha256()
        with torch.no_grad():
            for name, p in sorted(self.model.policy.state_dict().items()):
                if any(e in name for e in exclude):
                    continue
                h.update(name.encode())
                h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
        return h.hexdigest()[:16]

    def close(self) -> None:
        try:
            self.model.get_env().close()
        except Exception:
            pass
