r"""Cross-algorithm initialisation: GFlowNet forward policy -> PPO actor.

What this module is for
-----------------------
The primary conditions of this study pair each algorithm with itself: PPO
pretrains and PPO fine-tunes, or the GFlowNet does both.  This module supports
the *cross* pairing -- pretrain the GFlowNet on the corrupted environment, then
fine-tune **PPO** on the original one -- which is what
``gfn_pretrain_ppo_finetune`` runs.

Why an exact weight copy, and not distillation
----------------------------------------------
The claim a cross-algorithm arm has to support is "PPO started from what the
GFlowNet learned".  Behavioural cloning or distillation would introduce a
second optimisation, with its own budget and its own failure modes, between
the two phases -- and any result could then be blamed on the distillation step
rather than on the transfer.  A parameter copy has no free parameters and no
budget, so the fine-tuning phase provably begins at the GFlowNet's own policy.

This is possible only because the two networks were made architecturally
compatible on purpose:

======================================  ======================================
GFlowNet ``GFlowNetPolicy.pf``          SB3 ``ActorCriticPolicy``
======================================  ======================================
``Linear(obs+1, 128)``                  ``mlp_extractor.policy_net.0``
``Tanh``                                ``Tanh`` (SB3 ``activation_fn``)
``Linear(128, 128)``                    ``mlp_extractor.policy_net.2``
``Tanh``                                ``Tanh``
``Linear(128, 9)``                      ``action_net``
======================================  ======================================

so that with ``net_arch=dict(pi=[128, 128])``, ``activation_fn=nn.Tanh`` and a
``Discrete(9)`` action space, PPO's actor is the *same function class* as the
GFlowNet's forward policy, and its categorical logits are the same tensor the
GFlowNet takes a softmax over.  The ``t/T`` input is supplied by
:class:`environments.wrappers.TimeFeature`.

What is **not** transferred, and why that is unavoidable
-------------------------------------------------------
* **The critic.** A GFlowNet has no value function.  Its ``logZ`` head is not
  one: it estimates the log partition function of the *initial* (start, goal)
  instance, i.e. a property of the whole trajectory distribution, not the
  expected return-to-go from the current state.  Copying it into a critic
  would be a category error, so the critic is randomly initialised.  Every
  architecture-matched control in this study fine-tunes from a *learned*
  critic, so this asymmetry is stated with the results rather than hidden --
  and it can only ever *disadvantage* the cross-algorithm arm.
* **Optimiser moments.** Adam state is per-parameter and the two optimisers
  cover different parameter sets; PPO starts with fresh moments.

:func:`transfer_gflownet_policy_to_ppo` verifies the copy numerically rather
than trusting it: after loading, PPO's action logits are compared against the
GFlowNet's on a batch of probe observations, and the maximum absolute
difference and the greedy-action agreement rate are recorded in the run's
``transfer.json``.  A transfer that silently misaligned a layer would show up
there immediately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .gflownet.model import GFlowNetPolicy

#: ``GFlowNetPolicy.pf`` parameter -> SB3 ``ActorCriticPolicy`` parameter.
#: ``pf`` is ``[Linear, Tanh, Linear, Tanh, Linear]`` so the learnable indices
#: are 0, 2 and 4; SB3 splits the same stack into a trunk (``policy_net``,
#: indices 0 and 2) and a separate output layer (``action_net``).
PF_TO_PPO: dict[str, str] = {
    "pf.0.weight": "mlp_extractor.policy_net.0.weight",
    "pf.0.bias": "mlp_extractor.policy_net.0.bias",
    "pf.2.weight": "mlp_extractor.policy_net.2.weight",
    "pf.2.bias": "mlp_extractor.policy_net.2.bias",
    "pf.4.weight": "action_net.weight",
    "pf.4.bias": "action_net.bias",
}

#: Probe batch size for the numerical equivalence check.
N_PROBE = 512


def load_gflownet_net(checkpoint, hidden: int | None = None) -> tuple[GFlowNetPolicy, dict]:
    """Rebuild a :class:`GFlowNetPolicy` from a saved GFlowNet checkpoint."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    net = GFlowNetPolicy(obs_dim=6, n_actions=9, hidden=hidden or cfg.get("hidden", 128))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def _probe_observations(obs_space, n: int, seed: int) -> np.ndarray:
    """A reproducible batch of in-range observations for the equivalence check.

    Sampled uniformly inside the observation box rather than by rolling out a
    policy: the point is to compare two *functions* over their whole input
    domain, not on states one of them happens to visit.
    """
    rng = np.random.default_rng(seed)
    low = np.where(np.isfinite(obs_space.low), obs_space.low, -10.0)
    high = np.where(np.isfinite(obs_space.high), obs_space.high, 10.0)
    return rng.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)


@torch.no_grad()
def _ppo_logits(policy, obs: np.ndarray) -> torch.Tensor:
    """PPO's categorical logits -- the tensor the GFlowNet softmaxes."""
    t = torch.as_tensor(obs, dtype=torch.float32)
    features = policy.extract_features(t)
    if isinstance(features, tuple):  # SB3 returns (pi, vf) when extractors differ
        features = features[0]
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


@torch.no_grad()
def transfer_gflownet_policy_to_ppo(
    policy, checkpoint, *, obs_space, seed: int = 0, strict: bool = True
) -> dict:
    """Copy a GFlowNet forward policy into an SB3 actor **in place**.

    Parameters
    ----------
    policy      : the ``ActorCriticPolicy`` to overwrite (actor only).
    checkpoint  : path to a ``GFlowNetAgent.save`` ``.pt`` file.
    obs_space   : the ``Box(7,)`` the PPO agent will actually see; used to draw
                  probe observations for the equivalence check.
    strict      : raise if the copy does not reproduce the GFlowNet's logits.

    Returns
    -------
    dict
        A report suitable for serialising into ``transfer.json``: which
        parameters moved, their shapes, and the measured equivalence.
    """
    net, ckpt = load_gflownet_net(checkpoint)
    src = net.state_dict()
    dst = dict(policy.named_parameters())

    moved = []
    for src_name, dst_name in PF_TO_PPO.items():
        if src_name not in src:
            raise KeyError(f"GFlowNet checkpoint has no parameter {src_name!r}")
        if dst_name not in dst:
            raise KeyError(
                f"PPO policy has no parameter {dst_name!r}; the actor "
                f"architecture does not match the GFlowNet trunk"
            )
        s, d = src[src_name], dst[dst_name]
        if tuple(s.shape) != tuple(d.shape):
            raise ValueError(
                f"shape mismatch transferring {src_name} -> {dst_name}: "
                f"{tuple(s.shape)} vs {tuple(d.shape)}. The PPO actor must be "
                f"built with net_arch=dict(pi=[128, 128]) and Discrete(9)."
            )
        d.copy_(s)
        moved.append(dict(source=src_name, target=dst_name, shape=list(s.shape)))

    # ---- numerical proof that PPO now *is* the GFlowNet's forward policy ---
    obs = _probe_observations(obs_space, N_PROBE, seed)
    gfn_logits = net(torch.as_tensor(obs))
    ppo_logits = _ppo_logits(policy, obs)

    max_abs_diff = float((gfn_logits - ppo_logits).abs().max().item())
    agreement = float(
        (gfn_logits.argmax(-1) == ppo_logits.argmax(-1)).float().mean().item()
    )
    # Same logits => same categorical distribution; compare the probabilities
    # too so the check is meaningful even where the argmax is a near-tie.
    max_prob_diff = float(
        (torch.softmax(gfn_logits, -1) - torch.softmax(ppo_logits, -1))
        .abs().max().item()
    )

    report = dict(
        source_checkpoint=str(checkpoint),
        source_algo="gflownet",
        target_algo="ppo",
        parameters_transferred=moved,
        n_parameters_transferred=len(moved),
        critic_randomly_initialised=True,
        optimiser_moments_transferred=False,
        probe_observations=N_PROBE,
        probe_seed=seed,
        max_abs_logit_difference=max_abs_diff,
        max_abs_probability_difference=max_prob_diff,
        greedy_action_agreement=agreement,
        equivalent=bool(max_abs_diff < 1e-4 and agreement == 1.0),
        pretrain_steps_in_checkpoint=int(ckpt.get("num_timesteps", 0)),
    )

    if strict and not report["equivalent"]:
        raise AssertionError(
            "GFlowNet -> PPO transfer did not reproduce the source policy "
            f"(max |dlogit| = {max_abs_diff:.3e}, agreement = {agreement:.4f})"
        )
    return report


# ===========================================================================
# The reverse direction: PPO actor -> GFlowNet forward policy
# ===========================================================================
#
# Study A needs the mirror image of everything above: PPO pretrains on the
# corrupted environment and the **GFlowNet** continues from its policy on the
# original one.  Because the two networks were made architecturally compatible
# on purpose (see the table in this module's docstring), the same exact-copy
# contract applies in reverse -- no distillation, no free parameters, no extra
# budget, so the fine-tuning phase provably begins at PPO's own policy.
#
# What is **not** transferred, and why that is unavoidable
# -------------------------------------------------------
# * **log Z.**  PPO has no partition function.  Its critic is not one: V(s) is
#   the expected return-to-go from the *current* state, whereas log Z(c) is the
#   log partition function of the whole trajectory distribution for the initial
#   (start, goal) instance.  Copying a critic into the log Z head would be the
#   same category error as copying log Z into a critic, so the log Z head is
#   randomly initialised.
# * **Optimiser moments.**  As in the forward direction: the two optimisers
#   cover different parameter sets, so Adam starts with fresh moments.
#
# This asymmetry exactly mirrors the one in the GFlowNet -> PPO direction (a
# fresh critic there, a fresh log Z head here), which is what keeps the two
# studies comparable: each cross-algorithm arm gives up precisely one head.

#: SB3 ``ActorCriticPolicy`` parameter -> ``GFlowNetPolicy.pf`` parameter.
#: The exact inverse of :data:`PF_TO_PPO`.
PPO_TO_PF: dict[str, str] = {dst: src for src, dst in PF_TO_PPO.items()}


@torch.no_grad()
def transfer_ppo_policy_to_gflownet(
    net: GFlowNetPolicy, policy, *, obs_space, seed: int = 0, strict: bool = True
) -> dict:
    """Copy an SB3 actor into a GFlowNet forward policy **in place**.

    Parameters
    ----------
    net        : the :class:`GFlowNetPolicy` to overwrite (``pf`` only).
    policy     : a trained ``ActorCriticPolicy`` with ``net_arch=dict(pi=[128,128])``,
                 ``activation_fn=nn.Tanh`` and a ``Discrete(9)`` action space.
    obs_space  : the ``Box(7,)`` both networks see; used to draw probe
                 observations for the equivalence check.
    strict     : raise if the copy does not reproduce the source policy.

    Returns
    -------
    dict
        A report suitable for serialising into ``transfer.json``.
    """
    src = dict(policy.named_parameters())
    dst = dict(net.named_parameters())

    moved = []
    for src_name, dst_name in PPO_TO_PF.items():
        if src_name not in src:
            raise KeyError(
                f"PPO policy has no parameter {src_name!r}; the actor was not "
                f"built with net_arch=dict(pi=[128, 128]) over Discrete(9)"
            )
        if dst_name not in dst:
            raise KeyError(f"GFlowNet policy has no parameter {dst_name!r}")
        s, d = src[src_name], dst[dst_name]
        if tuple(s.shape) != tuple(d.shape):
            raise ValueError(
                f"shape mismatch transferring {src_name} -> {dst_name}: "
                f"{tuple(s.shape)} vs {tuple(d.shape)}. The PPO actor must be "
                f"built with net_arch=dict(pi=[128, 128]), Tanh, Discrete(9), "
                f"and the environment must supply the t/T feature (Box(7,))."
            )
        d.copy_(s)
        moved.append(dict(source=src_name, target=dst_name, shape=list(s.shape)))

    # ---- numerical proof that the GFlowNet now *is* PPO's actor -----------
    obs = _probe_observations(obs_space, N_PROBE, seed)
    ppo_logits = _ppo_logits(policy, obs)
    gfn_logits = net(torch.as_tensor(obs))

    max_abs_diff = float((gfn_logits - ppo_logits).abs().max().item())
    agreement = float(
        (gfn_logits.argmax(-1) == ppo_logits.argmax(-1)).float().mean().item()
    )
    max_prob_diff = float(
        (torch.softmax(gfn_logits, -1) - torch.softmax(ppo_logits, -1))
        .abs().max().item()
    )

    report = dict(
        source_algo="ppo",
        target_algo="gflownet",
        parameters_transferred=moved,
        n_parameters_transferred=len(moved),
        logz_head_randomly_initialised=True,
        critic_discarded=True,
        optimiser_moments_transferred=False,
        probe_observations=N_PROBE,
        probe_seed=seed,
        max_abs_logit_difference=max_abs_diff,
        max_abs_probability_difference=max_prob_diff,
        greedy_action_agreement=agreement,
        equivalent=bool(max_abs_diff < 1e-4 and agreement == 1.0),
    )

    if strict and not report["equivalent"]:
        raise AssertionError(
            "PPO -> GFlowNet transfer did not reproduce the source policy "
            f"(max |dlogit| = {max_abs_diff:.3e}, agreement = {agreement:.4f})"
        )
    return report
