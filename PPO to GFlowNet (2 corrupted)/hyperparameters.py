"""The single source of truth for both studies.

This file is **byte-identical** in ``RL+gflowne/`` and ``RL+PPO/``.  Its
SHA-256 is written into every run's ``config.json``, so "the two studies used
the same hyperparameters" is a checkable claim rather than a promise:

    shasum -a 256 "RL+gflowne/hyperparameters.py" "RL+PPO/hyperparameters.py"

The two studies are exact mirror images::

    Study A (RL+gflowne)   PPO      on corrupted  ->  GFlowNet on original
    Study B (RL+PPO)       GFlowNet on corrupted  ->  PPO      on original

Nothing differs between them except *which algorithm runs in which phase*.
Same corruption, same budgets, same seeds, same evaluation instances, same
network shape, same action interface.

The shared interface, and why it is required
--------------------------------------------
Cross-algorithm fine-tuning by direct weight copy is only meaningful if the
two policies are the *same function class*.  Both studies therefore run every
phase on one interface:

===========================  ==================================================
``Discrete(9)`` actions      bang-bang ``{-1,0,+1}^2``; the extreme points of
                             the force box (Pontryagin -- see
                             ``environments/wrappers.py``)
``action_repeat = 5``        one decision per 5 environment steps -> 60
                             decisions per 300-step episode
``time_feature = True``      observation is ``[x,y,vx,vy,gx,gy,t/T]``, ``Box(7,)``
trunk ``[128, 128]``, Tanh   PPO's ``mlp_extractor.policy_net`` == GFlowNet's
                             ``pf`` hidden stack
head ``Linear(128, 9)``      PPO's ``action_net`` == GFlowNet's ``pf.4``; PPO's
                             categorical logits *are* the tensor the GFlowNet
                             softmaxes
===========================  ==================================================

This is a real restriction relative to the unmodified benchmark (which has a
continuous action space), and it is applied **identically to both algorithms
in both studies**, so it can never be a difference between a fine-tuning arm
and its from-scratch control.

Budget currency
---------------
Every budget below is in **environment steps**, never decisions and never
gradient updates.  Under ``action_repeat = 5`` an agent gets 60 decisions per
episode for 300 environment steps; both algorithms convert internally.  A
fine-tuning arm and its from-scratch control therefore receive exactly the
same amount of interaction with the original environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Environment / corruption
# ---------------------------------------------------------------------------

ENV_ID = "PointMaze_UMaze-v3"
MAX_EPISODE_STEPS = 300

#: The corruption both studies pretrain on.  ``u -> -u``: both actuators are
#: wired backwards.  Chosen because it is (a) provably difficulty-preserving --
#: it is an isometry of the action box, so the corrupted MDP is isomorphic to
#: the original and therefore *exactly as hard* -- and (b) measured to destroy
#: transfer completely: a policy trained on it scores 0.000 zero-shot on the
#: original, against a 0.26 uniform-random floor.  Any transfer gap is thus a
#: genuine transfer effect, not a difficulty confound.
#: See ``environments/corruptions.py`` for the full family and selection rule.
CORRUPTION = "negate_both"

# ---------------------------------------------------------------------------
# Shared interface (see module docstring)
# ---------------------------------------------------------------------------

DISCRETE = True          # Discrete(9) bang-bang actions
ACTION_REPEAT = 5        # environment steps per decision
TIME_FEATURE = True      # observation carries t/T -> Box(7,)
HIDDEN = 128             # width of the shared 2-layer Tanh trunk
N_ACTIONS = 9
OBS_DIM = 6              # before the t/T feature is appended
DECISIONS_PER_EPISODE = MAX_EPISODE_STEPS // ACTION_REPEAT   # 60

# ---------------------------------------------------------------------------
# Budgets (environment steps) and seeds
# ---------------------------------------------------------------------------

PRETRAIN_STEPS = 300_000   # phase 1, on the CORRUPTED environment
ADAPT_STEPS = 300_000      # phase 2, on the ORIGINAL environment

#: Independent experimental units. One seed = one trained model per arm; the
#: seed is the *only* thing that varies within an arm.
#:
#: Sized by power analysis, not by convenience. A first pass at n = 5 measured
#: |Hedges' g| in the range 1.06-1.69 on the primary contrasts but could not
#: clear Holm correction over the four reported metrics -- at n = 5 the power to
#: detect g = 1.1 at the Holm worst-case alpha of 0.0125 is only 0.15.  At
#: n = 30 that rises to ~0.93-0.96, so a null result becomes informative rather
#: than merely underpowered.  (If the true effects are half the observed g --
#: the winner's-curse case -- power is ~0.3-0.5 even here, which is stated as a
#: limitation rather than fixed by adding seeds: n = 40 would buy only ~0.1 more.)
#:
#: The first five are the original pass, retained rather than rerun so no result
#: is discarded after being seen; the remaining 25 are a contiguous block chosen
#: before any of them was run, so the set cannot have been curated.
SEEDS = (42, 123, 456, 789, 1000) + tuple(range(2000, 2025))

#: Known, disclosed overlap: environment seeds are ``seed * 1000 + i`` for
#: ``i < 8`` (``algorithms/ppo/agent.py::_make_vec``), so seed 1000 alone starts
#: its 8 training envs on instances that also appear in the evaluation set
#: (``EVAL_SEED_BASE = 1_000_000``).  That is 8 first-reset episodes out of the
#: many thousands each run trains on, against 8 of the 100 evaluation
#: instances.  It is retained rather than silently dropped -- removing a seed
#: after seeing its result is the worse sin -- and ``analyze.py`` reports the
#: primary contrast with and without it as a sensitivity check.
EVAL_OVERLAP_SEEDS = (1000,)

#: Learning-curve resolution during phase 2, in environment steps.
CURVE_EVERY = 15_000
#: Episodes per learning-curve point (cheap, noisy -- the curve is descriptive).
CURVE_EPISODES = 30

#: Episodes per *final* frozen-model evaluation (the number every headline
#: metric is computed from).
EVAL_EPISODES = 100
#: Episode ``i`` always resets with ``seed = EVAL_SEED_BASE + i``, so every
#: model in both studies faces the identical 100 (start, goal) instances and
#: all contrasts are paired.
EVAL_SEED_BASE = 1_000_000

#: Success-rate thresholds for the sample-efficiency metric (§ "steps to
#: threshold").  Fixed here, before any result was seen.
SUCCESS_THRESHOLDS = (0.25, 0.50, 0.80)

# ---------------------------------------------------------------------------
# PPO -- identical in both studies, both phases, both environments
# ---------------------------------------------------------------------------

PPO_N_ENVS = 8

PPO_HPARAMS = dict(
    n_steps=256,              # per env -> rollout buffer 256 * 8 = 2048
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

def ppo_policy_kwargs() -> dict:
    """Actor/critic shaped to match the GFlowNet forward-policy trunk exactly.

    ``net_arch=dict(pi=[128,128], vf=[128,128])`` with ``Tanh`` makes
    ``mlp_extractor.policy_net`` structurally identical to ``GFlowNetPolicy.pf``'s
    hidden stack and ``action_net`` identical to ``pf.4``, which is what lets
    :mod:`algorithms.transfer` copy weights in either direction with no
    projection.  The critic is given the same width purely for symmetry; it is
    randomly initialised in every condition of both studies.
    """
    return dict(
        net_arch=dict(pi=[HIDDEN, HIDDEN], vf=[HIDDEN, HIDDEN]),
        activation_fn=torch.nn.Tanh,
    )


# ---------------------------------------------------------------------------
# GFlowNet -- identical in both studies, both phases, both environments
# ---------------------------------------------------------------------------

GFLOWNET_HPARAMS = dict(
    n_envs=8,
    horizon=MAX_EPISODE_STEPS,
    action_repeat=ACTION_REPEAT,
    beta=2.0,                 # reward temperature: R(tau) = (1 + G)^beta
    hidden=HIDDEN,
    lr_policy=1e-3,
    lr_logz=1e-1,             # log Z takes a much larger step (standard GFN practice)
    grad_steps_per_round=20,
    batch_trajectories=16,
    buffer_capacity=2000,
    eps_start=0.50,           # eps-uniform behaviour policy, mixed with Uniform(9)
    eps_end=0.05,
    eps_anneal_frac=0.5,
    replay_fraction=0.5,      # reward-stratified replay; legitimate for TB
    max_grad_norm=10.0,
)


def gflownet_config():
    from algorithms.gflownet.agent import GFlowNetConfig
    return GFlowNetConfig(**GFLOWNET_HPARAMS)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def hyperparameter_fingerprint() -> str:
    """SHA-256 of this file. Written into every run's ``config.json``.

    Equal fingerprints across the two study directories is the machine-checkable
    form of "both researches used the same hyperparameters".
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def resolved() -> dict:
    """Everything above, flattened, for serialisation into a run record."""
    return dict(
        env_id=ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
        corruption=CORRUPTION,
        discrete=DISCRETE,
        action_repeat=ACTION_REPEAT,
        time_feature=TIME_FEATURE,
        hidden=HIDDEN,
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        decisions_per_episode=DECISIONS_PER_EPISODE,
        pretrain_steps=PRETRAIN_STEPS,
        adapt_steps=ADAPT_STEPS,
        seeds=list(SEEDS),
        n_seeds=len(SEEDS),
        eval_overlap_seeds=list(EVAL_OVERLAP_SEEDS),
        curve_every=CURVE_EVERY,
        curve_episodes=CURVE_EPISODES,
        eval_episodes=EVAL_EPISODES,
        eval_seed_base=EVAL_SEED_BASE,
        success_thresholds=list(SUCCESS_THRESHOLDS),
        ppo=dict(PPO_HPARAMS, n_envs=PPO_N_ENVS,
                 net_arch=[HIDDEN, HIDDEN], activation="Tanh"),
        gflownet=dict(GFLOWNET_HPARAMS),
        hyperparameters_sha256=hyperparameter_fingerprint(),
    )
