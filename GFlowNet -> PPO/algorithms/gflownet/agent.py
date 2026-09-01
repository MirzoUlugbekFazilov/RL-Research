"""Conditional Trajectory-Balance GFlowNet behind the shared Agent interface.

Design decisions and their justification live in ``formulation.md``.  The
fine-tuning contract mirrors PPO's exactly: :meth:`GFlowNetAgent.finetune_from`
loads a checkpoint and swaps the environment, never re-initialising the
network, and records a parameter fingerprint so the sanity checks can prove it.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from environments.corruptions import DEFAULT_CORRUPTION
from environments.factory import make_env

from .model import GFlowNetPolicy


@dataclass
class GFlowNetConfig:
    """Hyperparameters. Identical across every condition and both environments."""

    n_envs: int = 8               # parallel rollout workers
    horizon: int = 300            # env steps per episode (PointMaze TimeLimit)
    action_repeat: int = 5        # env steps per decision; selected by pilot
    beta: float = 2.0             # reward temperature: R = (1+G)^beta; by pilot
    hidden: int = 128
    lr_policy: float = 1e-3
    lr_logz: float = 1e-1         # log Z needs a much larger step (standard GFN practice)
    grad_steps_per_round: int = 20
    batch_trajectories: int = 16  # trajectories per gradient step
    buffer_capacity: int = 2000
    eps_start: float = 0.50       # exploration mixture with Uniform(9)
    eps_end: float = 0.05
    eps_anneal_frac: float = 0.5  # fraction of the budget over which eps decays
    replay_fraction: float = 0.5  # share of each batch drawn from the buffer
    max_grad_norm: float = 10.0

    @property
    def decisions_per_episode(self) -> int:
        return self.horizon // self.action_repeat


@dataclass
class _Trajectory:
    z: np.ndarray        # (T, obs_dim + 1) network inputs, includes t/T
    actions: np.ndarray  # (T,) int64
    cond: np.ndarray     # (obs_dim,) initial observation
    log_reward: float    # beta * log(1 + G)
    ret: float           # native sparse return G
    success: bool


class _ReplayBuffer:
    """Reward-stratified replay.

    Successful and unsuccessful trajectories are kept in separate deques and
    sampled 50/50 when both are non-empty.  This is legitimate for TB (unlike
    for an on-policy gradient): the TB optimum is independent of the
    distribution the training trajectories are drawn from, provided that
    distribution has full support -- which the eps-uniform behaviour policy
    guarantees.  Under a sparse reward, uniform replay would otherwise show the
    network almost no successful trajectories.
    """

    def __init__(self, capacity: int, rng: random.Random):
        self.success: deque[_Trajectory] = deque(maxlen=capacity // 2)
        self.failure: deque[_Trajectory] = deque(maxlen=capacity // 2)
        self.rng = rng

    def add(self, traj: _Trajectory) -> None:
        (self.success if traj.success else self.failure).append(traj)

    def __len__(self) -> int:
        return len(self.success) + len(self.failure)

    def sample(self, n: int) -> list[_Trajectory]:
        if not len(self):
            return []
        pools = [p for p in (self.success, self.failure) if p]
        out = []
        for i in range(n):
            pool = pools[i % len(pools)]
            out.append(self.rng.choice(pool))
        return out


class GFlowNetAgent:
    algo = "gflownet"
    uses_discrete_actions = True
    #: The GFlowNet appends ``t/T`` to the observation *itself* (see ``act``
    #: and ``_collect``), so its environment must not also supply it. A PPO
    #: agent initialised from these weights gets the same input from
    #: :class:`environments.wrappers.TimeFeature` instead.
    uses_time_feature = False

    def __init__(self, net: GFlowNetPolicy, variant: str, corruption: str,
                 seed: int, cfg: GFlowNetConfig):
        self.net = net
        self.variant = variant
        self.corruption = corruption
        self.seed = seed
        self.cfg = cfg
        self.num_timesteps = 0
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._opt = torch.optim.Adam(
            [
                {"params": list(net.policy_parameters()), "lr": cfg.lr_policy},
                {"params": list(net.partition_parameters()), "lr": cfg.lr_logz},
            ]
        )
        self._buffer = _ReplayBuffer(cfg.buffer_capacity, self._rng)
        self._envs: list | None = None
        # per-episode state used by act() during evaluation
        self._eval_t = 0
        self.train_log: list[dict] = []

    # -- construction ------------------------------------------------------
    @classmethod
    def fresh(cls, variant: str = "original", *, corruption: str = DEFAULT_CORRUPTION,
              seed: int = 0, cfg: GFlowNetConfig | None = None) -> "GFlowNetAgent":
        cfg = cfg or GFlowNetConfig()
        torch.manual_seed(seed)
        net = GFlowNetPolicy(obs_dim=6, n_actions=9, hidden=cfg.hidden)
        return cls(net, variant, corruption, seed, cfg)

    @classmethod
    def finetune_from(cls, checkpoint, variant: str = "original", *,
                      corruption: str = DEFAULT_CORRUPTION, seed: int = 0,
                      cfg: GFlowNetConfig | None = None) -> "GFlowNetAgent":
        """Load a checkpoint and point it at a new environment.

        The network weights *and* the Adam moments are restored, matching what
        ``PPO.load`` does, so the two algorithms' fine-tuning conditions differ
        only in the algorithm.
        """
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = cfg or GFlowNetConfig(**ckpt["config"])
        net = GFlowNetPolicy(obs_dim=6, n_actions=9, hidden=cfg.hidden)
        net.load_state_dict(ckpt["state_dict"])
        agent = cls(net, variant, corruption, seed, cfg)
        if ckpt.get("optimizer") is not None:
            try:
                agent._opt.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass  # shape mismatch cannot happen here, but never block a run
        agent._loaded_from = str(checkpoint)
        agent._fingerprint_at_load = agent.parameter_fingerprint()
        return agent

    @classmethod
    def from_ppo(cls, checkpoint, variant: str = "original", *,
                 corruption: str = DEFAULT_CORRUPTION, seed: int = 0,
                 cfg: GFlowNetConfig | None = None) -> "GFlowNetAgent":
        """Start the GFlowNet from a **PPO** checkpoint and point it at ``variant``.

        This is Study A's cross-algorithm fine-tuning contract, and the exact
        mirror of :meth:`algorithms.ppo.agent.PPOAgent.from_gflownet`: PPO
        trains on the corrupted environment, and the GFlowNet continues from
        its actor on the original one.  The forward policy ``pf`` is
        overwritten with PPO's actor weights and the copy is verified
        numerically; the ``log Z`` head is necessarily fresh, because PPO has
        no partition function (its critic is a value function -- see
        :mod:`algorithms.transfer` for why copying it in would be a category
        error).

        The PPO checkpoint must have been trained on the shared interface --
        ``Discrete(9)``, ``action_repeat=5``, the ``t/T`` feature, and a
        ``[128, 128]`` Tanh actor -- or the shapes will not match and the
        transfer raises rather than silently misaligning a layer.

        The resulting agent carries ``transfer_report``, which the runner
        writes into the run's ``transfer.json``.
        """
        from stable_baselines3 import PPO

        from algorithms.transfer import transfer_ppo_policy_to_gflownet

        cfg = cfg or GFlowNetConfig()
        torch.manual_seed(seed)
        net = GFlowNetPolicy(obs_dim=6, n_actions=9, hidden=cfg.hidden)

        model = PPO.load(Path(checkpoint), device="cpu")
        agent = cls(net, variant, corruption, seed, cfg)
        agent.transfer_report = transfer_ppo_policy_to_gflownet(
            net, model.policy, obs_space=model.observation_space, seed=seed,
        )
        agent.transfer_report["source_checkpoint"] = str(checkpoint)
        agent.transfer_report["pretrain_steps_in_checkpoint"] = (
            int(model.num_timesteps) * cfg.action_repeat
        )
        agent._loaded_from = str(checkpoint)
        agent._fingerprint_at_load = agent.parameter_fingerprint()
        return agent

    @classmethod
    def load(cls, checkpoint, variant: str = "original", *,
             corruption: str = DEFAULT_CORRUPTION, seed: int = 0) -> "GFlowNetAgent":
        return cls.finetune_from(checkpoint, variant, corruption=corruption, seed=seed)

    # -- Agent protocol ----------------------------------------------------
    @property
    def action_repeat(self) -> int:
        """Env steps per decision; the evaluator builds a matching env."""
        return self.cfg.action_repeat

    def on_episode_start(self) -> None:
        self._eval_t = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """One decision for the evaluator.

        Tracks its own decision index because the network input needs ``t/T``.
        The action *repeat* is performed by the environment wrapper, so one
        call to ``act`` is exactly one decision in both training and
        evaluation.
        """
        frac = self._eval_t / self.cfg.decisions_per_episode
        z = torch.as_tensor(
            np.concatenate([np.asarray(obs, dtype=np.float32), [frac]]),
            dtype=torch.float32,
        ).unsqueeze(0)
        logits = self.net(z)[0]
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1).numpy()
            action = int(self._np_rng.choice(len(probs), p=probs))
        self._eval_t += 1
        return action

    def freeze(self) -> None:
        self.net.eval()
        self.on_episode_start()

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "optimizer": self._opt.state_dict(),
                "config": asdict(self.cfg),
                "num_timesteps": self.num_timesteps,
                "variant": self.variant,
                "corruption": self.corruption,
                "seed": self.seed,
            },
            path,
        )

    def parameter_fingerprint(self) -> str:
        h = hashlib.sha256()
        with torch.no_grad():
            for name, p in sorted(self.net.state_dict().items()):
                h.update(name.encode())
                h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
        return h.hexdigest()[:16]

    def close(self) -> None:
        for e in self._envs or []:
            e.close()
        self._envs = None

    # -- training ----------------------------------------------------------
    def _ensure_envs(self):
        if self._envs is None:
            self._envs = [
                make_env(self.variant, corruption=self.corruption, discrete=True,
                         action_repeat=self.cfg.action_repeat,
                         seed=self.seed * 1000 + i)
                for i in range(self.cfg.n_envs)
            ]
        return self._envs

    def _epsilon(self, total_timesteps: int) -> float:
        cfg = self.cfg
        span = max(1.0, cfg.eps_anneal_frac * total_timesteps)
        frac = min(1.0, self._steps_this_call / span)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    @torch.no_grad()
    def _collect(self, eps: float) -> list[_Trajectory]:
        """Roll out ``n_envs`` full episodes under the eps-uniform behaviour policy."""
        cfg = self.cfg
        envs = self._ensure_envs()
        n, D = len(envs), cfg.decisions_per_episode

        obs = np.stack([e.reset()[0] for e in envs]).astype(np.float32)
        cond = obs.copy()
        zs = np.zeros((n, D, obs.shape[1] + 1), dtype=np.float32)
        acts = np.zeros((n, D), dtype=np.int64)
        rets = np.zeros(n, dtype=np.float64)
        succ = np.zeros(n, dtype=bool)

        for d in range(D):
            z = np.concatenate([obs, np.full((n, 1), d / D, dtype=np.float32)], axis=1)
            zs[:, d] = z
            logits = self.net(torch.as_tensor(z))
            probs = torch.softmax(logits, dim=-1).numpy()
            probs = (1.0 - eps) * probs + eps / probs.shape[1]
            probs /= probs.sum(axis=1, keepdims=True)
            a = np.array([self._np_rng.choice(probs.shape[1], p=p) for p in probs])
            acts[:, d] = a

            for i, e in enumerate(envs):
                o, r, term, trunc, info = e.step(int(a[i]))
                obs[i] = o
                rets[i] += float(r)
                succ[i] = succ[i] or bool(info.get("success", False))
                inner = int(info.get("repeat_steps", 1))
                self.num_timesteps += inner
                self._steps_this_call += inner

        out = []
        for i in range(n):
            out.append(
                _Trajectory(
                    z=zs[i], actions=acts[i], cond=cond[i],
                    log_reward=float(cfg.beta * np.log1p(rets[i])),
                    ret=float(rets[i]), success=bool(succ[i]),
                )
            )
        return out

    def _tb_loss(self, batch: list[_Trajectory]) -> torch.Tensor:
        """Trajectory Balance: ``(log Z(c) + sum_t log P_F − log R)^2``.

        The ``log P_B`` term is absent because the DAG is a tree by
        construction (the step index is part of the state), so ``P_B ≡ 1``.
        """
        z = torch.as_tensor(np.stack([t.z for t in batch]))               # (B, D, 7)
        a = torch.as_tensor(np.stack([t.actions for t in batch]))         # (B, D)
        cond = torch.as_tensor(np.stack([t.cond for t in batch]))         # (B, 6)
        log_r = torch.as_tensor(
            np.array([t.log_reward for t in batch], dtype=np.float32)
        )

        log_pf = self.net.log_pf(z, a).sum(dim=1)                         # (B,)
        log_z = self.net.log_partition(cond)                              # (B,)
        return ((log_z + log_pf - log_r) ** 2).mean()

    def learn(self, total_timesteps: int, callback=None):
        """Train for ``total_timesteps`` **environment steps**.

        The budget is counted in environment steps -- the same currency PPO
        uses -- so the two algorithms are directly comparable regardless of how
        many gradient updates each performs internally.
        """
        cfg = self.cfg
        self.net.train()
        self._steps_this_call = 0
        target = total_timesteps
        round_idx = 0

        while self._steps_this_call < target:
            eps = self._epsilon(target)
            fresh = self._collect(eps)
            for t in fresh:
                self._buffer.add(t)

            n_replay = int(cfg.batch_trajectories * cfg.replay_fraction)
            n_fresh = cfg.batch_trajectories - n_replay
            losses = []
            for _ in range(cfg.grad_steps_per_round):
                batch = [self._rng.choice(fresh) for _ in range(n_fresh)]
                batch += self._buffer.sample(n_replay)
                loss = self._tb_loss(batch)
                self._opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self._opt.step()
                losses.append(float(loss.item()))

            round_idx += 1
            self.train_log.append(
                dict(
                    round=round_idx,
                    steps=self.num_timesteps,
                    steps_this_call=self._steps_this_call,
                    eps=eps,
                    tb_loss=float(np.mean(losses)),
                    behaviour_return=float(np.mean([t.ret for t in fresh])),
                    behaviour_success=float(np.mean([t.success for t in fresh])),
                )
            )
            if callback is not None:
                callback(self, self._steps_this_call)

        self.net.eval()
        return self
