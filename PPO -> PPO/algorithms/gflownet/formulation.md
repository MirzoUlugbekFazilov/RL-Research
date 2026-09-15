# GFlowNet formulation for PointMaze

This answers §5 of the brief *before* any code was written. The six required
points are numbered below.

A GFlowNet is not a drop-in replacement for PPO. PPO maximises
`E[return]`; a GFlowNet learns a sampler whose terminal-state distribution is
**proportional to a reward function**. Applying one to a continuous-control
MDP requires the construction below, and every departure from stock PointMaze
is listed in "Declared modifications" at the end.

---

## 1. Chosen formulation: conditional **Trajectory Balance** (TB)

We use Trajectory Balance (Malkin et al., 2022) with a **conditional**
partition function (Bengio et al., *GFlowNet Foundations*, 2023).

For a complete trajectory `τ = (s₀, a₀, s₁, …, a_{T-1}, s_T)` the loss is

```
L(τ) = ( log Z_θ(c) + Σ_t log P_F(a_t | z_t ; θ)
                    − Σ_t log P_B(s_t | s_{t+1})
                    − log R(τ) )²
```

At the optimum the forward policy samples complete trajectories with
probability proportional to `R(τ)`, and `Z_θ(c)` estimates the total flow for
conditioning information `c`.

TB rather than Detailed Balance or Flow Matching because the reward here is
**terminal and trajectory-level** (the sparse return is only known once the
episode ends), which is exactly TB's setting — it needs no per-state flow
estimator.

### Why a GFlowNet is a *meaningful* comparator here, not a costume

Tiapkin et al. (2024) and Deleu et al. (2024) show GFlowNets on a
deterministic MDP are equivalent to **entropy-regularised (MaxEnt) RL** with a
particular reward. So this is not "PPO with extra steps": PPO converges toward
a *single* high-return mode, the GFlowNet toward the *full Boltzmann
distribution* over high-return trajectories. In a maze with several viable
routes and many viable timings, that is a real behavioural difference, and it
is the reason the two algorithms might respond differently to a corruption —
which is the actual research question.

---

## 2. How continuous actions are handled

PointMaze's action is a force `u ∈ Box(-1, 1, (2,))`. GFlowNets require a
countable action set at each state (the flow-matching constraint is a sum over
parents/children). We therefore **discretise**, and the discretisation is
chosen on control-theoretic grounds rather than for convenience:

> The plant is a point mass with linear damping and a **box** control
> constraint — a linear system. Pontryagin's maximum principle gives the
> **bang-bang principle**: a time-optimal control for such a system can be
> taken on the boundary of the constraint set almost everywhere.

The action set is therefore the 9 points of `{-1, 0, +1}²`: the maximal-
magnitude force in each of the 8 axis/diagonal directions, plus the null
action. This is the smallest set that keeps a time-optimal control
representable up to angular quantisation.

**This is a genuine restriction of the action set and it is not hidden.** It
is controlled three ways:

1. It is applied through one shared wrapper (`environments/wrappers.py::BangBangDiscreteActions`),
   so every algorithm that uses it gets exactly the same thing.
2. A **control condition runs PPO on this same `Discrete(9)` space**, so
   "PPO vs GFlowNet" can be read off an identical action space instead of
   across a continuous/discrete gap.
3. `sanity_checks.py::check_discretisation_is_not_crippling` measures what the
   restriction costs, by comparing PPO-continuous with PPO-discrete on the
   original environment.

Continuous-action GFlowNets do exist (Lahlou et al., 2023, measure-theoretic
GFlowNets over continuous spaces). We deliberately did **not** use one: it
would add an unvalidated continuous-flow implementation to a study whose
question is about transfer, not about GFlowNet architecture, and it would make
the PPO comparison harder to interpret rather than easier.

---

## 3. State representation

The GFlowNet's DAG state is the **trajectory prefix**. Because PointMaze
transitions are deterministic and Markov given the action sequence, the prefix
is summarised losslessly by the physical state plus the step index:

```
z_t = [ x, y, vx, vy, g_x, g_y, t / T ]  ∈ ℝ⁷
```

The first six entries are exactly the observation PPO receives; the seventh is
the normalised step index.

**Including `t` is load-bearing, not cosmetic.** It makes every prefix reachable
by exactly one path, so the "DAG" is a **tree** and the backward policy is
forced to `P_B(s_t | s_{t+1}) = 1`, i.e. `log P_B ≡ 0` and the `P_B` term
vanishes from the TB loss. This is not an approximation or a modelling
shortcut — it is a consequence of the construction, and it is the same
situation as a GFlowNet over sequences, where generation order is fixed. It
also means the terminal state is in bijection with the trajectory, so
"sample terminal states ∝ R" and "sample trajectories ∝ R" coincide.

## 4. Action representation

`P_F(· | z_t)` is a categorical distribution over the 9 discrete actions,
produced as a softmax over the 9 logits of an MLP `ℝ⁷ → 128 → 128 → 9`
(tanh activations, matching PPO's `net_arch=[64, 64]` in spirit and slightly
wider to offset the loss of a value head).

`log Z_θ(c)` is a separate MLP head `ℝ⁶ → 128 → 128 → 1` on the **initial
observation** `c = (s₀, g)`. It must be conditional because start and goal are
resampled every episode, so the total flow genuinely differs per instance; a
single scalar `Z` would be misspecified.

## 5. Reward / flow objective

The GFlowNet reward must be strictly positive (`log R` appears in the loss).
The environment's native sparse return is

```
G(τ) = Σ_t r_t ∈ {0, 1, …, 300}    (r_t = 1 iff ‖achieved − desired‖ ≤ 0.45)
```

— it counts the steps spent inside the goal region, because `continuing_task=True`
means the episode does not end on contact. We define

```
R(τ) = (1 + G(τ))^β        ⇒     log R(τ) = β · log(1 + G(τ))
```

* the `1 +` offset keeps `log R` finite for a failed trajectory (`log R = 0`)
  instead of `−∞`;
* `β` is a temperature exponent that sharpens the target distribution. `β = 1`
  samples ∝ dwell time; larger `β` concentrates on the best trajectories.
  It is selected once by pilot (`scripts/pilot_gflownet.py`) and then held
  fixed for every condition and both environments.

`R` is a **strictly monotone transform of the environment's own return**. It
changes the sampling temperature, never the task: no reward shaping, no
distance bonus, no goal-relabelling. Both algorithms are *evaluated* on the
native return and the native success criterion.

## 6. Why this is appropriate for PointMaze

* The reward is terminal and trajectory-level → TB's native setting.
* There are genuinely many distinct successful trajectories (two route
  families around the barrier, and many timings), so "sample ∝ reward" is a
  meaningful objective and not a degenerate one with a single answer.
* The action space discretises on a principled control-theoretic basis rather
  than an arbitrary grid.
* The state is low-dimensional and fully observed, so the tree construction is
  exact rather than an approximation over a partially observed prefix.

---

## Training: off-policy, as GFlowNets are meant to be used

TB is valid for trajectories drawn from **any** full-support behaviour policy —
that is a core advantage of GFlowNets over on-policy PG. We use

* behaviour policy `(1 − ε)·P_F + ε·Uniform(9)`, with `ε` annealed 0.5 → 0.05,
  guaranteeing full support;
* a replay buffer of past trajectories, with each gradient step drawn half
  from fresh rollouts and half from replay. Stored trajectories keep their
  states/actions/`log R`; `log P_F` is **recomputed under current θ**, which is
  what makes off-policy reuse correct here.

## Declared modifications, in full

| # | Modification | Applies to | Control |
|---|---|---|---|
| 1 | Action space discretised to `Discrete(9)` | GFlowNet (+ PPO-discrete controls) | `ppo_discrete_*` and `ppo_bangbang_scratch` run PPO on the identical space. Measured cost to PPO: **−0.08** success (it *helps*). |
| 2 | Step index `t/T` appended to the network input | GFlowNet only | Gives the tree property, hence `P_B ≡ 1` exactly. PPO's feed-forward policy is time-agnostic and gets `gamma` instead. |
| 3 | Reward mapped `G ↦ (1+G)^β`, β = 2 | GFlowNet only | Strictly monotone in the environment's own return. Evaluation uses native `G` for both algorithms. |
| 4 | Action repeat `k = 5` | GFlowNet (+ `ppo_discrete_*`) | Applied identically to both; budget still counted in **environment steps**, so a repeat agent gets *fewer decisions*, never more interaction. Measured cost to PPO: **+0.10** success. `ppo_bangbang_scratch` (k=1) separates this from modification 1. |

Nothing else about the environment, reward, horizon, success criterion, or
evaluation protocol differs between the algorithms.

## Pilot results (`scripts/pilot_gflownet.py`, original environment only)

`β` and `k` were selected once, on the **original** environment, then frozen
for every condition and both environments — so the choice cannot have been
tuned to favour either transfer condition.

| β | repeat | decisions/episode | deterministic success | stochastic success |
|---:|---:|---:|---:|---:|
| 1 | 1 | 300 | 0.350 | 0.470 |
| 2 | 1 | 300 | 0.770 | 0.620 |
| 4 | 1 | 300 | 0.750 | 0.720 |
| 1 | 5 | 60 | 0.680 | 0.610 |
| **2** | **5** | **60** | **0.890** | 0.690 |
| 4 | 5 | 60 | 0.800 | 0.860 |

Selected on deterministic success (the primary protocol): **β = 2, k = 5**.
The action-repeat rows confirm the motivation in modification 4 — shortening
the TB trajectory from 300 to 60 decisions materially reduces the
gradient-variance problem that Trajectory Balance is known to have on long
trajectories.

Sanity check §16.3 verifies the formulation learns the task at all: 0.81
deterministic success at 300k environment steps, against a 0.26 random floor
and 0.85 for PPO.
